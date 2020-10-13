#include <vector>
#include <mutex>
#include <unordered_map>
#include <iostream>
#include <atomic>
#include <algorithm>
#include <sstream>
#include <fstream>

#include <boost/dynamic_bitset.hpp>

#include "omp.h"
#include "worker.h"
#include "radix_partition.cpp"
#include "murmur3.h"
#include "hash_join_chunk_chaining.h"

using namespace std;
using namespace boost;

namespace mt_atomic_chunk {

// Constants & Setup vars
static const long long n_sec = 1000000000L;
static const int tpl_per_chunk  = 64;
static unsigned ht_size = 0;     /* set in init */
static unsigned ht_mask = 0;     /* set in init */
static unsigned output_mask = 0; /* set in init */

// Parameters HT Generation 
static const double load_factor = 0.5;
static const int cleanup_threshold = 1;
static const unsigned output_size = 2097152; /* 2^21 */


/* keep track of processed tuples */
std::atomic<uint32_t> processed_tuples;

struct ht{
	atomic<uint32_t> counter;
	uint64_t address;
}; // 16 Byte

struct chunk_R{
	std::chrono::nanoseconds t_ns;
        x_t             x; /* key */
        y_t             y; /* value */
	unsigned        r; /* index */
}; // 32 Byte


struct chunk_S{
	std::chrono::nanoseconds t_ns;
        a_t             a; /* key */
        b_t             b; /* value */
	unsigned        s; /* index */
}; // 32 Byte


ht *hmR;
ht *hmS;

// Cleanup Bitmap
uint64_t *cleanupR;
uint64_t *cleanupS;

// Output Tuples Buffer
uint32_t *output;

/* return p with the smallest k for that holds: p = 2^k > n */
unsigned next_largest_power2(unsigned n){
	n--;
	n |= n >> 1;   // Divide by 2^k for consecutive doublings of k up to 32,
	n |= n >> 2;   // and then 'or' the results.
	n |= n >> 4;
	n |= n >> 8;
	n |= n >> 16;
	n++;
	return n;
}

void init(master_ctx_t *ctx){

	/* Init output tuple buffer */
	cout << "# Use a hast table load factor of " << load_factor << "\n";
	
	if (ctx->num_tuples_R != ctx->num_tuples_S){
		cout << "# WARNING: ctx->num_tuples_R != ctx->num_tuples_S. May lead to problems in hash table generation.\n";
	}

	unsigned tpl_per_window = ctx->window_size_R * ctx->rate_R;
	cout << "# Expected number of tuples per window: " <<  tpl_per_window << "\n";

	ht_size = next_largest_power2(tpl_per_window / tpl_per_chunk);
	if (ht_size < (tpl_per_window / tpl_per_chunk) / load_factor){
	       	ht_size = ht_size << 1;
	}
	ht_mask = ht_size - 1;

	cout << "# Use a hash table size of " << ht_size << " chunks with " << tpl_per_chunk << " tuples\n";
	
	unsigned chunk_size = tpl_per_chunk * sizeof(chunk_R);
	cout << "# The chunk size is " << chunk_size << " B\n";
	cout << "# Total hash table size is " 
		<< (long)ht_size * tpl_per_chunk * chunk_size / 1048576 /*MB*/ << " MB\n";

	assert(ht_size != 0);
	assert(ht_size % 2 == 0);
	assert(ht_mask != 0);

        posix_memalign((void**)&hmR, 64, ht_size*sizeof(ht));
        posix_memalign((void**)&hmS, 64, ht_size*sizeof(ht));

	for (unsigned i = 0; i < ht_size; i++){
		chunk_R *chunk;
		posix_memalign((void**)&chunk, 64, chunk_size);
		hmR[i].address = (uint64_t)chunk;
		hmR[i].counter = 0;
	}
	for (unsigned i = 0; i < ht_size; i++){
		chunk_S *chunk;
		posix_memalign((void**)&chunk, 64, chunk_size);
		hmS[i].address = (uint64_t)chunk;
		hmS[i].counter = 0;
	}

	/* Init output tuple buffer */
	output_mask = output_size - 1;
	assert(output_size % 2 == 0);
	assert(output_mask != 0);

	unsigned output_buffersize = output_size * sizeof(uint32_t) * 2;
	cout << "# Total output buffer size is " 
		<< output_buffersize / 1048576 /*MB*/ << " MB\n";

	posix_memalign((void**)&output, 64, output_buffersize);
	
	/* Init clean-up bitmap */
	//posix_memalign((void**)&cleanupR, 64, cacheline_size*ht_size)
	//posix_memalign((void**)&cleanupS, 64, cacheline_size*ht_size)
}

/* Calculate Latency for statistic output
 * INFLUENCES MULTITHREADED PERFORMANCE SIGNIFICANTLY */
inline
void calc_latency (worker_ctx_t *w_ctx, chunk_S chunk, unsigned index)
{
	/* Choose the older tuple to calc the latency.
	 * We can assume that the tuple in ht is always the older one */
        auto now = std::chrono::steady_clock::now();
        auto i = (std::chrono::duration_cast <std::chrono::nanoseconds>(now -  w_ctx->stats.start_time) - chunk.t_ns);
	#pragma omp critical
	{
	w_ctx->stats.summed_latency = std::chrono::duration_cast <std::chrono::nanoseconds>(w_ctx->stats.summed_latency + i);
	/*ofstream file;
	file.open ("latency_dump.txt",std::ios_base::app);
	file << r << " " << s << " " <<  i.count() 
		<< " " <<r_get_tns(ctx,r).count() << " " 
		<< std::chrono::duration_cast <std::chrono::nanoseconds>(now - w_ctx->stats.start_time).count() << "\n";
	file.close();*/
	}
}

inline
void calc_latency (worker_ctx_t *w_ctx, chunk_R chunk, unsigned index)
{
	/* Choose the older tuple to calc the latency.
	 * We can assume that the tuple in ht is always the older one */
        auto now = std::chrono::steady_clock::now();
        auto i = (std::chrono::duration_cast <std::chrono::nanoseconds>(now -  w_ctx->stats.start_time) - chunk.t_ns);
	#pragma omp critical
	{
	w_ctx->stats.summed_latency = std::chrono::duration_cast <std::chrono::nanoseconds>(w_ctx->stats.summed_latency + i);
	/*ofstream file;
	file.open ("latency_dump.txt",std::ios_base::app);
	file << r << " " << s << " " <<  i.count() 
		<< " " <<r_get_tns(ctx,r).count() << " " 
		<< std::chrono::duration_cast <std::chrono::nanoseconds>(now - w_ctx->stats.start_time).count() << "\n";
	file.close();*/
	}
}

void process_r_ht_cpu(master_ctx_t *ctx, worker_ctx_t *w_ctx){

	Timer::Timer timer = Timer::Timer();
	auto start_time = timer.now();

        // Build R HT
#pragma omp parallel for 
        for (unsigned r = *(w_ctx->r_processed);
            r < *(w_ctx->r_processed) + w_ctx->r_batch_size;
            r++){
                const uint32_t k = w_ctx->R.x[r];

		/* Linear */
		/*uint32_t hash = k;
		hash = (hash % ht_size);*/

		/* Murmur*/
		uint32_t hash;
		MurmurHash3_x86_32((void*)&k, sizeof(uint32_t), 1, &hash);
		hash = (hash & ht_mask);
		
		uint32_t tpl_cntr = hmR[hash].counter.fetch_add(1,std::memory_order_relaxed);

		if (tpl_cntr >= tpl_per_chunk) {
			cout << hash << " " << tpl_cntr << "\n";
			cout << "Chunk full at index: " << hash  << "\n";
			for (unsigned i = 0; i < tpl_cntr*2 ; i++) {
				cout << ((chunk_R*)hmR[hash].address)[i].x << " "
					<< ((chunk_R*)hmR[hash].address)[i].y << " "
					<< ((chunk_R*)hmR[hash].address)[i].t_ns.count() << "\n";
			}

			/* Write HT dump */
			ofstream file;
			file.open ("ht_dump.txt");
			for (unsigned i = 0; i < ht_size; i++) 
				file << hmR[i].counter.load(std::memory_order_relaxed) << "\n";
			file.close();
			exit(1);
		}

		((chunk_R*)hmR[hash].address)[tpl_cntr].x = k; // key
		((chunk_R*)hmR[hash].address)[tpl_cntr].y = w_ctx->R.y[r]; // value
		((chunk_R*)hmR[hash].address)[tpl_cntr].t_ns = w_ctx->R.t_ns[r]; // ts
		((chunk_R*)hmR[hash].address)[tpl_cntr].r = r; // index
        }


#ifdef DEBUG
	auto end_time = timer.now();
	w_ctx->stats.runtime_proc += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        cout << "Build R " <<
		std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() << "\n";
	start_time = timer.now();
#endif

        // Probe S HT
	unsigned emitted_sum = 0;
        unsigned to_delete_sum = 0;
	dynamic_bitset<> to_delete_bitmap(ht_size);
#pragma omp parallel for reduction(+: emitted_sum, to_delete_sum)
        for (unsigned r = *(w_ctx->r_processed);
            r < *(w_ctx->r_processed) + w_ctx->r_batch_size;
            r++){
                const uint32_t k = w_ctx->R.x[r];

		/* Linear*/
		/*uint32_t hash = k;
		hash = (hash % ht_size);*/

		/* Murmur */
		uint32_t hash;
		MurmurHash3_x86_32((void*)&k, sizeof(uint32_t), 1, &hash);
		hash = (hash & ht_mask);
		
		unsigned emitted_tuples = 0;
		unsigned to_delete_tuples = 0;
		vector<tuple<uint32_t, uint32_t>> out_tuples;
		const uint32_t tpl_cntr = hmS[hash].counter.load(std::memory_order_relaxed);
		if (tpl_cntr != 0){ // Not empty
			const chunk_S *chunk = (chunk_S*) hmS[hash].address; // head
			for (int j = 0; j < tpl_cntr; j++){
				if ((chunk[j].t_ns.count() + w_ctx->window_size_S * n_sec)
					> r_get_tns(ctx,r).count()) { // Valid
					if (chunk[j].a == k) { // match
						unsigned s = chunk[j].s;
						out_tuples.push_back(tuple<uint32_t, uint32_t>(r,s));
						emitted_tuples++;
						calc_latency(w_ctx, *chunk, j);
					}
				} else { // Invalid
					to_delete_bitmap[hash] = 1;
					to_delete_tuples++;
				}
			}
		}
	
		// Write output tuples
		int j = 0;
		for (auto i : out_tuples) {
			//if ((processed_tuples+emitted_sum+out_tuples.size()) >= output_size){
			//	cout << "output buffer full\n";
			//	exit(0);
			//}

			/* Bounded by output buffer size */
			output[((processed_tuples+emitted_sum+j) & output_mask)*2]   = get<0>(i);
			output[((processed_tuples+emitted_sum+j) & output_mask)*2+1] = get<1>(i);
			j++;
		}
		
		to_delete_sum += to_delete_tuples;
                emitted_sum += emitted_tuples;
        }

	processed_tuples += emitted_sum;
        *(w_ctx->r_processed) += w_ctx->r_batch_size;

#ifdef DEBUG
	end_time = timer.now();
	w_ctx->stats.runtime_proc += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        cout << "Probe S "
		<< std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() << "\n";
	start_time = timer.now();
#endif

	/* Clean-up S HT */
	if (to_delete_sum > cleanup_threshold) {
#pragma omp parallel for
		for (size_t i = 0; i < ht_size; i++){
			if (to_delete_bitmap.test(i)){
				uint32_t tpl_cnt = hmS[i].counter.load(std::memory_order_relaxed);
				chunk_S *chunk = (chunk_S*) hmS[i].address; // head
				for(size_t j = 0; j < tpl_cnt; j++) { // non-empty
					if ((chunk[j].t_ns.count() + w_ctx->window_size_S * n_sec)
						< r_get_tns(ctx,*(w_ctx->r_processed)).count()) {
						// Remove + Move
						for (int u = 0, l = 0; u < tpl_cnt; u++) {
							if ((u != j || u != j+1) && u != l) {
								chunk[l].t_ns = chunk[u].t_ns;
								chunk[l].a = chunk[u].a;
								chunk[l].b = chunk[u].b;
								chunk[l].s = chunk[u].s;
								l++;
							}
						}
						tpl_cnt--;
						hmS[i].counter--; // Update tpl counter
					}
				}
			}
		}
	}

#ifdef DEBUG
	end_time = timer.now();
	w_ctx->stats.runtime_proc += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        cout << "Clean up S " << 
		std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() << "\n";
#else
	auto end_time = timer.now();
	w_ctx->stats.runtime_proc += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
#endif
}

void process_s_ht_cpu(master_ctx_t *ctx, worker_ctx_t *w_ctx){

	Timer::Timer timer = Timer::Timer();
	auto start_time = timer.now();

	// Build S HT
#pragma omp parallel for 
	for (unsigned s = *(w_ctx->s_processed);
            s < *(w_ctx->s_processed) + w_ctx->s_batch_size;
            s++){
                const uint32_t k = w_ctx->S.a[s];

		/* Linear*/
		/*uint32_t hash = k;
		hash = (hash % ht_size);*/

		/* Murmur*/
		uint32_t hash;
		MurmurHash3_x86_32((void*)&k, sizeof(uint32_t), 1, &hash);
		hash = (hash & ht_mask);
		
		uint32_t tpl_cntr = hmS[hash].counter.fetch_add(1,std::memory_order_relaxed);

		if (tpl_cntr >= tpl_per_chunk) {
			cout << hash << " " << tpl_cntr << "\n";
			cout << "Chunk full at index: " << hash  << "\n";
			for (int i = 0; i < tpl_cntr*2 ; i++) {
				cout << ((chunk_S*)hmS[hash].address)[i].a << " "
					<< ((chunk_S*)hmS[hash].address)[i].b << " "
					<< ((chunk_S*)hmS[hash].address)[i].t_ns.count() << "\n";
			}

			/* Write HT dump */
			ofstream file;
			file.open ("ht_dump.txt");
			for (int i = 0; i < ht_size; i++) 
				file << hmS[i].counter.load(std::memory_order_relaxed) << "\n";
			file.close();

			exit(1);
		}
		
		((chunk_S*)hmS[hash].address)[tpl_cntr].a = k; // key
		((chunk_S*)hmS[hash].address)[tpl_cntr].b = w_ctx->S.b[s]; // value
		((chunk_S*)hmS[hash].address)[tpl_cntr].t_ns = w_ctx->S.t_ns[s]; // ts
		((chunk_S*)hmS[hash].address)[tpl_cntr].s = s; // index
        }


#ifdef DEBUG
	auto end_time = timer.now();
        cout << "Build S "
		<< std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() << "\n";
	w_ctx->stats.runtime_proc += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
	start_time = timer.now();
#endif


        // Probe R HT
	auto emitted_sum = 0;
	auto to_delete_sum = 0;
	dynamic_bitset<> to_delete_bitmap(ht_size);
#pragma omp parallel for reduction(+: emitted_sum, to_delete_sum)
        for (unsigned s = *(w_ctx->s_processed);
            s < *(w_ctx->s_processed) + w_ctx->s_batch_size;
            s++){
                const uint32_t k = w_ctx->S.a[s];

		/* Linear
		uint32_t hash = k;
		hash = (hash % ht_size);*/

		/* Murmur */
		uint32_t hash;
		MurmurHash3_x86_32((void*)&k, sizeof(uint32_t), 1, &hash);
		hash = (hash & ht_mask);
		
		unsigned emitted_tuples = 0;
		unsigned to_delete_tuples = 0;
		vector<tuple<uint32_t, uint32_t>> out_tuples;
		const uint32_t tpl_cntr = hmR[hash].counter.load(std::memory_order_relaxed);
		if (tpl_cntr != 0) { // Not empty
			const chunk_R *chunk = (chunk_R*) hmR[hash].address; // head
			for (unsigned j = 0; j < tpl_cntr; j++){
				if ((chunk[j].t_ns.count() + w_ctx->window_size_R * n_sec)
					> s_get_tns(ctx,s).count()) { // Valid
					if (chunk[j].x == k) { // match
						unsigned r = chunk[j].r;
						out_tuples.push_back(tuple<uint32_t, uint32_t>(r,s));
						emitted_tuples++;
						calc_latency(w_ctx, *chunk, j);
					} 
				} else { // Invalid
					to_delete_bitmap[hash] = 1;
					to_delete_tuples++;
				}
			}
		}

		// Write output tuples
		int j = 0;
		for (auto i : out_tuples) {
			//if ((processed_tuples+emitted_sum+out_tuples.size()) >= output_size){
			//	cout << "output buffer full\n";
			//	exit(0);
			//}
			/* Bounded by output buffer size */
			output[((processed_tuples+emitted_sum+j) & output_mask)*2]   = get<0>(i);
			output[((processed_tuples+emitted_sum+j) & output_mask)*2+1] = get<1>(i);
			j++;
		}
		
		to_delete_sum += to_delete_tuples;
		emitted_sum += emitted_tuples;
        }

	processed_tuples += emitted_sum;
        *(w_ctx->s_processed) += w_ctx->s_batch_size;

#ifdef DEBUG
	end_time = timer.now();
        cout << "Probe R " << 
		std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() << "\n";
	w_ctx->stats.runtime_proc += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
	start_time = timer.now();
#endif

	/* Clean-up R HT */
	//if (!to_delete_bitmap.none()) {
	if (to_delete_sum > cleanup_threshold) {
#pragma omp parallel for
		for (size_t i = 0; i < ht_size; i++){
			if (to_delete_bitmap.test(i)){
				uint32_t tpl_cnt = hmR[i].counter.load(std::memory_order_relaxed);
				chunk_R *chunk = (chunk_R*) hmR[i].address; // head
				for(size_t j = 0; j < tpl_cnt; j++) { // non-empty
					if ((chunk[j].t_ns.count() + w_ctx->window_size_R * n_sec)
						< s_get_tns(ctx,*(w_ctx->s_processed)).count()) {
						// Remove + Move
						for (int u = 0, l = 0; u < tpl_cnt; u++) {
							if ((u != j || u != j+1) && u != l) {
								chunk[l].t_ns = chunk[u].t_ns;
								chunk[l].x = chunk[u].x;
								chunk[l].y = chunk[u].y;
								chunk[l].r = chunk[u].r;
								l++;
							}
						}
						tpl_cnt--;
						hmR[i].counter--; // Update tpl counter
					}
				}
			}
		}
	}
	
#ifdef DEBUG
	end_time = timer.now();
	w_ctx->stats.runtime_proc += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        cout << "Clean up R " << 
		std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() << "\n";
#else
	auto end_time = timer.now();
	w_ctx->stats.runtime_proc += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
#endif
}
}
