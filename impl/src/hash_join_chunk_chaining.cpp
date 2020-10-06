#include <vector>
#include <mutex>
#include <unordered_map>
#include <iostream>
#include <atomic>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <bitset>

#include "omp.h"
#include "worker.h"
#include "radix_partition.cpp"
#include "murmur3.h"
#include "hash_join_chunk_chaining.h"

using namespace std;

namespace mt_atomic_chunk {
// do not change
static const long long n_sec = 1000000000L;
static const int uint64_t_size = 64;
static const int cacheline_size = 64;
static const int tpl_per_chunk  = uint64_t_size*1;

// Parameters
static const int ht_size = 2000000;
static const int cleanup_threshold = 1;
static const int output_buffer_tuples = 2000000;
static const int output_buffersize = output_buffer_tuples * sizeof(uint32_t) * 2;

// do not change
static const int chunk_size = tpl_per_chunk * sizeof(uint32_t) * 2;

/* keep track of processed tuples */
std::atomic<uint32_t> processed_tuples;

struct ht{
	atomic<uint32_t> counter;
	uint64_t address;
}; // 16 Byte

ht *hmR;
ht *hmS;

// Cleanup Bitmap
uint64_t *cleanupR;
uint64_t *cleanupS;

// Output Tuples Buffer
uint32_t *output;

void init_ht(){
	cout << "# Use a hash table size of " << ht_size << " chunks with " << tpl_per_chunk << " tuples\n";
	cout << "# Total hash table size is " << ht_size * tpl_per_chunk * sizeof(uint32_t) * 2 /*byte*/ 
		/ 1048576 /*MB*/ << " MB\n";

        posix_memalign((void**)&hmR, 64, ht_size*sizeof(ht));
        posix_memalign((void**)&hmS, 64, ht_size*sizeof(ht));

	for (int i = 0; i < ht_size; i++){
		uint32_t *chunk;
		posix_memalign((void**)&chunk, 64, chunk_size);
		hmR[i].address = (uint64_t)chunk;
		hmR[i].counter = 0;
	}
	for (int i = 0; i < ht_size; i++){
		uint32_t *chunk;
		posix_memalign((void**)&chunk, 64, chunk_size);
		hmS[i].address = (uint64_t)chunk;
		hmS[i].counter = 0;
	}

	// Init output tuples
	posix_memalign((void**)&output, 64, output_buffersize);
	
	// Init clean-up bitmap
	//posix_memalign((void**)&cleanupR, 64, cacheline_size*ht_size)
	//posix_memalign((void**)&cleanupS, 64, cacheline_size*ht_size)
}

void print_ht_entries(worker_ctx_t *w_ctx){
        for (int j = 0; j < ht_size; j++){
                for (int i = 0; i < hmS[j].counter; i++){
			fprintf (w_ctx->resultfile, "%d %d\n", 
					((uint64_t*) hmS[j].address)[i*2], 
					((uint64_t*) hmS[j].address)[i*2+1]);
		}
		fprintf (w_ctx->resultfile, "\n"); 

        }
 
	for (int j = 0; j < ht_size; j++){
                for (int i = 0; i < hmR[j].counter; i++){
			fprintf (w_ctx->resultfile, "%d %d\n", 
					((uint64_t*) hmR[j].address)[i*2], 
					((uint64_t*) hmR[j].address)[i*2+1]);
		}
		fprintf (w_ctx->resultfile, "\n"); 
        }
}

void printTop10() {
	// Top 10 entries:
	vector<int> top10 = {0,0,0,0,0,0,0,0,0,0};
	for (int j = 0; j < ht_size; j++){
		for (int a = 0; a < 10; a++){
			if (top10[a] < hmS[j].counter) {
				top10[a] = hmS[j].counter;
				break;
			}
		}
        }
	for (int a : top10){
		cout << a << "\n";
	}
	cout << "\n";
       	// Top 10 entries:
	top10 = {0,0,0,0,0,0,0,0,0,0};
	for (int j = 0; j < ht_size; j++){
		for (int a = 0; a < 10; a++){
			if (top10[a] < hmR[j].counter) {
				top10[a] = hmR[j].counter;
				break;
			}
		}
        }
	for (int a : top10){
		cout << a << "\n";
	}
        
}

void print_ht(worker_ctx_t *w_ctx){
        int i = 0;
        long z = 0;
        for (int j = 0; j < ht_size; j++){
                if (hmS[j].counter > 0)
                        i++;
                z += j % hmS[j].counter;
        }
        cout << "S: Used HT slots: " << i << " HT content hash: " << z << "\n";


        i= 0;
        z= 0;
        for (int j = 0; j < ht_size; j++){
                if (hmR[j].counter > 0)
                        i++;
                z += j % hmR[j].counter;

        }
        cout << "R: Used HT slots: " << i << " HT content hash: " << z << "\n";
	//printTop10();
	//print_ht_entries(w_ctx);
}


void calc_latency (worker_ctx_t *w_ctx, unsigned int r, unsigned int s)
{
	// Calculate Latency
	// INFLUENCES MULTITHREADED PERFORMANCE SIGNIFICANTLY:
        /* Choose the older tuple to calc the latency*/
        auto now = std::chrono::steady_clock::now();
        if (w_ctx->S.t_ns[s] < w_ctx->R.t_ns[r]){
                auto i = (std::chrono::duration_cast <std::chrono::nanoseconds>(now -  w_ctx->stats.start_time) - w_ctx->R.t_ns[r]);
                #pragma omp critical
                w_ctx->stats.summed_latency = std::chrono::duration_cast <std::chrono::nanoseconds>(w_ctx->stats.summed_latency + i);
        } else {
                auto i = (std::chrono::duration_cast <std::chrono::nanoseconds>(now -  w_ctx->stats.start_time) - w_ctx->S.t_ns[s]);
                #pragma omp critical
                w_ctx->stats.summed_latency = std::chrono::duration_cast <std::chrono::nanoseconds>(w_ctx->stats.summed_latency + i);
        }
}


inline void setBit(int index, unsigned char *bitarray) {
    bitarray[index/8] = bitarray[index/8] | 1 << 7-(index & 0x7);
}

void process_r_ht_cpu(master_ctx_t *ctx, worker_ctx_t *w_ctx){

	Timer::Timer timer = Timer::Timer();
	auto start_time = timer.now();

        // Build R HT
#pragma omp parallel for 
        for (unsigned r = *(w_ctx->r_processed);
            r < *(w_ctx->r_processed) + w_ctx->r_batch_size;
            r++){
                const uint32_t k = w_ctx->R.x[r] + w_ctx->R.y[r];
		

		/*if (k == 9898){
			cout <<  w_ctx->R.x[r] << " " <<  w_ctx->R.y[r] << "\n";
		}*/

		/* Linear */
		/*uint32_t hash = k;
		hash = (hash % ht_size);*/

		/* Murmur*/
		uint32_t hash;
		MurmurHash3_x86_32((void*)&k, sizeof(uint32_t), 1, &hash);
		hash = (hash % ht_size);
		
		uint32_t tpl_cntr = hmR[hash].counter.fetch_add(1,std::memory_order_relaxed);

		if (tpl_cntr >= tpl_per_chunk) {
			cout << hash << " " << tpl_cntr << "\n";
			cout << "Chunk full at index: " << hash  << "\n";
			for (int i = 0; i < tpl_cntr*2 ; i++)
				cout << ((uint32_t*) hmR[hash].address)[i] << "\n";
			/* Write HT dump */
			ofstream file;
			file.open ("ht_dump.txt");
			for (int i = 0; i < ht_size; i++) 
				file << hmR[i].counter.load(std::memory_order_relaxed) << "\n";
			file.close();
			exit(1);
		}

		((uint32_t*) hmR[hash].address)[tpl_cntr*2] = k; // key
		((uint32_t*) hmR[hash].address)[(tpl_cntr*2)+1] = r; // value
        }

	auto end_time = timer.now();
	w_ctx->stats.runtime_proc += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();

#ifdef DEBUG
        cout << "Build R " <<
		std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() << "\n";
#endif
	start_time = timer.now();

        // Probe S HT
	auto emitted_sum = 0;
        auto to_delete_sum = 0;
	bitset<ht_size> to_delete_bitmap;
#pragma omp parallel for reduction(+: emitted_sum, to_delete_sum)
        for (unsigned r = *(w_ctx->r_processed);
            r < *(w_ctx->r_processed) + w_ctx->r_batch_size;
            r++){
                const uint32_t k = w_ctx->R.x[r] + w_ctx->R.y[r];

		/* Linear*/
		/*uint32_t hash = k;
		hash = (hash % ht_size);*/

		/* Murmur */
		uint32_t hash;
		MurmurHash3_x86_32((void*)&k, sizeof(uint32_t), 1, &hash);
		hash = (hash % ht_size);
		
		unsigned emitted_tuples = 0;
		unsigned to_delete_tuples = 0;
		vector<tuple<uint32_t, uint32_t>> out_tuples;
		const uint32_t tpl_cntr = hmS[hash].counter.load(std::memory_order_relaxed);
		if (tpl_cntr != 0){ // Not empty
			const uint32_t *cur_chunk = (uint32_t*) hmS[hash].address; // head
			for (int j = 0; j < tpl_cntr; j++){
				const uint32_t s = cur_chunk[(j*2)+1];
				//cout << s_get_tns(ctx,s).count() + 
				//	w_ctx->window_size_S * n_sec
                                //        << " " <<  r_get_tns(ctx,r).count() << "\n";
				if ((s_get_tns(ctx,s).count() + w_ctx->window_size_S * n_sec)
					> r_get_tns(ctx,r).count()) { // Valid
					if (cur_chunk[j*2] == k) { // match
					//cout << "S: " << s << " " << k << "\n";
						out_tuples.push_back(tuple<uint32_t, uint32_t>(r,s));
						emitted_tuples++;
						calc_latency(w_ctx, r, s);
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
			//if ((processed_tuples+emitted_sum+out_tuples.size()) >= output_buffer_tuples){
			//	cout << "output buffer full\n";
			//	exit(0);
			//}

			/* Bounded by output buffer size */
			output[((processed_tuples+emitted_sum+j) % output_buffer_tuples)*2]   = get<0>(i);
			output[((processed_tuples+emitted_sum+j) % output_buffer_tuples)*2+1] = get<1>(i);
			j++;
		}
		
		to_delete_sum += to_delete_tuples;
                emitted_sum += emitted_tuples;
        }

	processed_tuples += emitted_sum;
	
        *(w_ctx->r_processed) += w_ctx->r_batch_size;

	end_time = timer.now();
	w_ctx->stats.runtime_proc += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();

#ifdef DEBUG
        cout << "Probe S "
		<< std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() << "\n";
#endif
	start_time = timer.now();

	//cout << "delete: " << to_delete_sum << " tuples\n";
	/* Clean-up S HT */
	if (to_delete_sum > cleanup_threshold) {
		//cout << "delete: " << to_delete_sum << " tuples\n";
#pragma omp parallel for
		for (size_t i = 0; i < ht_size; i++){
			if (to_delete_bitmap.test(i)){
				uint32_t tpl_cnt = hmS[i].counter.load(std::memory_order_relaxed);
				uint32_t *chunk = (uint32_t*) hmS[i].address; // head
				for(size_t j = 0; j < tpl_cnt; j++) { // non-empty
					const size_t s = chunk[(j*2)+1];
					if ((s_get_tns(ctx,s).count() + w_ctx->window_size_S * n_sec) 
						< r_get_tns(ctx,*(w_ctx->r_processed)).count()) {
						// Remove + Move
						for (int u = 0, l = 0; u < tpl_cnt; u++) {
							if ((u != j || u != j+1) && u != l) {
								chunk[l*2] = chunk[u*2];
								chunk[l*2+1] = chunk[u*2+1];
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

	end_time = timer.now();
	w_ctx->stats.runtime_proc += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
#ifdef DEBUG
        cout << "Clean up S " << 
		std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() << "\n";
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
                const uint32_t k = w_ctx->S.a[s] + w_ctx->S.b[s];

		/* Linear*/
		/*uint32_t hash = k;
		hash = (hash % ht_size);*/

		/* Murmur*/
		uint32_t hash;
		MurmurHash3_x86_32((void*)&k, sizeof(uint32_t), 1, &hash);
		hash = (hash % ht_size);
		
		uint32_t tpl_cntr = hmS[hash].counter.fetch_add(1,std::memory_order_relaxed);

		if (tpl_cntr >= tpl_per_chunk) {
			cout << hash << " " << tpl_cntr << "\n";
			cout << "Chunk full at index: " << hash  << "\n";
			for (int i = 0; i < tpl_cntr*2 ; i++)
				cout << ((uint32_t*) hmS[hash].address)[i] << "\n";
			/* Write HT dump */
			ofstream file;
			file.open ("ht_dump.txt");
			for (int i = 0; i < ht_size; i++) 
				file << hmS[i].counter.load(std::memory_order_relaxed) << "\n";
			file.close();

			exit(1);
		}

		((uint32_t*) hmS[hash].address)[tpl_cntr*2] = k; // key
		((uint32_t*) hmS[hash].address)[(tpl_cntr*2)+1] = s; // value
        }

	auto end_time = timer.now();

#ifdef DEBUG
        cout << "Build S "
		<< std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() << "\n";
#endif
	w_ctx->stats.runtime_proc += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();

	start_time = timer.now();

        // Probe R HT
	auto emitted_sum = 0;
	auto to_delete_sum = 0;
	bitset<ht_size> to_delete_bitmap;
#pragma omp parallel for reduction(+: emitted_sum, to_delete_sum)
        for (unsigned s = *(w_ctx->s_processed);
            s < *(w_ctx->s_processed) + w_ctx->s_batch_size;
            s++){
                const uint32_t k = w_ctx->S.a[s] + w_ctx->S.b[s];

		/* Linear
		uint32_t hash = k;
		hash = (hash % ht_size);*/

		/* Murmur */
		uint32_t hash;
		MurmurHash3_x86_32((void*)&k, sizeof(uint32_t), 1, &hash);
		hash = (hash % ht_size);
		
		unsigned emitted_tuples = 0;
		unsigned to_delete_tuples = 0;
		vector<tuple<uint32_t, uint32_t>> out_tuples;
		const uint32_t tpl_cntr = hmR[hash].counter.load(std::memory_order_relaxed);
		if (tpl_cntr != 0) { // Not empty
			const uint32_t *cur_chunk = (uint32_t*) hmR[hash].address; // head
			for (int j = 0; j < tpl_cntr; j++){
				const uint32_t r = cur_chunk[(j*2)+1];
				if ((r_get_tns(ctx,r).count() + w_ctx->window_size_R * n_sec)
					> s_get_tns(ctx,s).count()) { // Valid
					if (cur_chunk[j*2] == k) { // match
						//cout << "R: " << r << " " << k << "\n";
						out_tuples.push_back(tuple<uint32_t, uint32_t>(r,s));
						emitted_tuples++;
						calc_latency(w_ctx, r, s);
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
			//if ((processed_tuples+emitted_sum+out_tuples.size()) >= output_buffer_tuples){
			//	cout << "output buffer full\n";
			//	exit(0);
			//}
			/* Bounded by output buffer size */
			output[((processed_tuples+emitted_sum+j) % output_buffer_tuples)*2]   = get<0>(i);
			output[((processed_tuples+emitted_sum+j) % output_buffer_tuples)*2+1] = get<1>(i);
			j++;
		}
		
		to_delete_sum += to_delete_tuples;
		emitted_sum += emitted_tuples;
        }

	processed_tuples += emitted_sum;

	end_time = timer.now();
#ifdef DEBUG
        cout << "Probe R " << 
		std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() << "\n";
#endif
	w_ctx->stats.runtime_proc += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();

        *(w_ctx->s_processed) += w_ctx->s_batch_size;

	start_time = timer.now();

	/* Clean-up R HT */
	//if (!to_delete_bitmap.none()) {
	if (to_delete_sum > cleanup_threshold) {
		//cout << "delete " << to_delete_sum << " tuples\n";
#pragma omp parallel for
		for (size_t i = 0; i < ht_size; i++){
			if (to_delete_bitmap.test(i)){
				//cout << "delete " << i << "\n";
				uint32_t tpl_cnt = hmR[i].counter.load(std::memory_order_relaxed);
				uint32_t *chunk = (uint32_t*) hmR[i].address; // head
				if (i == 1500)
					cout << "Remove\n";
				//cout << tpl_cnt << "\n";
				//for(size_t j = 0; j < tpl_cnt; j++) { // non-empty
				//	cout << "before: "<< chunk[(j*2)+1] << "\n";
				//}
				for(size_t j = 0; j < tpl_cnt; j++) { // non-empty
					const size_t r = chunk[(j*2)+1];
					if ((r_get_tns(ctx,r).count() + w_ctx->window_size_R * n_sec) 
						< r_get_tns(ctx,*(w_ctx->r_processed)).count()) {
						// Remove + Move
						for (int u = 0, l = 0; u < tpl_cnt; u++) {
							if ((u != j || u != j+1) && u != l) {
								chunk[l*2] = chunk[u*2];
								chunk[l*2+1] = chunk[u*2+1];
								l++;
							}
						}
						tpl_cnt--;
						hmR[i].counter--; // Update tpl counter
					}
				}
				//for(size_t j = 0; j < tpl_cnt; j++) { // non-empty
				//	cout << "after: "<< chunk[(j*2)+1] << "\n";
				//}
			}
		}
	}
	
	end_time = timer.now();
#ifdef DEBUG
        cout << "Clean up R " << 
		std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() << "\n";
#endif
	w_ctx->stats.runtime_proc += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
}
}
