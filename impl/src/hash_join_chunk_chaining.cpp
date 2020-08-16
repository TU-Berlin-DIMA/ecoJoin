#include <vector>
#include <mutex>
#include <unordered_map>
#include <iostream>
#include <atomic>
#include <algorithm>
#include <sstream>

#include "omp.h"
#include "worker.h"
#include "radix_partition.cpp"
#include "murmur3.h"
#include "hash_join_chunk_chaining.h"

using namespace std;

namespace mt_atomic_chunk {
// do not change
static const long n_sec = 1000000000L;
static const int uint64_t_size = 64;
static const int cacheline_size = 64;
static const int tpl_per_chunk  = uint64_t_size;

// Parameters
static const int ht_size = 75000;
static const int cleanup_threshold = 20;

// do not change
static const int max_try = ht_size;
static const int chunk_size = tpl_per_chunk * sizeof(uint32_t) * 2;

/* keep track of processed tuples */
std::atomic<uint32_t> processed_tuples;

struct ht{
	atomic<uint32_t> counter;
	uint64_t address;
}; // 16 Byte

ht *hmR;
ht *hmS;
uint64_t *cleanupR;
uint64_t *cleanupS;

void init_ht(){
	cout << "# Use a hash table size of " << ht_size << " chunks with " << tpl_per_chunk << " tuples\n";
	cout << "# Total hash table size is " << ht_size * tpl_per_chunk * 8 /*byte*/ 
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


void emit_result (worker_ctx_t *w_ctx, unsigned int r, unsigned int s)
{
	// Calculate Latency
	// INFLUENCES MULTITHREADED PERFORMANCE SIGNIFICANTLY:
        /* Choose the older tuple to calc the latency*/
        /*auto now = std::chrono::system_clock::now();
        if (w_ctx->S.t_ns[s] < w_ctx->R.t_ns[r]){
                auto i = (std::chrono::duration_cast <std::chrono::nanoseconds>(now -  w_ctx->stats.start_time) - w_ctx->R.t_ns[r]);
                #pragma omp critical
                w_ctx->stats.summed_latency = std::chrono::duration_cast <std::chrono::nanoseconds>(w_ctx->stats.summed_latency + i);
        } else {
                auto i = (std::chrono::duration_cast <std::chrono::nanoseconds>(now -  w_ctx->stats.start_time) - w_ctx->S.t_ns[s]);
                #pragma omp critical
                w_ctx->stats.summed_latency = std::chrono::duration_cast <std::chrono::nanoseconds>(w_ctx->stats.summed_latency + i);
        }*/
	
	// Print into resultfile
	fprintf (w_ctx->resultfile, "%d %d\n", r,s);
}



void process_r_ht_cpu(worker_ctx_t *w_ctx){

	auto start_time = std::chrono::high_resolution_clock::now();
        // Build R HT
#pragma omp parallel for 
        for (unsigned r = *(w_ctx->r_processed);
            r < *(w_ctx->r_processed) + w_ctx->r_batch_size;
            r++){
                const uint32_t k = w_ctx->R.x[r] + w_ctx->R.y[r];

		/* Linear */
		uint32_t hash = k;
		hash = (hash % ht_size);

		/* Murmur
		uint32_t hash;
		MurmurHash3_x86_32((void*)&k, sizeof(uint32_t), 1, &hash);
		hash = (hash % ht_size);*/
		
		uint32_t tpl_cntr = hmR[hash].counter.fetch_add(1,std::memory_order_relaxed);

		if (tpl_cntr == tpl_per_chunk) {
			cout << "Chunk full\n";
			exit(1);
		}

		((uint64_t*) hmR[hash].address)[tpl_cntr*2] = k; // key
		((uint64_t*) hmR[hash].address)[(tpl_cntr*2)+1] = r; // value
        }
	auto end_time = std::chrono::high_resolution_clock::now();
        cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() << "\n";

	start_time = std::chrono::high_resolution_clock::now();
        // Probe S HT
	auto sum = 0;
#pragma omp parallel for reduction(+: sum)
        for (unsigned r = *(w_ctx->r_processed);
            r < *(w_ctx->r_processed) + w_ctx->r_batch_size;
            r++){
                const uint32_t k = w_ctx->R.x[r] + w_ctx->R.y[r];

		/* Linear*/
		uint32_t hash = k;
		hash = (hash % ht_size);

		/* Murmur
		uint32_t hash;
		MurmurHash3_x86_32((void*)&k, sizeof(uint32_t), 1, &hash);
		hash = (hash % ht_size);*/
		
		unsigned emitted_tuples = 0;
		const uint32_t tpl_cntr = hmS[hash].counter.load(std::memory_order_relaxed);
		if (tpl_cntr != 0){ // Not empty
			const uint64_t *cur_chunk = (uint64_t*) hmS[hash].address; // head
			for (int j = 0; j < tpl_cntr; j++){
				if (cur_chunk[j*2] == k) { // match
					const uint32_t s = cur_chunk[(j*2)+1];
					//cout << "S: " << s << " " << k << "\n";
					/*if ((w_ctx->S.t_ns[s].count() + w_ctx->window_size_S * n_sec)
						> w_ctx->R.t_ns[r].count()) { // Valid
						mt_atomic_chunk::emit_result(w_ctx, r, s);
						emitted_tuples++;
					}*/ /*else { // Unvalid
						cleanupS[s] |= 1UL << j;
					}*/
					emitted_tuples++;
				}
			}
		}
		//w_ctx->stats.processed_output_tuples += emitted_tuples;
		//processed_tuples += emitted_tuples;
		sum += emitted_tuples;
        }
	processed_tuples += sum;
	end_time = std::chrono::high_resolution_clock::now();
        cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() << "\n";

        *(w_ctx->r_processed) += w_ctx->r_batch_size;

	/* Clean-up S HT */
/*#pragma omp parallel for
	for (size_t i = 0; i < ht_size; i++){
		uint32_t tpl_cnt = hmS[i].counter.load(std::memory_order_relaxed);
		uint64_t *chunk = (uint64_t*) hmS[i].address; // head
		for(size_t j = 0; j < tpl_cnt; j++) { // non-empty
			const size_t s = chunk[(j*2)+1];
			if ((w_ctx->S.t_ns[s].count() + w_ctx->window_size_S * n_sec) 
				< w_ctx->R.t_ns[*(w_ctx->r_processed)].count()) {
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
	}*/
}

void process_s_ht_cpu(worker_ctx_t *w_ctx){

	auto start_time = std::chrono::high_resolution_clock::now();
	// Build S HT
#pragma omp parallel for 
	for (unsigned s = *(w_ctx->s_processed);
            s < *(w_ctx->s_processed) + w_ctx->s_batch_size;
            s++){
                const uint32_t k = w_ctx->S.a[s] + w_ctx->S.b[s];

		/* Linear*/
		uint32_t hash = k;
		hash = (hash % ht_size);

		/* Murmur
		uint32_t hash;
		MurmurHash3_x86_32((void*)&k, sizeof(uint32_t), 1, &hash);
		hash = (hash % ht_size);*/
		
		uint32_t tpl_cntr = hmS[hash].counter.fetch_add(1,std::memory_order_relaxed);

		if (tpl_cntr == tpl_per_chunk) {
			cout << "Chunk full\n";
			exit(1);
		}

		((uint64_t*) hmS[hash].address)[tpl_cntr*2] = k; // key
		((uint64_t*) hmS[hash].address)[(tpl_cntr*2)+1] = s; // value
        }
	auto end_time = std::chrono::high_resolution_clock::now();
        cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() << "\n";

	start_time = std::chrono::high_resolution_clock::now();

        // Probe R HT
	auto sum = 0;
#pragma omp parallel for reduction(+: sum)
        for (unsigned s = *(w_ctx->s_processed);
            s < *(w_ctx->s_processed) + w_ctx->s_batch_size;
            s++){
                const uint32_t k = w_ctx->S.a[s] + w_ctx->S.b[s];

		/* Linear*/
		uint32_t hash = k;
		hash = (hash % ht_size);

		/* Murmur
		uint32_t hash;
		MurmurHash3_x86_32((void*)&k, sizeof(uint32_t), 1, &hash);
		hash = (hash % ht_size);*/
		
		unsigned emitted_tuples = 0;
		const uint32_t tpl_cntr = hmR[hash].counter.load(std::memory_order_relaxed);
		if (tpl_cntr != 0) { // Not empty
			const uint64_t *cur_chunk = (uint64_t*) hmR[hash].address; // head
			for (int j = 0; j < tpl_cntr; j++){
				if (cur_chunk[j*2] == k) { // match
					const uint32_t r = cur_chunk[(j*2)+1];
					//cout << "R: " << r << " " << k << "\n";
					/*if ((w_ctx->R.t_ns[r].count() + w_ctx->window_size_R * n_sec)
						> w_ctx->S.t_ns[s].count()) {
						mt_atomic_chunk::emit_result(w_ctx, r, s);
						emitted_tuples++;
					} *//*else { // Unvalid
						del++;
						//cleanupR[r] |= 1UL << j;
					}*/
						emitted_tuples++;
				}
			}
		}
        	//#pragma omp atomic
		//w_ctx->stats.processed_output_tuples += emitted_tuples;
		sum += emitted_tuples;
        }

	processed_tuples += sum;
	end_time = std::chrono::high_resolution_clock::now();
        cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() << "\n";
	//cout << del << "\n";

	/*std::bitset<bitwidth> bs(clearupR[r]);
	if (bs.count() > cleanup_threshold) {
	
	}*/

        *(w_ctx->s_processed) += w_ctx->s_batch_size;

	// Clean-up S HT
/*#pragma omp parallel for
	for (size_t i = 0; i < ht_size; i++){
		uint32_t tpl_cnt = hmR[i].counter.load(std::memory_order_relaxed);
		uint64_t *chunk = (uint64_t*) hmR[i].address; // head
		for(size_t j = 0; j < tpl_cnt; j++) { // non-empty
			const size_t r = chunk[(j*2)+1];
			if ((w_ctx->R.t_ns[r].count() + w_ctx->window_size_R * n_sec) 
				< w_ctx->S.t_ns[*(w_ctx->s_processed)].count()) {
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
	}*/
}
}
