#include <vector>
#include <mutex>
#include <unordered_map>
#include <iostream>
#include <atomic>
#include <algorithm>

#include "omp.h"
#include "worker.h"
#include "radix_partition.cpp"
#include "murmur3.h"

static const long n_sec = 1000000000L;

static const int tpl_per_chunk  = 50;
static const int ht_size = 20000;
static const int max_try = ht_size;


using namespace std;

namespace mt_atomic_chunk {

static const int cacheline_size = 64;
static const int chunk_size = tpl_per_chunk * sizeof(uint64_t) * 2;

atomic<uint64_t> *hmR;
atomic<uint64_t> *hmS;

void init_ht(){
	cout << "Init HT\n";
	hmR = new atomic<uint64_t>[ht_size*2];
	hmS = new atomic<uint64_t>[ht_size*2];
	fill_n(hmR, ht_size*2, 0);
	fill_n(hmS, ht_size*2, 0);

	uint64_t *chunk;
	for (int i = 0; i < ht_size; i++){
		posix_memalign((void**)&chunk, 64, chunk_size);
		hmR[i*2] = (uint64_t)chunk;
		posix_memalign((void**)&chunk, 64, chunk_size);
		hmS[i*2] = (uint64_t)chunk;
	}
}

void process_s_ht_cpu(worker_ctx_t *w_ctx, unsigned threads){
        threads = 3;
        omp_set_num_threads(threads);

        // Build S HT
//#pragma omp parallel for
        for (unsigned s = *(w_ctx->s_processed);
            s < *(w_ctx->s_processed) + w_ctx->s_batch_size;
            s++){
                const uint64_t k = w_ctx->S.a[s] + w_ctx->S.b[s];

		uint64_t hash;
		MurmurHash3_x86_32((void*)&k, sizeof(uint64_t), 1, &hash);
		hash = (hash % ht_size)*2;
		
		// Block slot and update tuple counter
		uint64_t tpl_cntr = 0; 
		while (!atomic_compare_exchange_strong(&hmS[hash+1], &tpl_cntr, hmS[hash+1]+1));
		
		if (tpl_cntr == tpl_per_chunk) {
			cout << "Chunk full\n";
			exit(1);
		}

		//cout << "chunk s" << k << " "<< s <<" \n";
		((uint64_t*) hmS[hash].load())[tpl_cntr*2] = k; // key
		((uint64_t*) hmS[hash].load())[tpl_cntr*2+1] = s; // value
        }

        // Probe R HT
//#pragma omp parallel for
        for (unsigned s = *(w_ctx->s_processed);
            s < *(w_ctx->s_processed) + w_ctx->s_batch_size;
            s++){
		const uint32_t k = w_ctx->S.a[s] + w_ctx->S.b[s];

		uint64_t hash;
		MurmurHash3_x86_32((void*)&k, sizeof(uint64_t), 1, &hash);
		hash = (hash % ht_size)*2;
		
		if (hmR[hash+1] != 0) { // Not empty
			const uint64_t tpl_cntr = hmR[hash+1];
			atomic<uint64_t> *cur_chunk = ((atomic<uint64_t>*) hmR[hash].load()); // head
			for (int j = 0; j < tpl_cntr; j+=2){
				if (cur_chunk[j] == k) { // match
					const uint32_t r = cur_chunk[j+1];
					if (! (w_ctx->R.t_ns[r].count() + w_ctx->window_size_R * n_sec)
						< w_ctx->S.t_ns[s].count())
						emit_result(w_ctx, r, s);
				}
			}
		}
        }


	// Delete R HT
	/*
	const unsigned s_processed = *(w_ctx->s_processed);
//#pragma omp parallel for
	for (size_t i = 0; i < ht_size; i += 2){
		const uint64_t tpl_cnt = hmR[i+1];
		if (tpl_cnt != 0) { // non-empty
			atomic<uint64_t> *chunk = ((atomic<uint64_t>*) hmR[i].load()); // head
			for( size_t j = 0; j < hmR[i+1]; j++){

				const size_t r = chunk[j*2];

				if ((w_ctx->R.t_ns[r].count() + w_ctx->window_size_R*n_sec) 
						< w_ctx->S.t_ns[s_processed].count()){

					// (Re-)Move
					for (int u = 0, l = 0; u < hmR[i+1]; u++){
						if (u != j || u != j+1) {
							uint64_t z = chunk[u].load();
							chunk[l] = z;
							l++;
						}
					}
					hmR[i+1]--; // Update tpl count 
				}

			}
		}
	}
	*/
        *(w_ctx->s_processed) += w_ctx->s_batch_size;
}

void process_r_ht_cpu(worker_ctx_t *w_ctx, unsigned threads){
        threads = 3;
        omp_set_num_threads(threads);

        // Build R HT
//#pragma omp parallel for
        for (unsigned r = *(w_ctx->r_processed);
            r < *(w_ctx->r_processed) + w_ctx->r_batch_size;
            r++){
                const uint64_t k = w_ctx->R.x[r] + w_ctx->R.y[r];

		uint64_t hash;
		MurmurHash3_x86_32((void*)&k, sizeof(uint64_t), 1, &hash);
		hash = (hash % ht_size)*2;
		
		// Block slot and update tuple counter
		uint64_t tpl_cntr = 0; 
		while (!atomic_compare_exchange_strong(&hmR[hash+1], &tpl_cntr, hmR[hash+1]+1));
		
		if (tpl_cntr == tpl_per_chunk) {
			cout << "Chunk full\n";
			exit(1);
		}

		//cout << "chunk r" << k << " "<< r << " " << tpl_cntr << " \n";
		((uint64_t*) hmR[hash].load())[tpl_cntr*2] = k; // key
		((uint64_t*) hmR[hash].load())[tpl_cntr*2+1] = r; // value
        }

        // Probe R HT
//#pragma omp parallel for
        for (unsigned r = *(w_ctx->r_processed);
            r < *(w_ctx->r_processed) + w_ctx->r_batch_size;
            r++){
                const uint64_t k = w_ctx->R.x[r] + w_ctx->R.y[r];

		uint64_t hash;
		MurmurHash3_x86_32((void*)&k, sizeof(uint64_t), 1, &hash);
		hash = (hash % ht_size)*2;
		
		if (hmS[hash+1] != 0){ // Not empty
			const uint64_t tpl_cntr = hmS[hash+1];
			atomic<uint64_t> *cur_chunk = ((atomic<uint64_t>*) hmS[hash].load()); // head
			for (int j = 0; j < tpl_cntr; j+=2){
				if (cur_chunk[j] == k) { // match
					const uint32_t s = cur_chunk[j+1];
					if (! (w_ctx->S.t_ns[s].count() + w_ctx->window_size_S * n_sec)
						< w_ctx->R.t_ns[r].count())
						emit_result(w_ctx, r, s);
				}
			}
		}
        }


	// Delete S HT
	const unsigned r_processed = *(w_ctx->r_processed);
//#pragma omp parallel for
	/*
	for (size_t i = 0; i < ht_size; i += 2){
		const uint64_t tpl_cnt = hmS[i+1];
		if (tpl_cnt != 0) { // non-empty
			atomic<uint64_t> *chunk = ((atomic<uint64_t>*) hmS[i].load()); // head
			for( size_t j = 0; j < hmS[i+1]; j++){

				const size_t s = chunk[j*2];

				if ((w_ctx->S.t_ns[s].count() + w_ctx->window_size_S*n_sec) 
						< w_ctx->R.t_ns[r_processed].count()){
					
					// (Re-)Move
					for (int u = 0, l = 0; u < hmS[i+1]; u++){
						if (u != j || u != j+1){
							uint64_t z = chunk[u].load();
							chunk[l] = z;
							l++;
						}
					}
					hmS[i+1]--; // Update tpl count 
					
				}

			}
		}
	}*/


        *(w_ctx->r_processed) += w_ctx->r_batch_size;
}


// Legacy
void print_ht(){
        int i = 0;
        long z = 0;
        for (int j = 1; j < ht_size *2; j+=2){
                if (hmS[j] > 0)
                        i++;
                z += hmS[j];
        }
        cout << i << " " << z << "\n";
        i= 0;
        z= 0;
        for (int j = 1; j < ht_size *2; j+=2){
                if (hmR[j] > 0)
                        i++;
                z += hmR[j];
        }
        cout << i << " " << z << "\n";
}
}
