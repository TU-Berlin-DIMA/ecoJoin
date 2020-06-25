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

static const int ht_size = 200000;
static const int max_try = ht_size;

using namespace std;

namespace mt_atomic {


atomic<uint32_t> *hmR;
atomic<uint32_t> *hmS;

void init_ht(){
	hmR = new atomic<uint32_t>[ht_size*2];
	hmS = new atomic<uint32_t>[ht_size*2];
	fill_n(hmR, ht_size*2, 0);
	fill_n(hmS, ht_size*2, 0);
}

void process_s_ht_cpu(worker_ctx_t *w_ctx, unsigned threads){
        threads = 3;
        omp_set_num_threads(threads);

        // Build S HT
#pragma omp parallel for
        for (unsigned s = *(w_ctx->s_processed);
            s < *(w_ctx->s_processed) + w_ctx->s_batch_size;
            s++){
                const uint32_t k = (uint32_t) w_ctx->S.a[s] + (uint32_t) w_ctx->S.b[s];
		uint32_t hash;
		MurmurHash3_x86_32((void*)&k, sizeof(uint32_t), 1, &hash);

		hash = (hash % ht_size)*2;
		
		for (int i= 0; i < max_try; i++){
			uint32_t u = 0;
			if (hmS[hash] == 0){
				if (atomic_compare_exchange_strong(&hmS[hash] ,&u, k)){ // Swapped
					hmS[hash+1] = s;
					break;
				}
			}
			hash += 2;
			if(hash >= ht_size*2)
				hash = 0;
		}
        }

        // Probe R HT
#pragma omp parallel for
        for (unsigned s = *(w_ctx->s_processed);
            s < *(w_ctx->s_processed) + w_ctx->s_batch_size;
            s++){
		const uint32_t k = (uint32_t) w_ctx->S.a[s] + (uint32_t) w_ctx->S.b[s];
		uint32_t hash;
		MurmurHash3_x86_32((void*)&k, sizeof(uint32_t), 1, &hash);

		hash = (hash % ht_size)*2;
		
		for (int i= 0; i < max_try; i++){
			//cout <<  hmR[hash] << " " << k << " " << i << "\n";
			if (hmR[hash] == 0){ // Empty
				break;
			}
			if (hmR[hash] == k){ // found
				uint32_t r = hmR[hash+1];
                                if((w_ctx->R.t_ns[r].count() + w_ctx->window_size_R * n_sec)
                                                < w_ctx->S.t_ns[s].count()){
                                        hmR[hash] = 0;
                                        hmR[hash+1] = 0;
				} else 
                                        emit_result(w_ctx, r, s);
			}
			hash += 2;
			if(hash >= ht_size*2)
				hash = 0;
		}
        }
        *(w_ctx->s_processed) += w_ctx->s_batch_size;
}

void process_r_ht_cpu(worker_ctx_t *w_ctx, unsigned threads){
        threads = 3;
        omp_set_num_threads(threads);

        // Build R HT
#pragma omp parallel for
        for (unsigned r = *(w_ctx->r_processed);
            r < *(w_ctx->r_processed) + w_ctx->r_batch_size;
            r++){
                const uint32_t k = (uint32_t) w_ctx->R.x[r] + (uint32_t) w_ctx->R.y[r];
		uint32_t hash;
		MurmurHash3_x86_32((void*)&k, sizeof(uint32_t), 1, &hash);

		hash = (hash % ht_size)*2;
		
		for (int i= 0; i < max_try; i++){
			if (hmR[hash] == 0){
				uint32_t u = 0;
				if (atomic_compare_exchange_strong(&hmR[hash] ,&u, k)){ // swapped
					hmR[hash+1] = r;
					break;
				}
			}
			hash += 2;
			if(hash >= ht_size*2)
				hash = 0;
		}
        }

        // Probe S HT
#pragma omp parallel for
        for (unsigned r = *(w_ctx->r_processed);
            r < *(w_ctx->r_processed) + w_ctx->r_batch_size;
            r++){
		const uint32_t k = (uint32_t) w_ctx->R.x[r] + (uint32_t) w_ctx->R.y[r];
		uint32_t hash;
		MurmurHash3_x86_32((void*)&k, sizeof(uint32_t), 1, &hash);

		hash = (hash % ht_size)*2;
		
		for (int i= 0; i < max_try; i++){
			if (hmS[hash] == 0){ // Empty
				break;
			}
			if (hmS[hash] == k){ // found
				uint32_t s = hmS[hash+1];
                                if((w_ctx->S.t_ns[s].count() + w_ctx->window_size_S * n_sec)
                                                < w_ctx->R.t_ns[r].count()){
                                        hmS[hash] = 0;
                                        hmS[hash+1] = 0;
				} else
                                        emit_result(w_ctx, r, s);
			}
				
			hash += 2;
			if(hash >= ht_size*2)
				hash = 0;
		}
        }
        *(w_ctx->r_processed) += w_ctx->r_batch_size;
}

void print_ht(){
	int i = 0;
	long z = 0;
	for (int j = 0; j < ht_size *2; j++){
		if (hmS[j] > 0)
			i++;	
		z += hmS[j];
	}
	cout << i << " " << z << "\n";
	i= 0;
	z= 0;
	for (int j = 0; j < ht_size *2; j++){
		if (hmR[j] > 0)
			i++;	
		z += hmR[j];
	}
	cout << i << " " << z << "\n";
}
}
