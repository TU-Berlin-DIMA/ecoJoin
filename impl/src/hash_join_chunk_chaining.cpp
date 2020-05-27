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

namespace mt_atomic_chunk {

const int cacheline_size = 64;
const int tpl_per_chunk  = 5;
const int chunk_size     = tpl_per_chunk * sizeof(uint64_t) * 2 + 1 /* ptr at end */;
const int next_chunk_ptr = tpl_per_chunk * 2; /* index with ptr to next chunk */

atomic<uint64_t> *hmR;
atomic<uint64_t> *hmS;

void init_ht(){
	hmR = new atomic<uint64_t>[ht_size*2];
	hmS = new atomic<uint64_t>[ht_size*2];
	fill_n(hmR, ht_size*2, 0);
	fill_n(hmS, ht_size*2, 0);
}

inline uint64_t *find_chunk_by_id(uint64_t* head, size_t id){
        uint64_t *chunk_ptr = head;
        for (int i = 0; i < id; i++){
                chunk_ptr = (uint64_t*) chunk_ptr[next_chunk_ptr];
        }
        return chunk_ptr;
}

inline uint64_t *find_last_chunk(const uint64_t tpl_cntr, uint64_t* head){
	const size_t chunks = (tpl_cntr == 0) ? 0 : (tpl_cntr-1) / tpl_per_chunk;
	return find_chunk_by_id(head, chunks);
}

inline void delete_tuple_in_chunk(size_t tpl_id_in_chain, atomic<uint64_t> *cur_chunk, uint64_t *head){
	
	const size_t i = (tpl_id_in_chain % tpl_per_chunk)*2;
	uint64_t u = 0; 
	if(!atomic_compare_exchange_strong(&cur_chunk[i], &u, (uint64_t)0))
		return; // Already deleted
	cur_chunk[i+1] = 0;
	
	// Chunk is empty?
	unsigned z;
	const size_t ints_in_chunk = tpl_per_chunk*2+1;
	for (z = 0; z < ints_in_chunk; z+=2){
		if (cur_chunk[z] != 0)
			break;
	}
	if (z == tpl_per_chunk*2+1){ // is last chunk in chain
		const size_t cur_chunk_in_chain = tpl_id_in_chain / tpl_per_chunk; 
		uint64_t *prev_chunk = find_chunk_by_id(head, cur_chunk_in_chain - 1);	
		prev_chunk[next_chunk_ptr] = 0;
		free(cur_chunk);
	}
	if (z == tpl_per_chunk*2){ // is in mittle of chain
		const size_t cur_chunk_in_chain = tpl_id_in_chain / tpl_per_chunk;
		uint64_t *prev_chunk = find_chunk_by_id(head, cur_chunk_in_chain - 1);
		prev_chunk[next_chunk_ptr] = cur_chunk[next_chunk_ptr];
		free(cur_chunk);
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
		
		//cout << "chunk s" << k << " "<< s <<" \n";
		if (tpl_cntr == 0){ // Empty
			uint64_t *chunk;
			posix_memalign((void**)&chunk, 64, chunk_size);
			hmS[hash]    = (uint64_t)chunk;
			((uint64_t*) hmS[hash].load())[0] = k; // key
			((uint64_t*) hmS[hash].load())[1] = s; // value
		} else {
			if (tpl_cntr % tpl_per_chunk == 0){ // Chunk full
				uint64_t *cur_chunk = find_last_chunk(tpl_cntr, ((uint64_t*) hmS[hash].load()));
				uint64_t *next_chunk;
				posix_memalign((void**)&next_chunk, 64, chunk_size);
				cur_chunk[next_chunk_ptr] = (uint64_t)next_chunk;
				next_chunk[0] = k; // key
				next_chunk[1] = s; // value
			} else { 
				uint64_t *cur_chunk = find_last_chunk(tpl_cntr, ((uint64_t*) hmS[hash].load()));
				const int cur_idx = (tpl_cntr % tpl_per_chunk) *2;
				cur_chunk[cur_idx]   = k; // key
				cur_chunk[cur_idx+1] = s; // value
			}
		}
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
		
		if (hmR[hash+1] != 0){ // Not empty
			const uint64_t tpl_cntr = hmR[hash+1];
			atomic<uint64_t> *cur_chunk = ((atomic<uint64_t>*) hmR[hash].load()); // head
			for (int j = 0; j < tpl_cntr; j++){
				if (j != 0 && j % tpl_per_chunk == 0) { // Next chunk?
					cur_chunk = (atomic<uint64_t>*)cur_chunk[next_chunk_ptr].load();
				}
				
				const size_t tpl_id_in_chunk = (j % tpl_per_chunk)*2;
				if (cur_chunk[tpl_id_in_chunk] == k) { // match
					const uint32_t r = cur_chunk[tpl_id_in_chunk+1];
					if ((w_ctx->R.t_ns[r].count() + w_ctx->window_size_R * n_sec)
						< w_ctx->S.t_ns[s].count())
						delete_tuple_in_chunk(j, cur_chunk, ((uint64_t*) hmR[hash].load()));
					else
						emit_result(w_ctx, r, s);
				}
			}
		}
        }
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
		
		//cout << "chunk r" << k << " "<< r <<" \n";
		if (tpl_cntr == 0){ // Empty
			uint64_t *chunk;
			posix_memalign((void**)&chunk, 64, chunk_size);
			hmR[hash]    = (uint64_t)chunk;
			((uint64_t*) hmR[hash].load())[0] = k; // key
			((uint64_t*) hmR[hash].load())[1] = r; // value
		} else {
			if (tpl_cntr % tpl_per_chunk == 0){ // Chunk full
				uint64_t *cur_chunk = find_last_chunk(tpl_cntr, ((uint64_t*) hmR[hash].load()));
				uint64_t *next_chunk;
				posix_memalign((void**)&next_chunk, 64, chunk_size);
				cur_chunk[next_chunk_ptr] = (uint64_t)next_chunk;
				next_chunk[0] = k; // key
				next_chunk[1] = r; // value
			} else { 
				uint64_t *cur_chunk = find_last_chunk(tpl_cntr, ((uint64_t*) hmR[hash].load()));
				const int cur_idx = (tpl_cntr % tpl_per_chunk) *2;
				cur_chunk[cur_idx]   = k; // key
				cur_chunk[cur_idx+1] = r; // value
			}
		}
        }
		
        // Probe S HT
//#pragma omp parallel for
	for (unsigned r = *(w_ctx->r_processed);
            r < *(w_ctx->r_processed) + w_ctx->r_batch_size;
            r++){
		const uint32_t k = w_ctx->R.x[r] + w_ctx->R.y[r];

		uint64_t hash;
		MurmurHash3_x86_32((void*)&k, sizeof(uint64_t), 1, &hash);
		hash = (hash % ht_size)*2;
		
		if (hmS[hash+1] != 0){ // Not empty
			const uint64_t tpl_cntr = hmS[hash+1];
			atomic<uint64_t> *cur_chunk = ((atomic<uint64_t>*) hmS[hash].load()); // head
			for (int j = 0; j < tpl_cntr; j++){
				if (j != 0 && j % tpl_per_chunk == 0) { // Next chunk?
					cur_chunk = (atomic<uint64_t>*)cur_chunk[next_chunk_ptr].load();
				}
				
				const size_t tpl_id_in_chunk = (j % tpl_per_chunk)*2;
				if (cur_chunk[tpl_id_in_chunk] == k) { // match
					const uint32_t s = cur_chunk[tpl_id_in_chunk+1];
					if ((w_ctx->S.t_ns[s].count() + w_ctx->window_size_S * n_sec)
						< w_ctx->R.t_ns[r].count())
						delete_tuple_in_chunk(j, cur_chunk, ((uint64_t*) hmS[hash].load()));
					else
						emit_result(w_ctx, r, s);
				}
			}
		}
        }
        *(w_ctx->r_processed) += w_ctx->r_batch_size;
}

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
