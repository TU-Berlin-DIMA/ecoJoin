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

using namespace std;

static const long n_sec = 1000000000L;

static const int tpl_per_chunk  = 110;
static const int ht_size = 80000;
static const int max_try = ht_size;

namespace mt_atomic_chunk {

static const int cacheline_size = 64;
static const int chunk_size = tpl_per_chunk * sizeof(uint32_t) * 2;

struct ht{
	atomic<uint32_t> counter;
	uint64_t address;
}; 

ht *hmR;
ht *hmS;

void init_ht(){
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
        cout << i << " " << z << "\n";


        i= 0;
        z= 0;
        for (int j = 0; j < ht_size; j++){
                if (hmR[j].counter > 0)
                        i++;
                z += j % hmR[j].counter;

        }
        cout << i << " " << z << "\n";
	printTop10();
	print_ht_entries(w_ctx);
}


void emit_result (worker_ctx_t *w_ctx, unsigned int r, unsigned int s)
{
        //auto now = std::chrono::system_clock::now();

        /* Choose the older tuple to calc the latency*/
        /*if (w_ctx->S.t_ns[s] < w_ctx->R.t_ns[r]){
                auto i = (std::chrono::duration_cast <std::chrono::nanoseconds>(now -  w_ctx->stats.start_time) - w_ctx->R.t_ns[r]);
                //#pragma omp critical
                w_ctx->stats.summed_latency = std::chrono::duration_cast <std::chrono::nanoseconds>(w_ctx->stats.summed_latency + i);
                //std::cout << "R: " << std::chrono::duration_cast <std::chrono::milliseconds>(i).count() << "\n";
        } else {
                auto i = (std::chrono::duration_cast <std::chrono::nanoseconds>(now -  w_ctx->stats.start_time) - w_ctx->S.t_ns[s]);
                //#pragma omp critical
                w_ctx->stats.summed_latency = std::chrono::duration_cast <std::chrono::nanoseconds>(w_ctx->stats.summed_latency + i);
                //std::cout << "S: " << std::chrono::duration_cast <std::chrono::milliseconds>(i).count() << "\n";
        }*/
        //std::cout << r <<  " " << s << " " << *(w_ctx->s_available) <<  " " <<  *(w_ctx->s_processed) << " " << *(w_ctx->r_available) <<  " " <<  *(w_ctx->r_processed) << "\n";

        //std::cout << w_ctx->S.a[s] << " " << w_ctx->S.b[s] << " " << w_ctx->R.x[r] << " " << w_ctx->R.y[r] << "\n";
	// Print into resultfile
	//fprintf (w_ctx->resultfile, "%d %d\n", r,s);

        /* Output tuple statistics */
        #pragma omp atomic
        w_ctx->stats.processed_output_tuples++;
}



void process_r_ht_cpu(worker_ctx_t *w_ctx, unsigned threads){
        omp_set_num_threads(threads);

        // Build R HT
#pragma omp parallel for
        for (unsigned r = *(w_ctx->r_processed);
            r < *(w_ctx->r_processed) + w_ctx->r_batch_size;
            r++){
                const uint32_t k = w_ctx->R.x[r] + w_ctx->R.y[r];

		uint32_t hash;
		MurmurHash3_x86_32((void*)&k, sizeof(uint32_t), 1, &hash);
		hash = (hash % ht_size);
		
		uint32_t tpl_cntr = hmR[hash].counter.fetch_add(1,std::memory_order_relaxed);
		//uint32_t tpl_cntr = hmR[hash].counter.fetch_add(1);

		if (tpl_cntr == tpl_per_chunk) {
			cout << "Chunk full\n";
			exit(1);
		}

		((uint64_t*) hmR[hash].address)[tpl_cntr*2] = k; // key
		((uint64_t*) hmR[hash].address)[(tpl_cntr*2)+1] = r; // value
		//cout << "chunk r" << k << " "<< r << " " << tpl_cntr << " \n";
        }

        // Probe S HT
#pragma omp parallel for
        for (unsigned r = *(w_ctx->r_processed);
            r < *(w_ctx->r_processed) + w_ctx->r_batch_size;
            r++){
                const uint32_t k = w_ctx->R.x[r] + w_ctx->R.y[r];

		uint32_t hash;
		MurmurHash3_x86_32((void*)&k, sizeof(uint32_t), 1, &hash);
		hash = (hash % ht_size);
		
		const uint32_t tpl_cntr = hmS[hash].counter.load(std::memory_order_relaxed);
		if (tpl_cntr != 0){ // Not empty
			const uint64_t *cur_chunk = (uint64_t*) hmS[hash].address; // head
			for (int j = 0; j < tpl_cntr; j++){
				if (cur_chunk[j*2] == k) { // match
					const uint32_t s = cur_chunk[(j*2)+1];
					//cout << "S: " << s << " " << k << "\n";
					//if (! (w_ctx->S.t_ns[s].count() + w_ctx->window_size_S * n_sec)
					//	< w_ctx->R.t_ns[r].count())
					mt_atomic_chunk::emit_result(w_ctx, r, s);
				}
			}
		}
        }

	/* Delete S HT
	const unsigned r_processed = *(w_ctx->r_processed);
#pragma omp parallel for
	for (size_t i = 0; i < ht_size; i += 2){
		const uint32_t tpl_cnt = hmS[i+1];
		if (tpl_cnt != 0) { // non-empty
			atomic<uint32_t> *chunk = ((atomic<uint32_t>*) hmS[i].load()); // head
			for( size_t j = 0; j < hmS[i+1]; j++){

				const size_t s = chunk[j*2];

				if ((w_ctx->S.t_ns[s].count() + w_ctx->window_size_S*n_sec) 
						< w_ctx->R.t_ns[r_processed].count()){
					
					// (Re-)Move
					for (int u = 0, l = 0; u < hmS[i+1]; u++){
						if (u != j || u != j+1){
							uint32_t z = chunk[u].load();
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

void process_s_ht_cpu(worker_ctx_t *w_ctx, unsigned threads){
        omp_set_num_threads(threads);

	// Build S HT
#pragma omp parallel for 
	for (unsigned s = *(w_ctx->s_processed);
            s < *(w_ctx->s_processed) + w_ctx->s_batch_size;
            s++){
                const uint32_t k = w_ctx->S.a[s] + w_ctx->S.b[s];

		uint32_t hash;
		MurmurHash3_x86_32((void*)&k, sizeof(uint32_t), 1, &hash);
		hash = (hash % ht_size);
		
		uint32_t tpl_cntr = hmS[hash].counter.fetch_add(1,std::memory_order_relaxed);
		//uint32_t tpl_cntr = hmS[hash].counter.fetch_add(1);

		if (tpl_cntr == tpl_per_chunk) {
			cout << "Chunk full\n";
			exit(1);
		}

		((uint64_t*) hmS[hash].address)[tpl_cntr*2] = k; // key
		((uint64_t*) hmS[hash].address)[(tpl_cntr*2)+1] = s; // value
		//cout << "chunk s" << k << " "<< s << " " << tpl_cntr << " \n";
        }

        // Probe R HT
#pragma omp parallel for
        for (unsigned s = *(w_ctx->s_processed);
            s < *(w_ctx->s_processed) + w_ctx->s_batch_size;
            s++){
                const uint32_t k = w_ctx->S.a[s] + w_ctx->S.b[s];

		uint32_t hash;
		MurmurHash3_x86_32((void*)&k, sizeof(uint32_t), 1, &hash);
		hash = (hash % ht_size);
		
		const uint32_t tpl_cntr = hmR[hash].counter.load(std::memory_order_relaxed);
		if (tpl_cntr != 0) { // Not empty
			//cout << "R: " << hmR[hash+1] << "\n";
			const uint64_t *cur_chunk = (uint64_t*) hmR[hash].address; // head
			for (int j = 0; j < tpl_cntr; j++){
				if (cur_chunk[j*2] == k) { // match
					const uint32_t r = cur_chunk[(j*2)+1];
					//cout << "R: " << r << " " << k << "\n";
					//if (! (w_ctx->R.t_ns[r].count() + w_ctx->window_size_R * n_sec)
					//	< w_ctx->S.t_ns[s].count())
					mt_atomic_chunk::emit_result(w_ctx, r, s);
				}
			}
		}
        }

	// Delete R HT
	/*const unsigned s_processed = *(w_ctx->s_processed);
#pragma omp parallel for
	for (size_t i = 0; i < ht_size; i += 2){
		const uint32_t tpl_cnt = hmR[i+1];
		if (tpl_cnt != 0) { // non-empty
			atomic<uint32_t> *chunk = ((atomic<uint32_t>*) hmR[i].load()); // head
			for( size_t j = 0; j < hmR[i+1]; j++){

				const size_t r = chunk[j*2];

				if ((w_ctx->R.t_ns[r].count() + w_ctx->window_size_R*n_sec) 
						< w_ctx->S.t_ns[s_processed].count()){

					// (Re-)Move
					for (int u = 0, l = 0; u < hmR[i+1]; u++){
						if (u != j || u != j+1) {
							uint32_t z = chunk[u].load();
							chunk[l] = z;
							l++;
						}
					}
					hmR[i+1]--; // Update tpl count 
				}

			}
		}
	}*/
        *(w_ctx->s_processed) += w_ctx->s_batch_size;
}
}
