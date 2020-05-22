#include <vector>
#include <unordered_map>
#include <iostream>

#include "worker.h"

#define NSEC 1000000000L

using std::unordered_map;
using std::vector;

unordered_map<int, vector<int>> hmR = unordered_map<int, vector<int>>();
unordered_map<int, vector<int>> hmS = unordered_map<int, vector<int>>();

void process_r_ht_cpu(worker_ctx_t *w_ctx, unsigned threads){
	
	// Build R HT
	for (unsigned int r = *(w_ctx->r_processed);
	    r < *(w_ctx->r_processed) + w_ctx->r_batch_size;
	    r++){
		const int k = w_ctx->R.x[r] + w_ctx->R.y[r];
		if (hmR.find(k) == hmR.end()) {
			hmR[k] = vector<int>();
		}
		hmR[k].push_back(r);
	}

	// Probe S HT
	for (unsigned int r = *(w_ctx->r_processed);
	    r < *(w_ctx->r_processed) + w_ctx->r_batch_size;
	    r++){
		const int k = w_ctx->R.x[r] + w_ctx->R.y[r];
		if (hmS.find(k) != hmS.end()) {
  		        for (int i = 0; i < hmS[k].size(); i++){
				const int s = hmS[k][i];
				if((w_ctx->S.t_ns[s].count() + w_ctx->window_size_S*NSEC) < w_ctx->R.t_ns[r].count())
					hmS[k].erase(hmS[k].begin() + i);
				else
					emit_result(w_ctx, r, s);
			}
		}
	}
	*(w_ctx->r_processed) += w_ctx->r_batch_size;
}

void process_s_ht_cpu(worker_ctx_t *w_ctx, unsigned threads){

	// Build S HT
	for (unsigned int s = *(w_ctx->s_processed);
	    s < *(w_ctx->s_processed) + w_ctx->s_batch_size;
	    s++){
		const int k = w_ctx->S.a[s] + w_ctx->S.b[s];
		if (hmS.find(k) == hmS.end()) {
			hmS[k] = vector<int>();
		}
		hmS[k].push_back(s);
	}

	// Probe R HT
	for (unsigned int s = *(w_ctx->s_processed);
	    s < *(w_ctx->s_processed) + w_ctx->s_batch_size;
	    s++){

		const int k = w_ctx->S.a[s] + w_ctx->S.b[s];
		if (hmR.find(k) != hmR.end()) {
  		        for (int i = 0; i < hmR[k].size(); i++) {
				const int r = hmR[k][i];
				if((w_ctx->R.t_ns[r].count() + w_ctx->window_size_R*NSEC) < w_ctx->S.t_ns[s].count())
					hmR[k].erase(hmR[k].begin() + i);
				 else
					emit_result(w_ctx, r, s);
			}
		}
	}
	*(w_ctx->s_processed) += w_ctx->s_batch_size;
}
