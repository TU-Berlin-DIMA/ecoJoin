#include <vector>
#include <mutex>
#include <unordered_map>
#include <iostream>
#include <atomic>

#include "omp.h"
#include "worker.h"
#include "tbb/concurrent_unordered_map.h"
#include "radix_partition.cpp"

static const long n_sec = 1000000000L;

using namespace std;
using namespace tbb;

namespace mt_tbb {

/*struct tpl {
	
	 * Using mutex in a struct deletes its constructors 
	 * 
	using MutexType = std::mutex;
    	using ReadLock = std::unique_lock<MutexType>;
    	using WriteLock = std::unique_lock<MutexType>;
	
	vector<int> vec;
	mutable MutexType mut_;
	ReadLock  read_lock_;
	WriteLock write_lock_;
	
	 Move Assignment 
	tpl& operator=(const tpl& t) {
	       if (this != &t)
	       {
	           WriteLock lhs_lk(mut_, std::defer_lock);
	           ReadLock  rhs_lk(t.mut_, std::defer_lock);
	           std::lock(lhs_lk, rhs_lk);
	           vec = t.vec;
	       }
	       return *this;
	}
	
	/* Copy 
	tpl(const tpl& t) 
		: read_lock_(t.mut_)
    		, vec(t.vec)
	{
    		read_lock_.unlock();
	}

	/* Default 
	tpl(): read_lock_(), write_lock_(), vec(){}
};*/

struct tpl{
	vector<int> vec;
	omp_lock_t lock;
	
	tpl(): lock(), vec(){
		omp_init_lock(&lock);
	}
};

using c_map = concurrent_unordered_map<int, tpl >;
//using c_map = unordered_map<int, tpl >;

c_map hmR = c_map{};
c_map hmS = c_map{};

/*  threading building blocks implementation
 *  vectors and map need to be thread safe
 *  unforunatly, neither concurrent_vector nor concurrent_unordered_map
 *  support thread-safe erase, thus its implemented here.
 */
void process_s_ht_cpu(worker_ctx_t *w_ctx, unsigned threads){
	threads = 2;
	omp_set_num_threads(threads);
        
	// Build S HT
#pragma omp parallel for 
        for (unsigned int s = *(w_ctx->s_processed);
            s < *(w_ctx->s_processed) + w_ctx->s_batch_size;
            s++){
                const int k = w_ctx->S.a[s] + w_ctx->S.b[s];
		omp_set_lock(&(hmS[k].lock));
                if (hmS.find(k) == hmS.end()) {
			hmS[k] = tpl();
                }
                hmS[k].vec.push_back(s);
		omp_unset_lock(&(hmS[k].lock));
        }

        // Probe R HT
#pragma omp parallel for 
        for (unsigned int s = *(w_ctx->s_processed);
            s < *(w_ctx->s_processed) + w_ctx->s_batch_size;
            s++){
                const int k = w_ctx->S.a[s] + w_ctx->S.b[s];
                if (hmR.find(k) != hmR.end()) {
			omp_set_lock(&(hmR[k].lock));
                        for (int i = 0; i < hmR[k].vec.size(); i++) {
                                const int r = hmR[k].vec[i];
                                if((w_ctx->R.t_ns[r].count() + w_ctx->window_size_R * n_sec) 
						< w_ctx->S.t_ns[s].count())
                                        hmR[k].vec.erase(hmR[k].vec.begin() + i);
                                 else
                                        emit_result(w_ctx, r, s);
                        }
			omp_unset_lock(&(hmR[k].lock));
                }
        }
        *(w_ctx->s_processed) += w_ctx->s_batch_size;
}
	
void process_r_ht_cpu(worker_ctx_t *w_ctx, unsigned threads){
	threads = 2;
	omp_set_num_threads(threads);

	// Build R HT
#pragma omp parallel for 
        for (unsigned int r = *(w_ctx->r_processed);
            r < *(w_ctx->r_processed) + w_ctx->r_batch_size;
            r++){
                const int k = w_ctx->R.x[r] + w_ctx->R.y[r];
		omp_set_lock(&(hmR[k].lock));
                if (hmR.find(k) == hmR.end()) {
			hmR.insert(make_pair(k,tpl()));
                }
                hmR[k].vec.push_back(r);
		omp_unset_lock(&(hmR[k].lock));
        }

        // Probe S HT
#pragma omp parallel for 
        for (unsigned int r = *(w_ctx->r_processed);
            r < *(w_ctx->r_processed) + w_ctx->r_batch_size;
            r++){
                const int k = w_ctx->R.x[r] + w_ctx->R.y[r];
                if (hmS.find(k) != hmS.end()) {
			omp_set_lock(&(hmS[k].lock));
                        for (int i = 0; i < hmS[k].vec.size(); i++) {
                                const int s = hmS[k].vec[i];
                                if((w_ctx->S.t_ns[s].count() + w_ctx->window_size_S * n_sec) 
						< w_ctx->R.t_ns[r].count())
                                        hmS[k].vec.erase(hmS[k].vec.begin() + i);
                                 else
                                        emit_result(w_ctx, r, s);
                        }
			omp_unset_lock(&(hmS[k].lock));
                }
        }
        *(w_ctx->r_processed) += w_ctx->r_batch_size;

}

} // namespace mt_tbb

