#include <algorithm>
#include <atomic>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <boost/dynamic_bitset.hpp>

#include "murmur3.h"
#include "omp.h"
#include "worker.h"
// #include "radix_partition.cpp"
#include "hash_join_chunk_chaining.h"

using namespace std;
using namespace boost;

namespace mt_atomic_chunk {

// Constants & Setup vars
static const long long n_sec = 1000000000L;
static const int tpl_per_chunk = 64;
static unsigned ht_size = 0; /* set in init */
static unsigned ht_mask = 0; /* set in init */
static unsigned output_mask = 0; /* set in init */

// Parameters HT Generation
static const double load_factor = 0.2;
static const int cleanup_threshold = 1;
static const unsigned output_size = 2097152; /* 2^21 */

/* keep track of processed tuples */
std::atomic<uint32_t> processed_tuples;

struct ht {
    atomic<uint32_t> counter;
    uint64_t address;
}; // 16 Byte

struct chunk_R {
    std::chrono::nanoseconds t_ns;
    x_t x; /* key */
    y_t y; /* value */
    unsigned r; /* index */
}; // 32 Byte

struct chunk_S {
    std::chrono::nanoseconds t_ns;
    a_t a; /* key */
    b_t b; /* value */
    unsigned s; /* index */
}; // 32 Byte

ht* hmR;
ht* hmS;

// Cleanup Bitmap
dynamic_bitset<> to_delete_bitmap_S;
dynamic_bitset<> to_delete_bitmap_R;

// Output Tuples Buffer
uint32_t* output;

/* return p with the smallest k for that holds: p = 2^k > n */
unsigned next_largest_power2(unsigned n)
{
    n--;
    n |= n >> 1; // Divide by 2^k for consecutive doublings of k up to 32,
    n |= n >> 2; // and then 'or' the results.
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

void init(master_ctx_t* ctx)
{

    /* Init output tuple buffer */
    cout << "# Use a hast table load factor of " << load_factor << "\n";

    if (ctx->num_tuples_R != ctx->num_tuples_S) {
        cout << "# WARNING: ctx->num_tuples_R != ctx->num_tuples_S. May lead to problems in hash table generation.\n";
    }

    unsigned tpl_per_window = ctx->window_size_R * ctx->rate_R;
    cout << "# Expected number of tuples per window: " << tpl_per_window << "\n";

    ht_size = next_largest_power2(tpl_per_window / tpl_per_chunk);
    if (ht_size < (tpl_per_window / tpl_per_chunk) / load_factor) {
        ht_size = ht_size << 1;
    }
    ht_mask = ht_size - 1;

    cout << "# Use a hash table size of " << ht_size << " chunks with " << tpl_per_chunk << " tuples\n";

    unsigned chunk_size = tpl_per_chunk * sizeof(chunk_R);
    cout << "# The chunk size is " << chunk_size << " B\n";
    cout << "# Total hash table size is "
         << (long)ht_size * chunk_size / 1048576 /*MB*/ << " MB\n";

    assert(ht_size != 0);
    assert(ht_size % 2 == 0);
    assert(ht_mask != 0);

    posix_memalign((void**)&hmR, 64, ht_size * sizeof(ht));
    posix_memalign((void**)&hmS, 64, ht_size * sizeof(ht));

    for (unsigned i = 0; i < ht_size; i++) {
        chunk_R* chunk;
        posix_memalign((void**)&chunk, 64, chunk_size);
        hmR[i].address = (uint64_t)chunk;
        hmR[i].counter = 0;
    }
    for (unsigned i = 0; i < ht_size; i++) {
        chunk_S* chunk;
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
    to_delete_bitmap_S = dynamic_bitset<>(ht_size);
    to_delete_bitmap_R = dynamic_bitset<>(ht_size);
}

/* Calculate Latency for statistic output
 * INFLUENCES MULTITHREADED PERFORMANCE SIGNIFICANTLY */
inline void calc_latency_s(master_ctx_t* ctx, worker_ctx_t* w_ctx, unsigned s)
{
    /* Choose the older tuple to calc the latency */
    auto now = std::chrono::steady_clock::now();
    auto i = (std::chrono::duration_cast<std::chrono::nanoseconds>(now - w_ctx->stats.start_time) - s_get_tns(ctx, s));
#pragma omp critical
    {
        w_ctx->stats.summed_latency = std::chrono::duration_cast<std::chrono::nanoseconds>(w_ctx->stats.summed_latency + i);
        /*ofstream file;
	file.open ("latency_dump.txt",std::ios_base::app);
	file << r << " " << s << " " <<  i.count() 
		<< " " <<r_get_tns(ctx,r).count() << " " 
		<< std::chrono::duration_cast <std::chrono::nanoseconds>(now - w_ctx->stats.start_time).count() << "\n";
	file.close();*/
    }
}

inline void calc_latency_r(master_ctx_t* ctx, worker_ctx_t* w_ctx, unsigned r)
{
    /* Choose the older tuple to calc the latency */
    auto now = std::chrono::steady_clock::now();
    auto i = (std::chrono::duration_cast<std::chrono::nanoseconds>(now - w_ctx->stats.start_time) - r_get_tns(ctx, r));
#pragma omp critical
    {
        w_ctx->stats.summed_latency = std::chrono::duration_cast<std::chrono::nanoseconds>(w_ctx->stats.summed_latency + i);
        /*ofstream file;
	file.open ("latency_dump.txt",std::ios_base::app);
	file << r << " " << s << " " <<  i.count() 
		<< " " <<r_get_tns(ctx,r).count() << " " 
		<< std::chrono::duration_cast <std::chrono::nanoseconds>(now - w_ctx->stats.start_time).count() << "\n";
	file.close();*/
    }
}

void process_r_ht_cpu(master_ctx_t* ctx, worker_ctx_t* w_ctx)
{

    Timer::Timer timer = Timer::Timer();
    auto start_time = timer.now();

    // Build R HT
#pragma omp parallel for
    for (unsigned r = *(w_ctx->r_processed);
         r < *(w_ctx->r_processed) + w_ctx->r_batch_size;
         r++) {
        const uint32_t k = w_ctx->R.x[r];

        /* Linear */
        /*uint32_t hash = k;
		hash = (hash % ht_size);*/

        /* Murmur*/
        uint32_t hash;
        MurmurHash3_x86_32((void*)&k, sizeof(uint32_t), 1, &hash);
        hash = (hash & ht_mask);

        uint32_t tpl_cntr = hmR[hash].counter.fetch_add(1, std::memory_order_relaxed);
        //if (hash == 25)
        //	cout << tpl_cntr << " insert count in R " << w_ctx->R.t_ns[r].count() << "\n";

        if (tpl_cntr >= tpl_per_chunk) {
            cout << hash << " " << tpl_cntr << "\n";
            cout << "Chunk full at index: " << hash << " in R\n";
            for (unsigned i = 0; i < tpl_cntr * 2; i++) {
                cout << ((chunk_R*)hmR[hash].address)[i].x << " "
                     << ((chunk_R*)hmR[hash].address)[i].y << " "
                     << ((chunk_R*)hmR[hash].address)[i].t_ns.count() << "\n";
            }

            /* Write HT dump */
            ofstream file;
            file.open("ht_dump.txt");
            for (unsigned i = 0; i < ht_size; i++)
                file << hmR[i].counter.load(std::memory_order_relaxed) << "\n";
            file.close();
            exit(1);
        }

        if (tpl_cntr >= tpl_per_chunk * 0.5)
            to_delete_bitmap_R[hash] = 1;

        ((chunk_R*)hmR[hash].address)[tpl_cntr].x = k; // key
        ((chunk_R*)hmR[hash].address)[tpl_cntr].y = w_ctx->R.y[r]; // value
        ((chunk_R*)hmR[hash].address)[tpl_cntr].t_ns = r_get_tns(ctx, r); // ts
        ((chunk_R*)hmR[hash].address)[tpl_cntr].r = r; // index
    }

#ifdef DEBUG
    auto end_time = timer.now();
    w_ctx->stats.runtime_proc += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    w_ctx->stats.runtime_build += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    start_time = timer.now();
#endif

    // Probe S HT
    unsigned emitted_sum = 0;
    unsigned to_delete_sum = 0;
#pragma omp parallel for reduction(+ \
                                   : emitted_sum, to_delete_sum)
    for (unsigned r = *(w_ctx->r_processed);
         r < *(w_ctx->r_processed) + w_ctx->r_batch_size;
         r++) {
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
        if (tpl_cntr != 0) { // Not empty
            const chunk_S* chunk = (chunk_S*)hmS[hash].address; // head
            for (int j = 0; j < tpl_cntr; j++) {
                if ((chunk[j].t_ns.count() + w_ctx->window_size_S * n_sec)
                    > r_get_tns(ctx, r).count()) { // Valid
                    if (chunk[j].a == k) { // match
                        unsigned s = chunk[j].s;
                        out_tuples.push_back(tuple<uint32_t, uint32_t>(r, s));
                        emitted_tuples++;
#ifdef DEBUG
                        calc_latency_r(ctx, w_ctx, r);
                        //printf("from r: %d %d\n",r, s);
#endif
                    }
                } else { // Invalid
                    to_delete_bitmap_S[hash] = 1;
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
            output[((processed_tuples + emitted_sum + j) & output_mask) * 2] = get<0>(i);
            output[((processed_tuples + emitted_sum + j) & output_mask) * 2 + 1] = get<1>(i);
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
    w_ctx->stats.runtime_probe += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    start_time = timer.now();
#endif

    /* Clean-up S HT */
    if (to_delete_bitmap_S.count() > w_ctx->cleanup_threshold) {
#pragma omp parallel for
        for (size_t i = 0; i < ht_size; i++) {
            if (to_delete_bitmap_S.test(i)) {
                uint32_t tpl_cnt = hmS[i].counter.load(std::memory_order_relaxed);
                chunk_S* chunk = (chunk_S*)hmS[i].address; // head
                size_t j = 0;
                size_t u = tpl_cnt - 1ul;
                int deleted = 0;
                while (j <= u) {
                    if ((chunk[j].t_ns.count() + w_ctx->window_size_R * n_sec)
                        <= s_get_tns(ctx, *(w_ctx->s_processed)).count()) { // invalid
                        chunk[j].t_ns = chunk[u].t_ns;
                        chunk[j].a = chunk[u].a;
                        chunk[j].b = chunk[u].b;
                        chunk[j].s = chunk[u].s;

                        --u;
                        ++deleted;
                    } else {
                        ++j;
                    }
                }
                hmS[i].counter -= deleted;
            }
        }
        to_delete_bitmap_S.reset();
    }

#ifdef DEBUG
    end_time = timer.now();
    w_ctx->stats.runtime_proc += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    w_ctx->stats.runtime_cleanup += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
#else
    auto end_time = timer.now();
    w_ctx->stats.runtime_proc += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
#endif
}

void process_s_ht_cpu(master_ctx_t* ctx, worker_ctx_t* w_ctx)
{

    Timer::Timer timer = Timer::Timer();
    auto start_time = timer.now();

    // Build S HT
#pragma omp parallel for
    for (unsigned s = *(w_ctx->s_processed);
         s < *(w_ctx->s_processed) + w_ctx->s_batch_size;
         s++) {
        const uint32_t k = w_ctx->S.a[s];

        /* Linear*/
        /*uint32_t hash = k;
		hash = (hash % ht_size);*/

        /* Murmur*/
        uint32_t hash;
        MurmurHash3_x86_32((void*)&k, sizeof(uint32_t), 1, &hash);
        hash = (hash & ht_mask);

        uint32_t tpl_cntr = hmS[hash].counter.fetch_add(1, std::memory_order_relaxed);
        //if (hash == 25)
        //	cout << tpl_cntr << " insert count in S " << w_ctx->S.t_ns[s].count()
        //		<< " k: " << k << "hash: " << hash <<" \n";

        if (tpl_cntr >= tpl_per_chunk) {
            cout << hash << " " << tpl_cntr << "\n";
            cout << "Chunk full at index: " << hash << " in S\n";
            for (int i = 0; i < tpl_cntr * 2; i++) {
                cout << ((chunk_S*)hmS[hash].address)[i].a << " "
                     << ((chunk_S*)hmS[hash].address)[i].b << " "
                     << ((chunk_S*)hmS[hash].address)[i].t_ns.count() << "\n";
            }

            /* Write HT dump */
            ofstream file;
            file.open("ht_dump.txt");
            for (int i = 0; i < ht_size; i++)
                file << hmS[i].counter.load(std::memory_order_relaxed) << "\n";
            file.close();

            exit(1);
        }

        if (tpl_cntr >= tpl_per_chunk * 0.5)
            to_delete_bitmap_S[hash] = 1;

        ((chunk_S*)hmS[hash].address)[tpl_cntr].a = k; // key
        ((chunk_S*)hmS[hash].address)[tpl_cntr].b = w_ctx->S.b[s]; // value
        ((chunk_S*)hmS[hash].address)[tpl_cntr].t_ns = s_get_tns(ctx, s); // ts
        ((chunk_S*)hmS[hash].address)[tpl_cntr].s = s; // index
    }

#ifdef DEBUG
    auto end_time = timer.now();
    w_ctx->stats.runtime_proc += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    w_ctx->stats.runtime_build += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    start_time = timer.now();
#endif

    // Probe R HT
    auto emitted_sum = 0;
    auto to_delete_sum = 0;
    dynamic_bitset<> to_delete_bitmap(ht_size);
#pragma omp parallel for reduction(+ \
                                   : emitted_sum, to_delete_sum)
    for (unsigned s = *(w_ctx->s_processed);
         s < *(w_ctx->s_processed) + w_ctx->s_batch_size;
         s++) {
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
            const chunk_R* chunk = (chunk_R*)hmR[hash].address; // head
            for (unsigned j = 0; j < tpl_cntr; j++) {
                if ((chunk[j].t_ns.count() + w_ctx->window_size_R * n_sec)
                    > s_get_tns(ctx, s).count()) { // Valid
                    if (chunk[j].x == k) { // match
                        unsigned r = chunk[j].r;
                        out_tuples.push_back(tuple<uint32_t, uint32_t>(r, s));
                        emitted_tuples++;
#ifdef DEBUG
                        calc_latency_s(ctx, w_ctx, s);
                        //printf("from s: %d %d\n",r, s);
#endif
                    }
                } else { // Invalid
                    to_delete_bitmap_R[hash] = 1;
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
            output[((processed_tuples + emitted_sum + j) & output_mask) * 2] = get<0>(i);
            output[((processed_tuples + emitted_sum + j) & output_mask) * 2 + 1] = get<1>(i);
            j++;
        }

        to_delete_sum += to_delete_tuples;
        emitted_sum += emitted_tuples;
    }

    processed_tuples += emitted_sum;
    *(w_ctx->s_processed) += w_ctx->s_batch_size;

#ifdef DEBUG
    end_time = timer.now();
    w_ctx->stats.runtime_proc += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    w_ctx->stats.runtime_probe += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    start_time = timer.now();
#endif

    /* Clean-up R HT */
    if (to_delete_bitmap_R.count() > w_ctx->cleanup_threshold) {
#pragma omp parallel for
        for (size_t i = 0; i < ht_size; i++) {
            if (to_delete_bitmap_R.test(i)) {
                uint32_t tpl_cnt = hmR[i].counter.load(std::memory_order_relaxed);
                chunk_R* chunk = (chunk_R*)hmR[i].address; // head
                size_t j = 0;
                size_t u = tpl_cnt - 1ul;
                int deleted = 0;
                while (j <= u) {
                    if ((chunk[j].t_ns.count() + w_ctx->window_size_S * n_sec)
                        <= r_get_tns(ctx, *(w_ctx->r_processed)).count()) { // invalid
                        chunk[j].t_ns = chunk[u].t_ns;
                        chunk[j].x = chunk[u].x;
                        chunk[j].y = chunk[u].y;
                        chunk[j].r = chunk[u].r;

                        --u;
                        ++deleted;
                    } else {
                        ++j;
                    }
                }
                hmR[i].counter -= deleted;
            }
        }
        to_delete_bitmap_R.reset();
    }

#ifdef DEBUG
    end_time = timer.now();
    w_ctx->stats.runtime_proc += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    w_ctx->stats.runtime_cleanup += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
#else
    auto end_time = timer.now();
    w_ctx->stats.runtime_proc += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
#endif
}
}
