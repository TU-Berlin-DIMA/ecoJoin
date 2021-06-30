/*
 * Copyright (c) 2014 Cagri Balkesen, ETH Zurich
 * Copyright (c) 2014 Claude Barthels, ETH Zurich
 * Copyright (c) 2019 Clemens Lutz, German Research Center for Artificial
 * Intelligence
 *
 * Original sources by Cagri Balkesen and Claude Barthels are copyrighted under
 * the MIT license.
 *
 * Modications by Clemens Lutz are copyrighted under the Apache License 2.0.
 *
 * MIT license:
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Apache License 2.0:
 *
 * Author: Clemens Lutz, DFKI GmbH <clemens.lutz@dfki.de>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#if defined(__x86_64__)
#include <immintrin.h>
#elif defined(__ALTIVEC__)
#include <emmintrin.h>
#endif

#include <cassert>
#include <cstdint>
#include <cstring>

// Defines the cache-line size; usually this should be passed via the build
// script.
#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

// Defines the software write-combine buffer size; usually this should be passed
// via the build script.
#ifndef SWWC_BUFFER_SIZE
#define SWWC_BUFFER_SIZE CACHE_LINE_SIZE
#endif

using namespace std;

// Arguments to the partitioning function.
//
// Note that the struct's layout must be kept in sync with its counterpart in
// Rust.
struct RadixPartitionArgs {
  // Inputs
  const void *const __restrict__ join_attr_data;
  const void *const __restrict__ payload_attr_data;
  size_t const data_length;
  size_t const padding_length;
  uint32_t const radix_bits;

  // State
  uint64_t *const __restrict__ tmp_partition_offsets;
  void *const __restrict__ write_combine_buffer;

  // Outputs
  uint64_t *const __restrict__ partition_offsets;
  void *const __restrict__ partitioned_relation;
};

// A key-value tuple.
//
// Note that the struct's layout must be kept in sync with its counterpart in
// Rust.
template <typename K, typename V>
struct Tuple {
  K key;
  V value;
};

// A set of buffers used for software write-combinining.
//
// Supports two views of its data. The purpose is to align each buffer to the
// cache-line size. This requires periodic overwriting of `meta.slot` when a
// buffer becomes full. After emptying the buffer, the slot's value must be
// restored.
template <typename T, uint32_t size>
union WriteCombineBuffer {
  struct {
    T data[size / sizeof(T)];
  } tuples;

  struct {
    T data[(size - sizeof(uint64_t)) / sizeof(T)];
    char _padding[(size - sizeof(uint64_t)) -
                  (((size - sizeof(uint64_t)) / sizeof(T)) * sizeof(T))];
    uint64_t slot;  // Padding makes `slot` 8-byte aligned if sizeof(T) % 8 != 0
  } meta;

  // Computes the number of tuples contained in a buffer.
  static constexpr size_t tuples_per_buffer() { return size / sizeof(T); }
} __attribute__((packed));

// Computes the partition ID of a given key.
template <typename T, typename B>
size_t key_to_partition(T key, size_t mask, B bits) {
  return (static_cast<size_t>(key) & mask) >> bits;
}

/*
// Flushes a SWWC buffer from cache to memory.
//
// If possible, uses non-temporal SIMD writes, that require vector-length
// alignment. This in turn requires padding, because the front of a buffer may
// contain invalid data on the first flush.
void flush_buffer(void *const __restrict__ dst,
                  const void *const __restrict__ src) {
  auto byte_dst = static_cast<char *>(dst);
  auto byte_src = static_cast<const char *>(src);

#if defined(__AVX512F__)
  for (size_t i = 0; i < SWWC_BUFFER_SIZE; i += 64) {
    auto avx_dst = reinterpret_cast<__m512i *>(byte_dst + i);
    auto avx_src = reinterpret_cast<const __m512i *>(byte_src + i);

    _mm512_stream_si512(avx_dst, *avx_src);
  }
#elif defined(__AVX__)
  for (size_t i = 0; i < SWWC_BUFFER_SIZE; i += 32) {
    auto avx_dst = reinterpret_cast<__m256i *>(byte_dst + i);
    auto avx_src = reinterpret_cast<const __m256i *>(byte_src + i);

    _mm256_stream_si256(avx_dst, *avx_src);
  }
#elif defined(__ALTIVEC__) || defined(__SSE2__)
  // vec_st: 128-bit vector store; requires requires 16-byte alignment
  // See also:
  // https://gcc.gcc.gnu.narkive.com/cJndcMpR/vec-ld-versus-vec-vsx-ld-on-power8
  //
  // dcbtstt: cache-line non-temporal hint; unclear if requires 16-byte
  // alignment
  //
  // __dcbz(ptr): zeroes a cache-line in-cache to prevent load from memory
  //
  // _mm_stream_si128 wraps dcbtstt
  // See also:
  // https://github.com/gcc-mirror/gcc/blob/74c5e5f5bf7f2f13718008421cdf53bb0a814f4c/gcc/config/rs6000/emmintrin.h#L2249

  for (size_t i = 0; i < SWWC_BUFFER_SIZE; i += 16) {
    auto sse_dst = reinterpret_cast<__m128i *>(byte_dst + i);
    auto sse_src = reinterpret_cast<const __m128i *>(byte_src + i);

    _mm_stream_si128(sse_dst, *sse_src);
  }
#else
  memcpy(byte_dst, byte_src, SWWC_BUFFER_SIZE);
#endif
}*/

// Chunked radix partitioning.
//
// See the Rust module for details.
template <typename K, typename V>
void cpu_chunked_radix_partition(RadixPartitionArgs &args) {
  auto join_attr_data =
      static_cast<const K *const __restrict__>(args.join_attr_data);
  auto payload_attr_data =
      static_cast<const V *const __restrict__>(args.payload_attr_data);
  auto partitioned_relation =
      static_cast<Tuple<K, V> *const __restrict__>(args.partitioned_relation);

  const size_t fanout = 1UL << args.radix_bits;
  const size_t mask = fanout - 1;

  // Ensure counters are all zeroed
  for (size_t i = 0; i < fanout; ++i) {
    args.partition_offsets[i] = 0;
  }

  // 1. Compute local histograms per partition
  for (size_t i = 0; i < args.data_length; ++i) {
    auto key = join_attr_data[i];
    auto p_index = key_to_partition(key, mask, 0);
    args.partition_offsets[p_index] += 1;
  }

  // 2. Compute offsets with exclusive prefix sum
  for (size_t i = 0, sum = 0, offset = 0; i < fanout; ++i, offset = sum) {
    sum += args.partition_offsets[i];
    offset += (i + 1) * args.padding_length;
    args.partition_offsets[i] = offset;
    args.tmp_partition_offsets[i] = offset;
  }

  // 3. Partition
  for (size_t i = 0; i < args.data_length; ++i) {
    Tuple<K, V> tuple;
    tuple.key = join_attr_data[i];
    tuple.value = payload_attr_data[i];

    auto p_index = key_to_partition(tuple.key, mask, 0);
    auto &offset = args.tmp_partition_offsets[p_index];
    partitioned_relation[offset] = tuple;
    offset += 1;
  }
}

// Chunked radix partitioning with software write-combining.
//
// See the Rust module for details.
template <typename K, typename V>
void cpu_chunked_radix_partition_swwc(RadixPartitionArgs &args) {
  constexpr size_t tuples_per_buffer =
      WriteCombineBuffer<Tuple<K, V>, SWWC_BUFFER_SIZE>::tuples_per_buffer();

  // 512-bit intrinsics require 64-byte alignment
  assert(reinterpret_cast<size_t>(args.write_combine_buffer) % 64 == 0);

  // Padding must be a multiple of the buffer length.
  assert(args.padding_length % tuples_per_buffer == 0);

  auto join_attr_data =
      static_cast<const K *const __restrict__>(args.join_attr_data);
  auto payload_attr_data =
      static_cast<const V *const __restrict__>(args.payload_attr_data);
  auto partitioned_relation =
      static_cast<Tuple<K, V> *const __restrict__>(args.partitioned_relation);
  auto buffers = static_cast<
      WriteCombineBuffer<Tuple<K, V>, SWWC_BUFFER_SIZE> *const __restrict__>(
      args.write_combine_buffer);

  const size_t fanout = 1UL << args.radix_bits;
  const size_t mask = fanout - 1;

  // Ensure counters are all zeroed
  for (size_t i = 0; i < fanout; ++i) {
    args.partition_offsets[i] = 0;
  }

  // 1. Compute local histograms per partition
  for (size_t i = 0; i < args.data_length; ++i) {
    auto key = join_attr_data[i];
    auto p_index = key_to_partition(key, mask, 0);
    args.partition_offsets[p_index] += 1;
  }

  // 2. Compute offsets with exclusive prefix sum
  for (size_t i = 0, sum = 0, offset = 0; i < fanout; ++i, offset = sum) {
    sum += args.partition_offsets[i];
    offset += (i + 1) * args.padding_length;
    args.partition_offsets[i] = offset;
    buffers[i].meta.slot = offset;
  }

  // 3. Partition into software write combine buffers
  for (size_t i = 0; i < args.data_length; ++i) {
    Tuple<K, V> tuple;
    tuple.key = join_attr_data[i];
    tuple.value = payload_attr_data[i];

    auto p_index = key_to_partition(tuple.key, mask, 0);
    auto &buffer = buffers[p_index];

    size_t slot = buffer.meta.slot;
    size_t buffer_slot = slot % tuples_per_buffer;

    // `buffer.meta.slot` is overwritten on buffer_slot == (tuples_per_buffer -
    // 1), and restored after the buffer flush.
    buffer.tuples.data[buffer_slot] = tuple;

    // Flush buffer
    // Can occur on partially filled buffer due to cache-line alignment,
    // because first output slot might not be at offset % tuples_per_buffer == 0
    if (buffer_slot + 1 == tuples_per_buffer) {
      flush_buffer(partitioned_relation + (slot + 1) - tuples_per_buffer,
                   buffer.tuples.data);
    }

    // Restore `buffer.meta.slot` after overwriting it above, and increment its
    // value.
    buffer.meta.slot = slot + 1;
  }

  // Flush remainders of all buffers.
  for (size_t i = 0; i < fanout; ++i) {
    size_t slot = buffers[i].meta.slot;
    size_t remaining = slot % tuples_per_buffer;

    for (size_t j = slot - remaining, k = 0; k < remaining; ++j, ++k) {
      partitioned_relation[j] = buffers[i].tuples.data[k];
    }
  }
}

/*
// Exports the the size of all SWWC buffers.
extern "C" size_t cpu_swwc_buffer_bytes() { return SWWC_BUFFER_SIZE; }

// Exports the partitioning function for 8-byte key/value tuples.
extern "C" void cpu_chunked_radix_partition_int32_int32(
    RadixPartitionArgs *args) {
  cpu_chunked_radix_partition<int32_t, int32_t>(*args);
}

// Exports the partitioning function for 16-byte key/value tuples.
extern "C" void cpu_chunked_radix_partition_int64_int64(
    RadixPartitionArgs *args) {
  cpu_chunked_radix_partition<int64_t, int64_t>(*args);
}

// Exports the partitioning function for 8-byte key/value tuples.
extern "C" void cpu_chunked_radix_partition_swwc_int32_int32(
    RadixPartitionArgs *args) {
  cpu_chunked_radix_partition_swwc<int32_t, int32_t>(*args);
}

// Exports the partitioning function for 16-byte key/value tuples.
extern "C" void cpu_chunked_radix_partition_swwc_int64_int64(
    RadixPartitionArgs *args) {
  cpu_chunked_radix_partition_swwc<int64_t, int64_t>(*args);
}*/
