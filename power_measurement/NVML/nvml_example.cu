/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/* This example demonstrates how to use NVML library with
 * C++11 multithreading to create GPU monitoring with a
 * high sampling rate by storing nvidia-smi data to RAM
 * and then writing the output to a file once computation
 * is complete.
 */

/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Includes, cuda */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

/* Includes, custom */
#include "nvmlClass.h"

void calculate( int const & m, int const & n, int const & k, nvmlClass & nvml ) {

	float alpha = 1.0f, beta = 0.0f;
	int lda = m, ldb = k, ldc = m;

	size_t sizeA = m * k;
	size_t sizeB = k * n;
	size_t sizeC = m * n;

	cublasHandle_t handle;

	/* Initialize CUBLAS */
	checkCudaErrors( cublasCreate( &handle ) );

	/* Initialize CUBLAS */
	printf( "cublasSgemm %dx%dx%d test running..\n", m, n, k );

	/* Allocate host memory for the matrices */
	thrust::host_vector<float> h_A( sizeA, 0 );
	thrust::host_vector<float> h_B( sizeB, 0 );
	thrust::host_vector<float> h_C( sizeC, 0 );
	thrust::host_vector<float> h_C_ref( sizeC, 0 );

	/* Fill the matrices with test data */
	/* Assume square matrices */
	for ( int i = 0; i < m * m; i++ ) {
		h_A[i] = rand( ) / static_cast<float>( RAND_MAX );
		h_B[i] = rand( ) / static_cast<float>( RAND_MAX );
	}

	/* Create thread to gather GPU stats */
	std::thread threadStart( &nvmlClass::getStats, &nvml );   // threadStart starts running

	/* Allocate device memory for the matrices */
	thrust::device_vector<float> d_A( h_A );
	thrust::device_vector<float> d_B( h_B );
	thrust::device_vector<float> d_C( sizeC, 0 );

	/* Retrieve raw pointer for device data */
	float * d_A_ptr = thrust::raw_pointer_cast( &d_A[0] );
	float * d_B_ptr = thrust::raw_pointer_cast( &d_B[0] );
	float * d_C_ptr = thrust::raw_pointer_cast( &d_C[0] );

	/* Performs operation using cublas */
	cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A_ptr, lda, d_B_ptr, ldb, &beta, d_C_ptr, ldc );
	checkCudaErrors( cudaDeviceSynchronize( ) );

	/* Allocate host memory for reading back the result from device memory */
	h_C = d_C;

	/* Create thread to kill GPU stats */
	/* Join both threads to main */
	std::thread threadKill( &nvmlClass::killThread, &nvml );
	threadStart.join( );
	threadKill.join( );

	/* Shutdown */
	checkCudaErrors( cublasDestroy( handle ) );
}

/* Main */
int main( int argc, char **argv ) {

	int dev = findCudaDevice( argc, ( const char ** ) argv );
	if ( dev == -1 ) throw std::runtime_error( "!!!! No CUDA device found\n" );

	std::string filename = { "../data/gpuStats.csv" };

	// Create NVML class to retrieve GPU stats
	nvmlClass nvml( dev, filename );

	for ( int i = 512; i <= 1024; i *= 2 )
		calculate( i, i, i, nvml );

	return ( EXIT_SUCCESS );
}
