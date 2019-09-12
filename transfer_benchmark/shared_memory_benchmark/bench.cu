#include <chrono>
#include <iostream>

#define START_M(name) \
    std::chrono::time_point<std::chrono::system_clock> start_name, end_name; \
    start_name = std::chrono::system_clock::now();

#define END_M(name) \
    end_name = std::chrono::system_clock::now(); \
    long elapsed_seconds_name = std::chrono::duration_cast<std::chrono::nanoseconds> (end_name-start_name).           count(); \
    std::cout << elapsed_seconds_name << "\n";
	
__global__ 
void dynamicReverse(int *d, int n)
{
  __shared__ int s[5120];
  int t = threadIdx.x;
  int tr = n-t-1;
  for (int i = 0; i < 21; i++){
	s[t] = d[t];
	__syncthreads();
	d[t] = s[tr];
  }
}

__global__ 
void dynamicReverse(int *d, int n, int *s)
{
  int t = threadIdx.x;
  int tr = n-t-1;
  for (int i = 0; i < 21; i++){
	s[t] = d[t];
	__syncthreads();
	d[t] = s[tr];
  }
}


int main(void)
{
	for (int n = 1; n < 2048; n =  n *2) {

		printf("n %d\n",n);
		printf("shared memory size:  %d B\n",n * sizeof(int));

		int a[n], r[n], d[n];

		for (int i = 0; i < n; i++) {
			a[i] = i;
			r[i] = n-i-1;
			d[i] = 0;
		}

		int *d_d, *s_d;
		cudaMalloc(&d_d, n * sizeof(int)); 
		cudaMalloc(&s_d, n * sizeof(int)); 


		
	printf("Shared\n");
	for (int j = 0; j < 50; j++) {
		{
			START_M(_)
			cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
			dynamicReverse<<<1,n,n*sizeof(int)>>>(d_d, n);
			cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);
			END_M()
			for (int i = 0; i < n; i++) 
				if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)n", i, i, d[i], r[i]);

		}
	}
	printf("Non_Shared\n");
	for (int j = 0; j < 50; j++) {
		{
			START_M()
			cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
			dynamicReverse<<<1,n>>>(d_d, n, s_d);
			cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);
			END_M()
			for (int i = 0; i < n; i++) 
				if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)n", i, i, d[i], r[i]);

		}
	}
		
		cudaFree(s_d);
		cudaFree(d_d);
	}
  //}
  
  //for (int i = 0; i < n; i++) 
  //  if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)n", i, i, d[i], r[i]);
}
