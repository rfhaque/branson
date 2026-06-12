#include "../branson_vector.h"
#include <algorithm>
#include <set>
#include <stdlib.h>
#include <iostream>

#include "config.h"

__global__ void test_warp_atomic_inc_kernel(
    unsigned int* counter,
    unsigned char* predicates,
    unsigned char* results,
    unsigned int* positions,
    int num_threads)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_threads) {
        bool pred = predicates[tid] != 0;
        unsigned int pos;
        bool result = warp_atomic_inc_ballot(counter, pred, pos);
        results[tid] = result ? 1 : 0;
        if (result) {
            positions[tid] = pos;
        }
    }
}

// allocate and copy helper
template<typename T>
inline T* allocAndCopy(branson::vector<T>& host_data) {
  T* device_ptr;
  size_t size = host_data.size() * sizeof(T);
  auto err = cudaMalloc((void **)&device_ptr, size);
  Insist(!err, "error in alloacte");
  err = cudaMemcpy(device_ptr, host_data.data(), size, cudaMemcpyHostToDevice);
  Insist(!err, "error in copy");
  return device_ptr;
}

// copy back and free helper
template<typename T>
inline branson::vector<T> copyBackAndFree(T* device_ptr, size_t count) {
  branson::vector<T> host_data(count);
  size_t size = count * sizeof(T);
  auto err = cudaMemcpy(host_data.data(), device_ptr, size, cudaMemcpyDeviceToHost);
  Insist(!err, "error in copy");
  err = cudaFree(device_ptr);
  Insist(!err, "error in free");
  return host_data;
}

// Test 1: All threads in a warp with pred=true
int test_inc_ballot_all_true() {
  constexpr int n_threads = WARP_SIZE; // One warp
  int n_fail = 0;

  // Setup predicates (all true)
  branson::vector<unsigned char> predicates(n_threads, 1);
  branson::vector<unsigned char> results(n_threads, 0);
  branson::vector<unsigned int> positions(n_threads, 0);

  std::cout<<"alloc"<<std::endl;
  unsigned int* d_counter;
  auto err = cudaMalloc((void**)&d_counter, sizeof(unsigned int));
  Insist(!err, "error in malloc");
  err = cudaMemset(d_counter, 0, sizeof(unsigned int));
  Insist(!err, "error in memset");

  auto d_predicates = allocAndCopy(predicates);
  auto d_results = allocAndCopy(results);
  auto d_positions = allocAndCopy(positions);

  std::cout<<"kernel"<<std::endl;
  // Launch kernel
  test_warp_atomic_inc_kernel<<<1, WARP_SIZE>>>(
      d_counter, d_predicates, d_results, d_positions, n_threads);

  auto sync_error = cudaDeviceSynchronize();
  Insist(!sync_error, "error in synchronize");

  std::cout<<"copy back"<<std::endl;
  // Copy results back
  auto h_results = copyBackAndFree(d_results, n_threads);
  auto h_positions = copyBackAndFree(d_positions, n_threads);

  unsigned int final_counter;
  err = cudaMemcpy(&final_counter, d_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  Insist(!err, "error in copy");
  err = cudaFree(d_counter);
  Insist(!err, "error in free");
  err = cudaFree(d_predicates);
  Insist(!err, "error in free");

  // Verify all threads returned true
  for (int i = 0; i < n_threads; i++) {
    if(!h_results[i]) {
      n_fail++;
      std::cout<< "Thread " << i << " should return true";
    }
  }

  // Verify counter was incremented by number of threads
  if(final_counter != n_threads) n_fail++;

  // Verify all positions are unique and in range [0, n_threads]
  std::set<unsigned int> unique_positions(h_positions.begin(), h_positions.end());
  if(unique_positions.size() !=  n_threads) n_fail++;
  if(*unique_positions.begin() != 0u) n_fail++;
  if(*unique_positions.rbegin() != n_threads-1) n_fail++;
  return n_fail;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank;
  int n_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

  // require one rank for this test
  if(n_ranks > 1) {
    std::cout<<"Test must be run with one rank, exiting...";
    std::exit(EXIT_FAILURE);
  }

  // require GPU avaialble for this test
  int device_count = 0;
  #ifdef USE_GPU
    auto err= cudaGetDeviceCount(&device_count);
    Insist(!err, "error in get device count");
  #endif

  if (device_count == 0) {
    std::cout<<"Test must be run with one rank, exiting...";
    std::exit(EXIT_FAILURE);
  }

  int n_fail{0};

  std::cout<<"About to run test_inc_ballot_all_true"<<std::endl;
  if (rank ==0) {
    n_fail += test_inc_ballot_all_true();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return n_fail;
}
