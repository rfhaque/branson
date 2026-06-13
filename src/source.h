//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   source.h
 * \author Alex Long
 * \date   December 2 2015
 * \brief  Allows transport function to create particles when needed
 * \note   Copyright (C) 2017 Los Alamos National Security, LLC.
 *         All rights reserved
 */
//---------------------------------------------------------------------------//

#ifndef source_h_
#define source_h_

#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "timer.h"
#include "gpu_setup.h"
#include "constants.h"
#include "cell.h"
#include "mesh.h"
#include "photon.h"
#include "sampling_functions.h"
#include "temporary_arrays.h"

GPU_KERNEL void make_source_photons( Cell  const * const cells,  const double dt, const uint32_t seed, uint64_t const * const photon_stream_numbers, double const * const photon_E, uint32_t const * const photon_type,  int const * const photon_source_face,  uint32_t const * const photon_cell_index, const uint64_t n_photons, Photon * const all_photons) {
  using Constants::c;
#ifdef USE_GPU
  int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n_photons) {
#else
  for (size_t i=0;i<n_photons;++i) {
#endif
    auto source_type = photon_type[i];
    RNG rng(seed, photon_stream_numbers[i]);
    // census and emission get uniform position and isotropic angle, source gets position on face and
    // angle surface normal distribution
    auto &cell = cells[photon_cell_index[i]];
    auto pos = (source_type != 2) ? get_uniform_position_in_cell(cell, rng) : get_uniform_position_on_face(cell, rng, photon_source_face[i]);
    auto angle = (source_type != 2) ? get_uniform_angle(rng) : get_source_angle_on_face(rng, photon_source_face[i]);
    double distance_to_census = (source_type ==0) ? c*dt : rng.generate_random_number() * c * dt;
    uint32_t group = std::floor(rng.generate_random_number() * double(BRANSON_N_GROUPS));

    all_photons[i] = Photon(cell.get_global_index(), group, source_type, Constants::event_type::BORN_SOURCE,  pos, angle, photon_E[i], distance_to_census, rng);
  }
#ifdef USE_GPU
  __syncthreads();
#endif
}

// for SoA data structures, cell ID, E, E0 and source_type are already set, use them to sample
// position and angle
GPU_KERNEL void set_source_photons( Cell  const * const cells,  const double dt, const uint32_t seed, uint64_t const * const photon_stream_numbers, uint32_t const * const photon_type, int const * const photon_source_face,  const uint64_t n_photons, uint32_t *photon_cell_index, RNG *rng, std::array< double,3 > *pos, std::array<double,3> *angle, double *life_dx, uint32_t *group) {
  using Constants::c;
#ifdef USE_GPU
  int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n_photons) {
#else
  for (size_t i=0;i<n_photons;++i) {
#endif
    auto source_type = photon_type[i];
    rng[i] = RNG(seed, photon_stream_numbers[i]);
    // census and emission get uniform position and isotropic angle, source gets position on face and
    // angle surface normal distribution
    auto &cell = cells[photon_cell_index[i]];
    pos[i] = (source_type != 2) ? get_uniform_position_in_cell(cell, rng[i]) : get_uniform_position_on_face(cell, rng[i], photon_source_face[i]);
    angle[i] = (source_type != 2) ? get_uniform_angle(rng[i]) : get_source_angle_on_face(rng[i], photon_source_face[i]);
    life_dx[i] = (source_type ==0) ? c*dt : rng[i].generate_random_number() * c * dt;
    group[i] = std::floor(rng[i].generate_random_number() * double(BRANSON_N_GROUPS));
    // cell index comes in as local, is then set to global
    photon_cell_index[i] = cell.get_global_index();
  }
#ifdef USE_GPU
  __syncthreads();
#endif
}

GPU_KERNEL void initialize_source_particle_data(
    Cell const * const cells, double const * const E_cell_census,
    double const * const E_cell_emission, double const * const E_cell_source,
    uint32_t const * const source_census_count,
    uint32_t const * const source_emission_count,
    uint32_t const * const source_boundary_count,
    uint64_t const * const source_offset, const uint64_t census_rank_stream_num_offset,
    const uint64_t cycle_stream_num_offset, const uint64_t rank_stream_num_offset,
    const uint32_t n_local_cells, uint64_t * const photon_stream_nums,
    double * const photon_E, uint32_t * const photon_type,
    int * const photon_source_face, uint32_t * const photon_cell_index) {
#ifdef USE_GPU
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n_local_cells) {
#else
  for (uint32_t i = 0; i < n_local_cells; ++i) {
#endif
    const uint64_t cell_offset = source_offset[i];
    const uint32_t n_census = source_census_count[i];
    const uint32_t n_emission = source_emission_count[i];
    const uint32_t n_boundary = source_boundary_count[i];

    if (n_census > 0) {
      const double photon_census_E = E_cell_census[i] / n_census;
      for (uint32_t p = 0; p < n_census; ++p) {
        const uint64_t photon_index = cell_offset + p;
        photon_stream_nums[photon_index] =
            census_rank_stream_num_offset + photon_index;
        photon_E[photon_index] = photon_census_E;
        photon_type[photon_index] = 0;
        photon_source_face[photon_index] = -1;
        photon_cell_index[photon_index] = i;
      }
    }

    if (n_emission > 0) {
      const double photon_emission_E = E_cell_emission[i] / n_emission;
      const uint64_t emission_offset = cell_offset + n_census;
      for (uint32_t p = 0; p < n_emission; ++p) {
        const uint64_t photon_index = emission_offset + p;
        photon_stream_nums[photon_index] =
            cycle_stream_num_offset + rank_stream_num_offset + photon_index;
        photon_E[photon_index] = photon_emission_E;
        photon_type[photon_index] = 1;
        photon_source_face[photon_index] = -1;
        photon_cell_index[photon_index] = i;
      }
    }

    if (n_boundary > 0) {
      const double photon_source_E = E_cell_source[i] / n_boundary;
      const int face = cells[i].get_source_face();
      const uint64_t source_offset_base = cell_offset + n_census + n_emission;
      for (uint32_t p = 0; p < n_boundary; ++p) {
        const uint64_t photon_index = source_offset_base + p;
        photon_stream_nums[photon_index] =
            cycle_stream_num_offset + rank_stream_num_offset + photon_index;
        photon_E[photon_index] = photon_source_E;
        photon_type[photon_index] = 2;
        photon_source_face[photon_index] = face;
        photon_cell_index[photon_index] = i;
      }
    }
  }
#ifdef USE_GPU
  __syncthreads();
#endif
}

inline uint32_t get_source_photon_count(const double cell_E,
                                        const uint64_t n_user_photons,
                                        const double total_E) {
  if (cell_E <= 0.0) {
    return 0;
  }

  uint32_t n_particles = int(n_user_photons * cell_E / total_E);
  if (n_particles == 0) {
    n_particles = 1;
  }
  return n_particles;
}

// template function for making photons
template <typename Census_T>
void make_photons(const double dt, const Mesh &mesh, const int rank, const uint32_t cycle,
                    const uint32_t seed, const uint64_t n_user_photons,
                    const double total_E, GPU_Setup<Census_T> &gpu_setup,
                    Temporary_Arrays &temporary_arrays) {

  bool make_initial_census_flag{cycle==1};
  auto E_cell_census = mesh.get_census_E();
  auto E_cell_emission = mesh.get_emission_E();
  auto E_cell_source = mesh.get_source_E();
  // for RNG offsets, each cycle allows for one hundred million photons across one hundred
  // thousand ranks, increment the ten trillon place for the next cycle, using cycle plus one for
  // the cycle offset gives the initial census their own space
  const uint64_t census_rank_stream_num_offset{n_user_photons * rank};
  const uint64_t cycle_stream_num_offset{10000000000000UL * static_cast<uint64_t>(cycle)};
  const uint64_t rank_stream_num_offset{n_user_photons * static_cast<uint64_t>(rank)};

  temporary_arrays.reset_source_arrays();

  // figure out how many to make to size all_photons vector
  uint64_t n_photons = 0;
  uint32_t local_cell_index = 0;
  for (auto const &cell : mesh) {
    int i = mesh.get_local_index(cell.get_global_index());
    // initial census
    temporary_arrays.source_census_count[local_cell_index] =
        make_initial_census_flag
            ? get_source_photon_count(E_cell_census[i], n_user_photons, total_E)
            : 0;
    temporary_arrays.source_emission_count[local_cell_index] =
        get_source_photon_count(E_cell_emission[i], n_user_photons, total_E);
    temporary_arrays.source_boundary_count[local_cell_index] =
        get_source_photon_count(E_cell_source[i], n_user_photons, total_E);
    temporary_arrays.source_offset[local_cell_index] = n_photons;
    n_photons += temporary_arrays.source_census_count[local_cell_index];
    n_photons += temporary_arrays.source_emission_count[local_cell_index];
    n_photons += temporary_arrays.source_boundary_count[local_cell_index];
    ++local_cell_index;
  }

  Insist(local_cell_index == temporary_arrays.get_n_source_cells(),
         "Source temporary array size does not match local cell count");

#ifdef caliper_FOUND
    CALI_MARK_BEGIN("vec_make_photons");
#endif
  uint64_t *photon_stream_nums = nullptr;
  double *photon_E = nullptr;
  uint32_t *photon_type = nullptr;
  int *photon_source_face = nullptr;
  uint32_t *photon_cell_index = nullptr;
#ifdef USE_GPU
  if (n_photons > 0) {
    auto alloc_err =
        cudaMalloc((void **)&photon_stream_nums, n_photons * sizeof(uint64_t));
    Insist(!alloc_err, "CUDA/HIP error allocating photon stream numbers");
    alloc_err = cudaMalloc((void **)&photon_E, n_photons * sizeof(double));
    Insist(!alloc_err, "CUDA/HIP error allocating photon energies");
    alloc_err = cudaMalloc((void **)&photon_type, n_photons * sizeof(uint32_t));
    Insist(!alloc_err, "CUDA/HIP error allocating photon types");
    alloc_err =
        cudaMalloc((void **)&photon_source_face, n_photons * sizeof(int));
    Insist(!alloc_err, "CUDA/HIP error allocating photon source faces");
    alloc_err = cudaMalloc((void **)&photon_cell_index,
                           n_photons * sizeof(uint32_t));
    Insist(!alloc_err, "CUDA/HIP error allocating photon cell indices");
  }
#else
  hostMalloc(&photon_stream_nums, n_photons * sizeof(uint64_t));
  hostMalloc(&photon_E, n_photons * sizeof(double));
  hostMalloc(&photon_type, n_photons * sizeof(uint32_t));
  hostMalloc(&photon_source_face, n_photons * sizeof(int));
  hostMalloc(&photon_cell_index, n_photons * sizeof(uint32_t));
#endif
#ifdef caliper_FOUND
    CALI_MARK_END("vec_make_photons");
#endif

#ifdef USE_GPU
  const uint32_t n_local_cells = mesh.get_n_local_cells();
  double *device_E_cell_census_ptr = nullptr;
  double *device_E_cell_emission_ptr = nullptr;
  double *device_E_cell_source_ptr = nullptr;
  uint32_t *device_source_census_count_ptr = nullptr;
  uint32_t *device_source_emission_count_ptr = nullptr;
  uint32_t *device_source_boundary_count_ptr = nullptr;
  uint64_t *device_source_offset_ptr = nullptr;

  if (n_local_cells > 0) {
    auto alloc_err =
        cudaMalloc((void **)&device_E_cell_census_ptr, n_local_cells * sizeof(double));
    Insist(!alloc_err, "CUDA/HIP error allocating census energy data");
    alloc_err = cudaMalloc((void **)&device_E_cell_emission_ptr,
                           n_local_cells * sizeof(double));
    Insist(!alloc_err, "CUDA/HIP error allocating emission energy data");
    alloc_err =
        cudaMalloc((void **)&device_E_cell_source_ptr, n_local_cells * sizeof(double));
    Insist(!alloc_err, "CUDA/HIP error allocating source energy data");
    alloc_err = cudaMalloc((void **)&device_source_census_count_ptr,
                           n_local_cells * sizeof(uint32_t));
    Insist(!alloc_err, "CUDA/HIP error allocating census counts");
    alloc_err = cudaMalloc((void **)&device_source_emission_count_ptr,
                           n_local_cells * sizeof(uint32_t));
    Insist(!alloc_err, "CUDA/HIP error allocating emission counts");
    alloc_err = cudaMalloc((void **)&device_source_boundary_count_ptr,
                           n_local_cells * sizeof(uint32_t));
    Insist(!alloc_err, "CUDA/HIP error allocating source counts");
    alloc_err = cudaMalloc((void **)&device_source_offset_ptr,
                           n_local_cells * sizeof(uint64_t));
    Insist(!alloc_err, "CUDA/HIP error allocating source offsets");

    auto copy_err =
        cudaMemcpy(device_E_cell_census_ptr, E_cell_census.data(),
                   n_local_cells * sizeof(double), cudaMemcpyHostToDevice);
    Insist(!copy_err, "CUDA/HIP error copying census energy data to device");
    copy_err = cudaMemcpy(device_E_cell_emission_ptr, E_cell_emission.data(),
                          n_local_cells * sizeof(double),
                          cudaMemcpyHostToDevice);
    Insist(!copy_err, "CUDA/HIP error copying emission energy data to device");
    copy_err = cudaMemcpy(device_E_cell_source_ptr, E_cell_source.data(),
                          n_local_cells * sizeof(double),
                          cudaMemcpyHostToDevice);
    Insist(!copy_err, "CUDA/HIP error copying source energy data to device");
    copy_err = cudaMemcpy(device_source_census_count_ptr,
                          temporary_arrays.source_census_count,
                          n_local_cells * sizeof(uint32_t),
                          cudaMemcpyHostToDevice);
    Insist(!copy_err, "CUDA/HIP error copying census counts to device");
    copy_err = cudaMemcpy(device_source_emission_count_ptr,
                          temporary_arrays.source_emission_count,
                          n_local_cells * sizeof(uint32_t),
                          cudaMemcpyHostToDevice);
    Insist(!copy_err, "CUDA/HIP error copying emission counts to device");
    copy_err = cudaMemcpy(device_source_boundary_count_ptr,
                          temporary_arrays.source_boundary_count,
                          n_local_cells * sizeof(uint32_t),
                          cudaMemcpyHostToDevice);
    Insist(!copy_err, "CUDA/HIP error copying source counts to device");
    copy_err =
        cudaMemcpy(device_source_offset_ptr, temporary_arrays.source_offset,
                   n_local_cells * sizeof(uint64_t), cudaMemcpyHostToDevice);
    Insist(!copy_err, "CUDA/HIP error copying source offsets to device");
  }

  if (n_photons > 0) {
    const int n_threads = Constants::n_threads_per_block;
    const int n_blocks = (n_local_cells + n_threads - 1) / n_threads;
    initialize_source_particle_data<<<n_blocks, n_threads>>>(
        gpu_setup.get_device_cells_ptr(), device_E_cell_census_ptr,
        device_E_cell_emission_ptr, device_E_cell_source_ptr,
        device_source_census_count_ptr, device_source_emission_count_ptr,
        device_source_boundary_count_ptr, device_source_offset_ptr,
        census_rank_stream_num_offset, cycle_stream_num_offset,
        rank_stream_num_offset, n_local_cells, photon_stream_nums, photon_E,
        photon_type, photon_source_face, photon_cell_index);

    auto kernel_err = cudaGetLastError();
    Insist(!kernel_err, "CUDA/HIP error in source initialization kernel");
    auto sync_err = cudaDeviceSynchronize();
    Insist(!sync_err, "CUDA/HIP error synchronizing source initialization");
  }

  auto free_err = cudaFree(device_E_cell_census_ptr);
  Insist(!free_err, "error freeing device_E_cell_census_ptr");
  free_err = cudaFree(device_E_cell_emission_ptr);
  Insist(!free_err, "error freeing device_E_cell_emission_ptr");
  free_err = cudaFree(device_E_cell_source_ptr);
  Insist(!free_err, "error freeing device_E_cell_source_ptr");
  free_err = cudaFree(device_source_census_count_ptr);
  Insist(!free_err, "error freeing device_source_census_count_ptr");
  free_err = cudaFree(device_source_emission_count_ptr);
  Insist(!free_err, "error freeing device_source_emission_count_ptr");
  free_err = cudaFree(device_source_boundary_count_ptr);
  Insist(!free_err, "error freeing device_source_boundary_count_ptr");
  free_err = cudaFree(device_source_offset_ptr);
  Insist(!free_err, "error freeing device_source_offset_ptr");
#else
  for (uint32_t i = 0; i < temporary_arrays.get_n_source_cells(); ++i) {
    const uint64_t cell_offset = temporary_arrays.source_offset[i];
    const uint32_t n_census = temporary_arrays.source_census_count[i];
    const uint32_t n_emission = temporary_arrays.source_emission_count[i];
    const uint32_t n_boundary = temporary_arrays.source_boundary_count[i];

    if (n_census > 0) {
      const double photon_census_E = E_cell_census[i] / n_census;
      for (uint32_t p = 0; p < n_census; ++p) {
        const uint64_t photon_index = cell_offset + p;
        photon_stream_nums[photon_index] =
            census_rank_stream_num_offset + photon_index;
        photon_E[photon_index] = photon_census_E;
        photon_type[photon_index] = 0;
        photon_source_face[photon_index] = -1;
        photon_cell_index[photon_index] = i;
      }
    }

    if (n_emission > 0) {
      const double photon_emission_E = E_cell_emission[i] / n_emission;
      const uint64_t emission_offset = cell_offset + n_census;
      for (uint32_t p = 0; p < n_emission; ++p) {
        const uint64_t photon_index = emission_offset + p;
        photon_stream_nums[photon_index] =
            cycle_stream_num_offset + rank_stream_num_offset + photon_index;
        photon_E[photon_index] = photon_emission_E;
        photon_type[photon_index] = 1;
        photon_source_face[photon_index] = -1;
        photon_cell_index[photon_index] = i;
      }
    }

    if (n_boundary > 0) {
      const double photon_source_E = E_cell_source[i] / n_boundary;
      const int face = mesh.get_cells()[i].get_source_face();
      const uint64_t source_offset_base = cell_offset + n_census + n_emission;
      for (uint32_t p = 0; p < n_boundary; ++p) {
        const uint64_t photon_index = source_offset_base + p;
        photon_stream_nums[photon_index] =
            cycle_stream_num_offset + rank_stream_num_offset + photon_index;
        photon_E[photon_index] = photon_source_E;
        photon_type[photon_index] = 2;
        photon_source_face[photon_index] = face;
        photon_cell_index[photon_index] = i;
      }
    }
  }
#endif

  uint64_t *device_photon_stream_nums_ptr = photon_stream_nums;
  double *device_photon_E_ptr = photon_E;
  uint32_t *device_source_type_ptr = photon_type;
  int *device_photon_source_face_ptr = photon_source_face;
  uint32_t *device_cell_index_ptr = photon_cell_index;

  if (n_photons == 0) {
#ifndef USE_GPU
    hostFree(photon_stream_nums);
    hostFree(photon_E);
    hostFree(photon_type);
    hostFree(photon_source_face);
    hostFree(photon_cell_index);
#endif
    return;
  }
  // where Aos and SoA diverge--input data above is used for both, but AoS makes its array of
  // photons to be populated by GPU kernel now, SoA can use some of these input structs directly

  if constexpr(std::is_same_v<Census_T, std::vector<Photon>>) {
    #ifdef USE_GPU
    // Allocate photon data
    Photon *device_photon_ptr;
    auto alloc_err = cudaMalloc((void **)&device_photon_ptr, sizeof(Photon) * n_photons);
    Insist(!alloc_err, "CUDA/HIP error allocating photons");

    #else
    std::vector<Photon> &census_photons = gpu_setup.get_census_photons();
    auto n_census_photons = census_photons.size();
    census_photons.resize(n_census_photons + n_photons);
    Photon *device_photon_ptr = census_photons.data() + n_census_photons;
    #endif

    #ifdef USE_GPU
    // Kernel settings
    int n_threads = Constants::n_threads_per_block;
    int n_blocks = (n_photons + n_threads - 1) / n_threads;
    make_source_photons<<<n_blocks, n_threads>>>(gpu_setup.get_device_cells_ptr(), dt, seed, device_photon_stream_nums_ptr, device_photon_E_ptr, device_source_type_ptr, device_photon_source_face_ptr, device_cell_index_ptr, n_photons, device_photon_ptr);

    auto kernel_err = cudaGetLastError();
    Insist(!kernel_err, "CUDA/HIP error in source kernel launch");
    auto sync_err = cudaDeviceSynchronize();
    Insist(!sync_err, "CUDA/HIP error synchronizing after source kernel");

    std::vector<Photon> &census_photons = gpu_setup.get_census_photons();
    auto n_census_photons = census_photons.size();
    census_photons.resize(n_census_photons + n_photons);

    // Copy photons back to host
    auto copy_err = cudaMemcpy(census_photons.data() + n_census_photons, device_photon_ptr, n_photons * sizeof(Photon), cudaMemcpyDeviceToHost);
    Insist(!copy_err, "CUDA/HIP error copying photons back to host");

    // Free device memory specific to AoS
    auto free_err = cudaFree(device_photon_ptr);
    Insist(!free_err, "error freeing device_photon_ptr");
    #else

    make_source_photons(mesh.get_const_cells_ptr(), dt, seed, device_photon_stream_nums_ptr, device_photon_E_ptr, device_source_type_ptr, device_photon_source_face_ptr, device_cell_index_ptr, n_photons, device_photon_ptr);

    #endif
  }
  else {
    #ifdef USE_GPU
    // Allocate photon data
    RNG *device_rng_ptr;
    auto alloc_err = cudaMalloc((void **)&device_rng_ptr, sizeof(RNG) * n_photons);
    Insist(!alloc_err, "CUDA/HIP error allocating rng");

    std::array<double,3> *device_pos_ptr;
    alloc_err = cudaMalloc((void **)&device_pos_ptr, sizeof(std::array<double,3>) * n_photons);
    Insist(!alloc_err, "CUDA/HIP error allocating position");

    std::array<double,3> *device_angle_ptr;
    alloc_err = cudaMalloc((void **)&device_angle_ptr, sizeof(std::array<double,3>) * n_photons);
    Insist(!alloc_err, "CUDA/HIP error allocating angle");

    double *device_life_dx_ptr;
    alloc_err = cudaMalloc((void **)&device_life_dx_ptr, sizeof(double) * n_photons);
    Insist(!alloc_err, "CUDA/HIP error allocating life_dx");

    uint32_t *device_group_ptr;
    alloc_err = cudaMalloc((void **)&device_group_ptr,  sizeof(uint32_t) * n_photons);
    Insist(!alloc_err, "CUDA/HIP error allocating group");
    #else
    PhotonArray &census_photons = gpu_setup.get_census_photons();
    auto n_census_photons = census_photons.size();
    census_photons.resize(n_census_photons + n_photons);
    RNG *device_rng_ptr = census_photons.rng.data() + n_census_photons;
    std::array<double,3> *device_pos_ptr = census_photons.pos.data() + n_census_photons;
    std::array<double,3> *device_angle_ptr = census_photons.angle.data() + n_census_photons;
    double *device_life_dx_ptr = census_photons.life_dx.data() + n_census_photons;
    uint32_t *device_group_ptr = census_photons.group.data() + n_census_photons;
    #endif

    #ifdef USE_GPU
    // Kernel settings
    int n_threads = Constants::n_threads_per_block;
    int n_blocks = (n_photons + n_threads - 1) / n_threads;
    set_source_photons<<<n_blocks, n_threads>>>(gpu_setup.get_device_cells_ptr(), dt, seed,
      device_photon_stream_nums_ptr, device_source_type_ptr, device_photon_source_face_ptr,
      n_photons, device_cell_index_ptr, device_rng_ptr, device_pos_ptr, device_angle_ptr,
      device_life_dx_ptr, device_group_ptr);

    auto kernel_err = cudaGetLastError();
    Insist(!kernel_err, "CUDA/HIP error in source kernel launch");
    auto sync_err = cudaDeviceSynchronize();
    Insist(!sync_err, "CUDA/HIP error synchronizing after source kernel");

#ifdef caliper_FOUND
    CALI_MARK_BEGIN("vec_resize_make_photons");
#endif
    PhotonArray &census_photons = gpu_setup.get_census_photons();
    auto n_census_photons = census_photons.size();
    census_photons.resize(n_census_photons + n_photons);
#ifdef caliper_FOUND
    CALI_MARK_END("vec_resize_make_photons");
#endif

    // Copy photon arrays back to host photon array object
    // CPU -> CPU copies
    auto copy_err = cudaMemcpy(census_photons.E.data() + n_census_photons,
                               photon_E, n_photons * sizeof(double),
                               cudaMemcpyDeviceToHost);
    Insist(!copy_err, "CUDA/HIP error copying photon E to host");
    std::copy(census_photons.E.begin() + n_census_photons, census_photons.E.end(),
              census_photons.E0.begin() + n_census_photons);
    copy_err = cudaMemcpy(census_photons.source_type.data() + n_census_photons,
                          photon_type, n_photons * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost);
    Insist(!copy_err, "CUDA/HIP error copying photon source_type to host");
    std::fill(census_photons.descriptors.begin() + n_census_photons, census_photons.descriptors.end(), Constants::event_type::BORN_SOURCE);
    // GPU -> CPU copies
    copy_err = cudaMemcpy(census_photons.cell_ID.data() + n_census_photons, device_cell_index_ptr, n_photons * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    Insist(!copy_err, "CUDA/HIP error copying cell indices back to host");
    copy_err = cudaMemcpy(census_photons.group.data() + n_census_photons, device_group_ptr, n_photons * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    Insist(!copy_err, "CUDA/HIP error copying group back to host");
    copy_err = cudaMemcpy(census_photons.pos.data() + n_census_photons, device_pos_ptr, n_photons * sizeof(std::array<double,3>), cudaMemcpyDeviceToHost);
    Insist(!copy_err, "CUDA/HIP error copying pos to host");
    copy_err = cudaMemcpy(census_photons.angle.data() + n_census_photons, device_angle_ptr, n_photons * sizeof(std::array<double,3>), cudaMemcpyDeviceToHost);
    Insist(!copy_err, "CUDA/HIP error copying angle to host");
    copy_err = cudaMemcpy(census_photons.life_dx.data() + n_census_photons, device_life_dx_ptr, n_photons * sizeof(double), cudaMemcpyDeviceToHost);
    Insist(!copy_err, "CUDA/HIP error copying life_dx to host");
    copy_err = cudaMemcpy(census_photons.rng.data() + n_census_photons, device_rng_ptr, n_photons * sizeof(RNG), cudaMemcpyDeviceToHost);
    Insist(!copy_err, "CUDA/HIP error copying rng back to host");

    // Free device memory specific to SoA
    auto free_err = cudaFree(device_group_ptr );
    Insist(!free_err, "error freeing device_group_ptr");
    free_err = cudaFree(device_pos_ptr );
    Insist(!free_err, "error freeing device_pos_ptr");
    free_err = cudaFree(device_angle_ptr );
    Insist(!free_err, "error freeing device_angle_ptr");
    free_err = cudaFree(device_life_dx_ptr );
    Insist(!free_err, "error freeing device_lift_dx_ptr");
    free_err = cudaFree(device_rng_ptr );
    Insist(!free_err, "error freeing device_rng_ptr");
  #else
    set_source_photons(mesh.get_const_cells_ptr(), dt, seed,
      device_photon_stream_nums_ptr, device_source_type_ptr, device_photon_source_face_ptr,
      n_photons, device_cell_index_ptr, device_rng_ptr, device_pos_ptr, device_angle_ptr,
      device_life_dx_ptr, device_group_ptr);

    std::copy( photon_cell_index , photon_cell_index+n_photons, census_photons.cell_ID.begin() + n_census_photons);
    std::copy( photon_E, photon_E+n_photons,  census_photons.E.begin() + n_census_photons);
    std::copy(photon_E, photon_E+n_photons, census_photons.E0.begin() + n_census_photons);
    std::copy( photon_type, photon_type+n_photons, census_photons.source_type.begin() + n_census_photons);
    std::fill(census_photons.descriptors.begin() + n_census_photons, census_photons.descriptors.end(), Constants::event_type::BORN_SOURCE);
  #endif
  }

#ifdef USE_GPU
  // Free device memory
  auto free_err = cudaFree(device_photon_stream_nums_ptr);
  Insist(!free_err, "error freeing device_photon_seed_ptr");
  free_err = cudaFree(device_photon_E_ptr);
  Insist(!free_err, "error freeing device_photon_E_ptr");
  free_err = cudaFree(device_source_type_ptr);
  Insist(!free_err, "error freeing device_source_type_ptr");
  free_err = cudaFree(device_photon_source_face_ptr);
  Insist(!free_err, "error freeing device_source_face_ptr");
  free_err = cudaFree(device_cell_index_ptr);
  Insist(!free_err, "error freeing device_cell_index_ptr");
#else
  hostFree(photon_stream_nums);
  hostFree(photon_E);
  hostFree(photon_type);
  hostFree(photon_source_face);
  hostFree(photon_cell_index);
#endif
}


#endif // source_h_
//----------------------------------------------------------------------------//
// end of source.h
//----------------------------------------------------------------------------//
