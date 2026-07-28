//----------------------------------*-C++-*-----------------------------------//
/*!
 * \file   photon_data.h
 * \author Alex Long
 * \date   July 18 2014
 * \brief  Holds values and functions needed for transporting photon_data
 * \note   Copyright (C) 2017 Los Alamos National Security, LLC.
 *         All rights reserved
 */
//----------------------------------------------------------------------------//

#ifndef photon_data_h_
#define photon_data_h_

#include <cmath>
#include <iostream>
#include <vector>
#include <array>

#include "constants.h"
#include "config.h"
#include "RNG.h"

//==============================================================================
/*!
 * \class Photon_Data
 * \brief Contains position, direction, cell ID and energy for transport.
 *
 * Holds all of the internal state of a photon_data and provides functions for
 * sorting photons based on census and global cell ID.
 */
//==============================================================================
template <typename Census_T>
class Photon_Data {
public:
  //! Constructor
  Photon_Data(Census_T &photons_in) :
    original_n_photons(photons_in.size()),
    n_photons(photons_in.size()),
    h_photon_ptr(nullptr),
    h_cell_ID_ptr(nullptr), h_group_ptr(nullptr),
    h_source_type_ptr(nullptr),
    h_descriptors_ptr(nullptr),
    h_pos_ptr(nullptr),
    h_angle_ptr(nullptr),
    h_E_ptr(nullptr),
    h_E0_ptr(nullptr),
    h_life_dx_ptr(nullptr),
    h_RNG_ptr(nullptr),
    d_photon_ptr(nullptr),
    d_cell_ID_ptr(nullptr), d_group_ptr(nullptr),
    d_source_type_ptr(nullptr),
    d_descriptors_ptr(nullptr),
    d_pos_ptr(nullptr),
    d_angle_ptr(nullptr),
    d_E_ptr(nullptr),
    d_E0_ptr(nullptr),
    d_life_dx_ptr(nullptr),
    d_RNG_ptr(nullptr),
    d_active_indices(nullptr),
    d_scatter_indices(nullptr),
    d_boundary_indices(nullptr),
    d_census_indices(nullptr),
    d_absorbed_E(nullptr),
    d_track_length_E(nullptr),
    h_initial_indices(n_photons),
    h_zeros(n_photons, 0.0)
  {
    std::iota(h_initial_indices.begin(), h_initial_indices.end(), 0);
    if constexpr (std::is_same_v<Census_T, std::vector<Photon>>) {
      h_photon_ptr = photons_in.data();
    }
    else {
      h_cell_ID_ptr = photons_in.cell_ID.data();
      h_group_ptr = photons_in.group.data();
      h_source_type_ptr = photons_in.source_type.data();
      h_descriptors_ptr = photons_in.descriptors.data();
      h_pos_ptr = photons_in.pos.data();
      h_angle_ptr = photons_in.angle.data();
      h_E_ptr = photons_in.E.data();
      h_E0_ptr = photons_in.E0.data();
      h_life_dx_ptr = photons_in.life_dx.data();
      h_RNG_ptr = photons_in.rng.data();
    }
#ifdef USE_GPU

    // Event-specific index lists
    auto malloc_err = cudaMalloc((void **)&d_active_indices, sizeof(int32_t) * n_photons);
    Insist(!malloc_err, "GPU AoS Malloc: d_active_indices");
    malloc_err = cudaMalloc((void **)&d_scatter_indices, sizeof(int32_t) * n_photons);
    Insist(!malloc_err, "GPU AoS Malloc: d_scatter_indices");
    malloc_err = cudaMalloc((void **)&d_boundary_indices, sizeof(int32_t) * n_photons);
    Insist(!malloc_err, "GPU AoS Malloc: d_boundary_indices");
    malloc_err = cudaMalloc((void **)&d_census_indices, sizeof(int32_t) * n_photons);
    Insist(!malloc_err, "GPU AoS Malloc: d_census_indices");
    malloc_err = cudaMalloc((void **)&d_absorbed_E, sizeof(double) * n_photons);
    Insist(!malloc_err, "GPU AoS Malloc: d_absorbed_E");
    malloc_err = cudaMalloc((void **)&d_track_length_E, sizeof(double) * n_photons);
    Insist(!malloc_err, "GPU AoS Malloc: d_track_length_E");

    if constexpr (std::is_same_v<Census_T, std::vector<Photon>>) {
      // Allocate and copy photons
      malloc_err = cudaMalloc((void **)&d_photon_ptr, sizeof(Photon) * n_photons);
      Insist(!malloc_err, "CUDA/HIP error allocating photons");
      auto copy_err = cudaMemcpy(d_photon_ptr, h_photon_ptr, sizeof(Photon) * n_photons, cudaMemcpyHostToDevice);
      Insist(!copy_err, "CUDA/HIP error copying photons to device");
    } else {
      // Allocate SoA data on GPU
      malloc_err = cudaMalloc((void**)&d_cell_ID_ptr, n_photons * sizeof(uint32_t));
      if (malloc_err) std::cout<<"Error allocating cell_ID_ptr"<<std::endl;
      auto copy_err = cudaMemcpy(d_cell_ID_ptr, h_cell_ID_ptr, n_photons * sizeof(uint32_t), cudaMemcpyHostToDevice);
      if (copy_err) std::cout<<"Error copying cell_ID_ptr"<<std::endl;

      malloc_err = cudaMalloc((void**)&d_group_ptr, n_photons * sizeof(uint32_t));
      if (malloc_err) std::cout<<"Error allocating group_ptr"<<std::endl;
      copy_err = cudaMemcpy(d_group_ptr, h_group_ptr, n_photons * sizeof(uint32_t), cudaMemcpyHostToDevice);
      if (copy_err) std::cout<<"Error copying group_ptr"<<std::endl;

      malloc_err = cudaMalloc((void**)&d_descriptors_ptr, n_photons * sizeof(unsigned char));
      if (malloc_err) std::cout<<"Error allocating descriptors_ptr"<<std::endl;
      copy_err = cudaMemcpy(d_descriptors_ptr, h_descriptors_ptr, n_photons * sizeof(unsigned char), cudaMemcpyHostToDevice);
      if (copy_err) std::cout<<"Error copying decsriptors_ptr"<<std::endl;

      malloc_err = cudaMalloc((void**)&d_pos_ptr, n_photons * sizeof(std::array<double, 3>));
      if (malloc_err) std::cout<<"Error allocating pos_ptr"<<std::endl;
      copy_err = cudaMemcpy(d_pos_ptr, h_pos_ptr, n_photons * sizeof(std::array<double, 3>), cudaMemcpyHostToDevice);
      if (copy_err) std::cout<<"Error copying pos_ptr"<<std::endl;

      malloc_err = cudaMalloc((void**)&d_angle_ptr, n_photons * sizeof(std::array<double, 3>));
      if (malloc_err) std::cout<<"Error allocating angle_ptr"<<std::endl;
      copy_err = cudaMemcpy(d_angle_ptr, h_angle_ptr, n_photons * sizeof(std::array<double, 3>), cudaMemcpyHostToDevice);
      if (copy_err) std::cout<<"Error copying angle_ptr"<<std::endl;

      malloc_err = cudaMalloc((void**)&d_E_ptr, n_photons * sizeof(double));
      if (malloc_err) std::cout<<"Error allocating E_ptr"<<std::endl;
      copy_err = cudaMemcpy(d_E_ptr, h_E_ptr, n_photons * sizeof(double), cudaMemcpyHostToDevice);
      if (copy_err) std::cout<<"Error copying E_ptr"<<std::endl;

      malloc_err = cudaMalloc((void**)&d_E0_ptr, n_photons * sizeof(double));
      if (malloc_err) std::cout<<"Error allocating E0_ptr"<<std::endl;
      copy_err = cudaMemcpy(d_E0_ptr, h_E0_ptr, n_photons * sizeof(double), cudaMemcpyHostToDevice);
      if (copy_err) std::cout<<"Error copying E0_ptr"<<std::endl;

      malloc_err = cudaMalloc((void**)&d_life_dx_ptr, n_photons * sizeof(double));
      if (malloc_err) std::cout<<"Error allocating life_dx_ptr"<<std::endl;
      copy_err = cudaMemcpy(d_life_dx_ptr, h_life_dx_ptr, n_photons * sizeof(double), cudaMemcpyHostToDevice);
      if (copy_err) std::cout<<"Error copying life_dx_ptr"<<std::endl;

      malloc_err = cudaMalloc((void**)&d_RNG_ptr, n_photons * sizeof(RNG));
      if (malloc_err) std::cout<<"Error allocating RNG_ptr"<<std::endl;
      copy_err = cudaMemcpy(d_RNG_ptr, h_RNG_ptr, n_photons * sizeof(RNG), cudaMemcpyHostToDevice);
      if (copy_err) std::cout<<"Error copying RNG_ptr"<<std::endl;

    }
#endif
  }

  //! Destructor
  ~Photon_Data() {
    #ifdef USE_GPU
    auto free_err = cudaFree(d_active_indices);
    if (free_err) std::cout<<"Error freeing active_indices"<<std::endl;
    free_err = cudaFree(d_scatter_indices);
    if (free_err) std::cout<<"Error freeing scatter_indices"<<std::endl;
    free_err = cudaFree(d_boundary_indices);
    if (free_err) std::cout<<"Error freeing boundary_indices"<<std::endl;
    free_err = cudaFree(d_census_indices);
    if (free_err) std::cout<<"Error freeing census_indices"<<std::endl;
    free_err = cudaFree(d_absorbed_E);
    if (free_err) std::cout<<"Error freeing absorbed_E"<<std::endl;
    free_err = cudaFree(d_track_length_E);
    if (free_err) std::cout<<"Error freeing E_track_length_E"<<std::endl;

    if (d_cell_ID_ptr) {
      // free for SoA
      free_err = cudaFree(d_cell_ID_ptr);
      if (free_err) std::cout<<"Error freeing cell_ID_ptr"<<std::endl;
      free_err = cudaFree(d_group_ptr);
      if (free_err) std::cout<<"Error freeing group_ptr"<<std::endl;
      free_err = cudaFree(d_descriptors_ptr);
      if (free_err) std::cout<<"Error freeing descriptors_ptr"<<std::endl;
      free_err = cudaFree(d_pos_ptr);
      if (free_err) std::cout<<"Error freeing pos_ptr"<<std::endl;
      free_err = cudaFree(d_angle_ptr);
      if (free_err) std::cout<<"Error freeing angle_ptr"<<std::endl;
      free_err = cudaFree(d_E_ptr);
      if (free_err) std::cout<<"Error freeing E_ptr"<<std::endl;
      free_err = cudaFree(d_E0_ptr);
      if (free_err) std::cout<<"Error freeing E0_ptr"<<std::endl;
      free_err = cudaFree(d_life_dx_ptr);
      if (free_err) std::cout<<"Error freeing life_dx_ptr"<<std::endl;
      free_err = cudaFree(d_RNG_ptr);
      if (free_err) std::cout<<"Error freeing RNG_ptr"<<std::endl;
    }
    // free for AoS
    if (d_photon_ptr) {
      free_err = cudaFree(d_photon_ptr);
      Insist(!free_err, "error freeing device_photons_ptr");
    }
    #endif
  }

  void sync() {
    // if device pointers aren't allocated don't need to do anything as CPU pointer is underlying vector data
    #ifdef USE_GPU
    if constexpr (std::is_same_v<Census_T, std::vector<Photon>>) {
      // Copy AoS data back to host
      auto copy_err = cudaMemcpy(h_photon_ptr, d_photon_ptr, n_photons * sizeof(Photon), cudaMemcpyDeviceToHost);
      Insist(!copy_err, "CUDA/HIP error copying photons back to host");
    }
    else {
      // Copy SoA data back to host
      auto copy_err = cudaMemcpy(h_cell_ID_ptr, d_cell_ID_ptr, n_photons * sizeof(uint32_t), cudaMemcpyDeviceToHost);
      Insist(!copy_err, "Error in copying cell_ID_ptr");
      copy_err = cudaMemcpy(h_group_ptr, d_group_ptr, n_photons * sizeof(uint32_t), cudaMemcpyDeviceToHost);
      Insist(!copy_err, "Error in copying group_ptr");
      copy_err = cudaMemcpy(h_descriptors_ptr, d_descriptors_ptr, n_photons * sizeof(unsigned char), cudaMemcpyDeviceToHost);
      Insist(!copy_err, "Error in copying descriptors_ptr");
      copy_err = cudaMemcpy(h_pos_ptr, d_pos_ptr, n_photons * sizeof(std::array<double, 3>), cudaMemcpyDeviceToHost);
      Insist(!copy_err, "Error in copying pos_ptr");
      copy_err = cudaMemcpy(h_angle_ptr, d_angle_ptr, n_photons * sizeof(std::array<double, 3>), cudaMemcpyDeviceToHost);
      Insist(!copy_err, "Error in copying angle_ptr");
      copy_err = cudaMemcpy(h_E_ptr, d_E_ptr, n_photons * sizeof(double), cudaMemcpyDeviceToHost);
      Insist(!copy_err, "Error in copying E_ptr");
      copy_err = cudaMemcpy(h_life_dx_ptr, d_life_dx_ptr, n_photons * sizeof(double), cudaMemcpyDeviceToHost);
      Insist(!copy_err, "Error in copying life_dx_ptr");
      copy_err = cudaMemcpy(h_RNG_ptr, d_RNG_ptr, n_photons * sizeof(RNG), cudaMemcpyDeviceToHost);
      Insist(!copy_err, "Error in copying RNG_ptr");
    }
    #endif
  }

  void reset_without_allocating(Census_T &new_photons) {
    n_photons = new_photons.size();
    Insist(n_photons <= original_n_photons, "Error, reset with bigger array than initialized with!");

    if constexpr (std::is_same_v<Census_T, std::vector<Photon>>) {
      h_photon_ptr = new_photons.data();
    }
    else {
      h_cell_ID_ptr = new_photons.cell_ID.data();
      h_group_ptr = new_photons.group.data();
      h_source_type_ptr = new_photons.source_type.data();
      h_descriptors_ptr = new_photons.descriptors.data();
      h_pos_ptr = new_photons.pos.data();
      h_angle_ptr = new_photons.angle.data();
      h_E_ptr = new_photons.E.data();
      h_E0_ptr = new_photons.E0.data();
      h_life_dx_ptr = new_photons.life_dx.data();
      h_RNG_ptr = new_photons.rng.data();
    }
#ifdef USE_GPU
    if constexpr (std::is_same_v<Census_T, std::vector<Photon>>) {
      // Allocate and copy photons
      auto copy_err = cudaMemcpy(d_photon_ptr, h_photon_ptr, sizeof(Photon) * n_photons, cudaMemcpyHostToDevice);
      Insist(!copy_err, "CUDA/HIP error copying photons to device");
    } else {
      auto copy_err = cudaMemcpy(d_cell_ID_ptr, h_cell_ID_ptr, n_photons * sizeof(uint32_t), cudaMemcpyHostToDevice);
      if (copy_err) std::cout<<"Error copying cell_ID_ptr"<<std::endl;

      copy_err = cudaMemcpy(d_group_ptr, h_group_ptr, n_photons * sizeof(uint32_t), cudaMemcpyHostToDevice);
      if (copy_err) std::cout<<"Error copying group_ptr"<<std::endl;

      copy_err = cudaMemcpy(d_descriptors_ptr, h_descriptors_ptr, n_photons * sizeof(unsigned char), cudaMemcpyHostToDevice);
      if (copy_err) std::cout<<"Error copying decsriptors_ptr"<<std::endl;

      copy_err = cudaMemcpy(d_pos_ptr, h_pos_ptr, n_photons * sizeof(std::array<double, 3>), cudaMemcpyHostToDevice);
      if (copy_err) std::cout<<"Error copying pos_ptr"<<std::endl;

      copy_err = cudaMemcpy(d_angle_ptr, h_angle_ptr, n_photons * sizeof(std::array<double, 3>), cudaMemcpyHostToDevice);
      if (copy_err) std::cout<<"Error copying angle_ptr"<<std::endl;

      copy_err = cudaMemcpy(d_E_ptr, h_E_ptr, n_photons * sizeof(double), cudaMemcpyHostToDevice);
      if (copy_err) std::cout<<"Error copying E_ptr"<<std::endl;

      copy_err = cudaMemcpy(d_E0_ptr, h_E0_ptr, n_photons * sizeof(double), cudaMemcpyHostToDevice);
      if (copy_err) std::cout<<"Error copying E0_ptr"<<std::endl;

      copy_err = cudaMemcpy(d_life_dx_ptr, h_life_dx_ptr, n_photons * sizeof(double), cudaMemcpyHostToDevice);
      if (copy_err) std::cout<<"Error copying life_dx_ptr"<<std::endl;

      copy_err = cudaMemcpy(d_RNG_ptr, h_RNG_ptr, n_photons * sizeof(RNG), cudaMemcpyHostToDevice);
      if (copy_err) std::cout<<"Error copying RNG_ptr"<<std::endl;
    }
#endif
  }

  void reset_event_based_data() {
#ifdef USE_GPU
    auto err = cudaMemcpy(d_active_indices, h_initial_indices.data(), sizeof(int32_t) * n_photons, cudaMemcpyHostToDevice);
    Insist(!err, "GPU AoS Memcpy H2D: d_active_indices");
    err = cudaMemcpy(d_absorbed_E, h_zeros.data(), sizeof(double) * n_photons, cudaMemcpyHostToDevice);
    Insist(!err, "GPU AoS Memcpy H2D: d_abs_E");
    err = cudaMemcpy(d_track_length_E, h_zeros.data(), sizeof(double) * n_photons, cudaMemcpyHostToDevice);
    Insist(!err, "GPU AoS Memcpy H2D: d_track_length_E");
#endif
  }

  //--------------------------------------------------------------------------//
  // member data                                                              //
  //--------------------------------------------------------------------------//
  size_t original_n_photons;
  size_t n_photons;
  Photon * h_photon_ptr;
  // host side SoA data
  uint32_t *h_cell_ID_ptr;
  uint32_t *h_group_ptr;
  uint32_t *h_source_type_ptr;
  unsigned char *h_descriptors_ptr;
  std::array<double, 3> *h_pos_ptr;
  std::array<double, 3> *h_angle_ptr;
  double *h_E_ptr;
  double *h_E0_ptr;
  double *h_life_dx_ptr;
  RNG *h_RNG_ptr;

  Photon * d_photon_ptr;
  // device side SoA data
  uint32_t *d_cell_ID_ptr;
  uint32_t *d_group_ptr;
  uint32_t *d_source_type_ptr;
  unsigned char *d_descriptors_ptr;
  std::array<double, 3> *d_pos_ptr;
  std::array<double, 3> *d_angle_ptr;
  double *d_E_ptr;
  double *d_E0_ptr;
  double *d_life_dx_ptr;
  RNG *d_RNG_ptr;

  // device buffer for event-based GPU buffers
  int32_t *d_active_indices;
  int32_t *d_scatter_indices;
  int32_t *d_boundary_indices;
  int32_t *d_census_indices;
  double *d_absorbed_E;
  double *d_track_length_E;
  std::vector<int32_t> h_initial_indices;
  std::vector<double> h_zeros;
};

#endif // photon_data_h_
