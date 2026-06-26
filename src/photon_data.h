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
    photons(photons_in),
    photon_ptr(nullptr), cell_ID_ptr(nullptr), group_ptr(nullptr),
    source_type_ptr(nullptr),
    descriptors_ptr(nullptr),
    pos_ptr(nullptr),
    angle_ptr(nullptr),
    E_ptr(nullptr),
    E0_ptr(nullptr),
    life_dx_ptr(nullptr),
    RNG_ptr(nullptr)
  {
#ifdef USE_GPU
    auto n_photons = photons.size();
    if constexpr (std::is_same_v<Census_T, std::vector<Photon>>) {
      // Allocate and copy photons
      Photon *device_photons_ptr;
      auto alloc_err = cudaMalloc((void **)&device_photons_ptr, sizeof(Photon) * n_photons);
      Insist(!alloc_err, "CUDA/HIP error allocating photons");
      auto copy_err = cudaMemcpy(device_photons_ptr, photons.data(), sizeof(Photon) * n_photons, cudaMemcpyHostToDevice);
      Insist(!copy_err, "CUDA/HIP error copying photons to device");
    } else {
      // Allocate SoA data on GPU
      auto malloc_err = cudaMalloc((void**)&cell_ID_ptr, n_photons * sizeof(uint32_t));
      if (malloc_err) std::cout<<"Error allocating cell_ID_ptr"<<std::endl;
      auto copy_err = cudaMemcpy(cell_ID_ptr, photons.cell_ID.data(), n_photons * sizeof(uint32_t), cudaMemcpyHostToDevice);
      if (copy_err) std::cout<<"Error copying cell_ID_ptr"<<std::endl;

      malloc_err = cudaMalloc((void**)&group_ptr, n_photons * sizeof(uint32_t));
      if (malloc_err) std::cout<<"Error allocating group_ptr"<<std::endl;
      copy_err = cudaMemcpy(group_ptr, photons.group.data(), n_photons * sizeof(uint32_t), cudaMemcpyHostToDevice);
      if (copy_err) std::cout<<"Error copying group_ptr"<<std::endl;

      malloc_err = cudaMalloc((void**)&descriptors_ptr, n_photons * sizeof(unsigned char));
      if (malloc_err) std::cout<<"Error allocating descriptors_ptr"<<std::endl;
      copy_err = cudaMemcpy(descriptors_ptr, photons.descriptors.data(), n_photons * sizeof(unsigned char), cudaMemcpyHostToDevice);
      if (copy_err) std::cout<<"Error copying decsriptors_ptr"<<std::endl;

      malloc_err = cudaMalloc((void**)&pos_ptr, n_photons * sizeof(std::array<double, 3>));
      if (malloc_err) std::cout<<"Error allocating pos_ptr"<<std::endl;
      copy_err = cudaMemcpy(pos_ptr, photons.pos.data(), n_photons * sizeof(std::array<double, 3>), cudaMemcpyHostToDevice);
      if (copy_err) std::cout<<"Error copying pos_ptr"<<std::endl;

      malloc_err = cudaMalloc((void**)&angle_ptr, n_photons * sizeof(std::array<double, 3>));
      if (malloc_err) std::cout<<"Error allocating angle_ptr"<<std::endl;
      copy_err = cudaMemcpy(angle_ptr, photons.angle.data(), n_photons * sizeof(std::array<double, 3>), cudaMemcpyHostToDevice);
      if (copy_err) std::cout<<"Error copying angle_ptr"<<std::endl;

      malloc_err = cudaMalloc((void**)&E_ptr, n_photons * sizeof(double));
      if (malloc_err) std::cout<<"Error allocating E_ptr"<<std::endl;
      copy_err = cudaMemcpy(E_ptr, photons.E.data(), n_photons * sizeof(double), cudaMemcpyHostToDevice);
      if (copy_err) std::cout<<"Error copying E_ptr"<<std::endl;

      malloc_err = cudaMalloc((void**)&E0_ptr, n_photons * sizeof(double));
      if (malloc_err) std::cout<<"Error allocating E0_ptr"<<std::endl;
      copy_err = cudaMemcpy(E0_ptr, photons.E0.data(), n_photons * sizeof(double), cudaMemcpyHostToDevice);
      if (copy_err) std::cout<<"Error copying E0_ptr"<<std::endl;

      malloc_err = cudaMalloc((void**)&life_dx_ptr, n_photons * sizeof(double));
      if (malloc_err) std::cout<<"Error allocating life_dx_ptr"<<std::endl;
      copy_err = cudaMemcpy(life_dx_ptr, photons.life_dx.data(), n_photons * sizeof(double), cudaMemcpyHostToDevice);
      if (copy_err) std::cout<<"Error copying life_dx_ptr"<<std::endl;

      malloc_err = cudaMalloc((void**)&RNG_ptr, n_photons * sizeof(RNG));
      if (malloc_err) std::cout<<"Error allocating RNG_ptr"<<std::endl;
      copy_err = cudaMemcpy(RNG_ptr, photons.rng.data(), n_photons * sizeof(RNG), cudaMemcpyHostToDevice);
      if (copy_err) std::cout<<"Error copying RNG_ptr"<<std::endl;
    }
#else
    if constexpr (std::is_same_v<Census_T, std::vector<Photon>>) {
      photon_ptr = photons.data();
    }
    else {
      cell_ID_ptr = photons.cell_ID.data();
      group_ptr = photons.group.data();
      source_type_ptr = photons.source_type.data();
      descriptors_ptr = photons.descriptors.data();
      pos_ptr = photons.pos.data();
      angle_ptr = photons.angle.data();
      E_ptr = photons.E.data();
      E0_ptr = photons.E0.data();
      life_dx_ptr = photons.life_dx.data();
      RNG_ptr = photons.rng.data();
    }
#endif
  }

  //! Destructor
  ~Photon_Data() {
    #ifdef USE_GPU
    if (cell_ID_ptr) {
      // free for SoA
      auto free_err = cudaFree(cell_ID_ptr);
      if (free_err) std::cout<<"Error freeing cell_ID_ptr"<<std::endl;
      free_err = cudaFree(group_ptr);
      if (free_err) std::cout<<"Error freeing group_ptr"<<std::endl;
      free_err = cudaFree(descriptors_ptr);
      if (free_err) std::cout<<"Error freeing descriptors_ptr"<<std::endl;
      free_err = cudaFree(pos_ptr);
      if (free_err) std::cout<<"Error freeing pos_ptr"<<std::endl;
      free_err = cudaFree(angle_ptr);
      if (free_err) std::cout<<"Error freeing angle_ptr"<<std::endl;
      free_err = cudaFree(E_ptr);
      if (free_err) std::cout<<"Error freeing E_ptr"<<std::endl;
      free_err = cudaFree(E0_ptr);
      if (free_err) std::cout<<"Error freeing E0_ptr"<<std::endl;
      free_err = cudaFree(life_dx_ptr);
      if (free_err) std::cout<<"Error freeing life_dx_ptr"<<std::endl;
      free_err = cudaFree(RNG_ptr);
      if (free_err) std::cout<<"Error freeing RNG_ptr"<<std::endl;
    }
    // free for AoS
    if (photon_ptr) {
      auto free_err = cudaFree(photon_ptr);
      Insist(!free_err, "error freeing device_photons_ptr");
    }
    #endif
  }

  void sync() {
    // if device pointers aren't allocated don't need to do anything as CPU pointer is underlying vector data
    #ifdef USE_GPU
    auto n_photons = photons.size();
    if constexpr (std::is_same_v<Census_T, std::vector<Photon>>) {
      // Copy AoS data back to host
      auto copy_err = cudaMemcpy(photons.data(), photon_ptr, n_photons * sizeof(Photon), cudaMemcpyDeviceToHost);
      Insist(!copy_err, "CUDA/HIP error copying photons back to host");
    }
    else {
      // Copy SoA data back to host
      auto copy_err = cudaMemcpy(photons.cell_ID.data(), cell_ID_ptr, n_photons * sizeof(uint32_t), cudaMemcpyDeviceToHost);
      Insist(!copy_err, "Error in copying cell_ID_ptr");
      copy_err = cudaMemcpy(photons.group.data(), group_ptr, n_photons * sizeof(uint32_t), cudaMemcpyDeviceToHost);
      Insist(!copy_err, "Error in copying group_ptr");
      copy_err = cudaMemcpy(photons.descriptors.data(), descriptors_ptr, n_photons * sizeof(unsigned char), cudaMemcpyDeviceToHost);
      Insist(!copy_err, "Error in copying descriptors_ptr");
      copy_err = cudaMemcpy(photons.pos.data(), pos_ptr, n_photons * sizeof(std::array<double, 3>), cudaMemcpyDeviceToHost);
      Insist(!copy_err, "Error in copying pos_ptr");
      copy_err = cudaMemcpy(photons.angle.data(), angle_ptr, n_photons * sizeof(std::array<double, 3>), cudaMemcpyDeviceToHost);
      Insist(!copy_err, "Error in copying angle_ptr");
      copy_err = cudaMemcpy(photons.E.data(), E_ptr, n_photons * sizeof(double), cudaMemcpyDeviceToHost);
      Insist(!copy_err, "Error in copying E_ptr");
      copy_err = cudaMemcpy(photons.life_dx.data(), life_dx_ptr, n_photons * sizeof(double), cudaMemcpyDeviceToHost);
      Insist(!copy_err, "Error in copying life_dx_ptr");
      copy_err = cudaMemcpy(photons.rng.data(), RNG_ptr, n_photons * sizeof(RNG), cudaMemcpyDeviceToHost);
      Insist(!copy_err, "Error in copying RNG_ptr");
    }
    #endif
  }

  //--------------------------------------------------------------------------//
  // member data                                                              //
  //--------------------------------------------------------------------------//
  Census_T &photons;
  Photon * photon_ptr;
  uint32_t *cell_ID_ptr;
  uint32_t *group_ptr;
  uint32_t *source_type_ptr;
  unsigned char *descriptors_ptr;
  std::array<double, 3> *pos_ptr;
  std::array<double, 3> *angle_ptr;
  double *E_ptr;
  double *E0_ptr;
  double *life_dx_ptr;
  RNG *RNG_ptr;
};

#endif // photon_data_h_

