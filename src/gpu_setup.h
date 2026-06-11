//--------------------------------------------*-C++-*---------------------------------------------//
/*!
 * \file   gpu_setup.h
 * \author Alex Long
 * \date   Oct 20 2022
 * \brief  Struct that holds device pointers to particles, mesh and tallies
 * \note   Copyright (C) 2022 Triad National Security, LLC., All rights reserved.
 */
//------------------------------------------------------------------------------------------------//

#ifndef gpu_setup_h_
#define gpu_setup_h_

#include <vector>

#include "cell.h"
#include "config.h"

template <typename Census_T>
class GPU_Setup {

public:
  //! Constructor
  GPU_Setup(const int rank, const int n_ranks, const bool use_gpu_transporter, const std::vector<Cell> &cpu_cells, uint64_t n_user_photons)
    : m_use_gpu_transporter(use_gpu_transporter), device_cells_ptr(nullptr)
  {
#ifdef USE_GPU
    if(m_use_gpu_transporter) {
      // MPI rank to GPU mapping
      set_device_ID(rank, n_ranks);

      std::cout<<"Allocating and transferring "<<cpu_cells.size()<<" cell(s) to the GPU"<<std::endl;
      // allocate and copy cells
      cudaError_t err = cudaMalloc((void **)&device_cells_ptr, sizeof(Cell) * cpu_cells.size());
      Insist(!err, "CUDA/HIP error in allocating cells data");
      err = cudaMemcpy(device_cells_ptr, cpu_cells.data(), sizeof(Cell) * cpu_cells.size(),
                       cudaMemcpyHostToDevice);
      Insist(!err, "CUDA/HIP error in copying cells data");
    }
#endif
    if constexpr (std::is_same_v<Census_T, std::vector<Photon>>) {
#ifdef caliper_FOUND
    CALI_MARK_BEGIN("gpu_setup_0");
#endif
      census_photons.reserve(n_user_photons);
#ifdef caliper_FOUND
    CALI_MARK_END("gpu_setup_0");
#endif
    } else {
#ifdef caliper_FOUND
    CALI_MARK_BEGIN("gpu_setup_1");
#endif
      census_photons.cell_ID.reserve(n_user_photons);
      census_photons.group.reserve(n_user_photons);
      census_photons.source_type.reserve(n_user_photons);
      census_photons.descriptors.reserve(n_user_photons);
      census_photons.pos.reserve(n_user_photons);
      census_photons.angle.reserve(n_user_photons);
      census_photons.E.reserve(n_user_photons);
      census_photons.E0.reserve(n_user_photons);
      census_photons.life_dx.reserve(n_user_photons);
      census_photons.rng.reserve(n_user_photons);
#ifdef caliper_FOUND
    CALI_MARK_END("gpu_setup_1");
#endif
    }
  }

  //! Destructor
  ~GPU_Setup() {
#ifdef USE_GPU
    if(m_use_gpu_transporter) {
      auto free_err = cudaFree(device_cells_ptr);
      Insist(!free_err, "Error in freeing");
    }
#endif
  }

  Census_T & get_census_photons() {return census_photons;}
  Cell *get_device_cells_ptr() const {return device_cells_ptr;}
  bool use_gpu_transporter() const {return m_use_gpu_transporter;}

private:

//------------------------------------------------------------------------------------------------//
/*!
 * \brief Assign GPUs to MPI ranks
 *
 * If there are more ranks on a node than GPUs available on a node, multiple ranks will be allowed
 * to use the same GPU.
 *
 * \param[in] rank MPI rank of calling processor
 * \param[in] n_ranks total number of MPI ranks
 */
void set_device_ID(const int rank, const int n_ranks) {
#ifdef USE_GPU
  // device set
  int n_devices;
  auto err_count = cudaGetDeviceCount(&n_devices);
  Insist(!err_count, "error in device count");
  int my_device = 0;
  if (n_ranks <= n_devices)
    my_device = rank;
  else
    my_device = rank % n_devices;
  auto err_set = cudaSetDevice(my_device);
  Insist(!err_set, "error in device set");
  int my_bus_id = 0;
  auto attribute_err = cudaDeviceGetAttribute(&my_bus_id, cudaDevAttrPciBusId, my_device);
  Insist(!attribute_err, "error in device attribute");
  std::cout << "N devices: " << n_devices << std::endl;
  std::cout << "rank: " << rank << ", device: " << my_device << " bus id: ";
  std::cout << my_bus_id << std::endl;
#endif
}

  bool m_use_gpu_transporter;
  Cell *device_cells_ptr;
  Census_T census_photons;
};

#endif // gpu_setup_h_
//----------------------------------------------------------------------------//
// end of gpu_setup.h
//----------------------------------------------------------------------------//

