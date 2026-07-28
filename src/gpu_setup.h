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
#include "sampling_functions.h"

template <typename Census_T>
class GPU_Setup {

public:
  //! Constructor
  GPU_Setup(const int rank, const int n_ranks, const bool use_gpu_transporter, const std::vector<Cell> &cpu_cells, uint64_t n_user_photons)
    : host_cells_ptr(cpu_cells.data()), m_use_gpu_transporter(use_gpu_transporter), device_cells_ptr(nullptr), n_cells(cpu_cells.size())
  {
      // Precompute emission group data for event-based transport (both CPU and GPU)
      emission_groups.resize(n_cells);
      for (size_t i = 0; i < n_cells; ++i) {
        emission_groups[i] = precompute_emission_group_data(cpu_cells[i]);
      }

      cell_tallies.resize(n_cells);

#ifdef USE_GPU
    if(m_use_gpu_transporter) {
      // MPI rank to GPU mapping
      set_device_ID(rank, n_ranks);

      if (rank ==0) {
        std::cout<<"Allocating and transferring "<<cpu_cells.size()<<" cell(s) to the GPU"<<std::endl;
      }
      // allocate and copy cells
      auto malloc_err = cudaMalloc((void **)&device_cells_ptr, sizeof(Cell) * cpu_cells.size());
      Insist(!malloc_err, "CUDA/HIP error in allocating cells data");
      auto copy_err = cudaMemcpy(device_cells_ptr, cpu_cells.data(), sizeof(Cell) * cpu_cells.size(),
                       cudaMemcpyHostToDevice);
      Insist(!copy_err, "CUDA/HIP error in copying cells data");

      // Allocate and copy cell tallies (zero initialize on device)
      malloc_err = cudaMalloc((void **)&d_cell_tallies, sizeof(Cell_Tally) * cell_tallies.size());
      Insist(!malloc_err, "CUDA/HIP error allocating cell tallies");
      copy_err = cudaMemcpy(d_cell_tallies, cell_tallies.data(), cell_tallies.size()* sizeof(Cell_Tally), cudaMemcpyHostToDevice);
      Insist(!copy_err, "CUDA/HIP error copying cell tallies");

      // Allocate and copy emission Groups
      malloc_err = cudaMalloc((void **)&d_emission_groups, sizeof(EmissionGroupData) * n_cells);
      Insist(!malloc_err, "CUDA/HIP error allocating emission groups");
      copy_err = cudaMemcpy(d_emission_groups, emission_groups.data(), sizeof(EmissionGroupData) * n_cells, cudaMemcpyHostToDevice);
      Insist(!copy_err, "CUDA/HIP error copying emission groups");
    }
#else
    d_emission_groups = emission_groups.data();
    d_cell_tallies = cell_tallies.data();
#endif

    const size_t photon_reserve =
        (n_ranks > 0) ? static_cast<size_t>(n_user_photons / static_cast<uint64_t>(n_ranks)) : 0;
    if constexpr (std::is_same_v<Census_T, std::vector<Photon>>) {
      census_photons.reserve(photon_reserve);
    } else {
      census_photons.cell_ID.reserve(photon_reserve);
      census_photons.group.reserve(photon_reserve);
      census_photons.source_type.reserve(photon_reserve);
      census_photons.descriptors.reserve(photon_reserve);
      census_photons.pos.reserve(photon_reserve);
      census_photons.angle.reserve(photon_reserve);
      census_photons.E.reserve(photon_reserve);
      census_photons.E0.reserve(photon_reserve);
      census_photons.life_dx.reserve(photon_reserve);
      census_photons.rng.reserve(photon_reserve);
    }
  }

  // re-sync data
  void update_cells_and_reset_tallies(const int rank, const std::vector<Cell> &cpu_cells, uint64_t n_user_photons) {
      host_cells_ptr = cpu_cells.data();
      // Precompute emission group data for event-based transport (both CPU and GPU) and reset
      // tallies
      for (size_t i = 0; i < n_cells; ++i) {
        emission_groups[i] = precompute_emission_group_data(cpu_cells[i]);
        cell_tallies[i].zero();
      }


#ifdef USE_GPU
    if(m_use_gpu_transporter) {
      if (rank ==0) {
        std::cout<<"Transferring "<<cpu_cells.size()<<" cell(s) to the GPU"<<std::endl;
      }
      // copy cells
      auto copy_err = cudaMemcpy(device_cells_ptr, cpu_cells.data(), sizeof(Cell) * cpu_cells.size(),
                       cudaMemcpyHostToDevice);
      Insist(!copy_err, "CUDA/HIP error in copying cells data");

      // copy cell tallies (zero initialize on device)
      copy_err = cudaMemcpy(d_cell_tallies, cell_tallies.data(), cell_tallies.size()* sizeof(Cell_Tally), cudaMemcpyHostToDevice);
      Insist(!copy_err, "CUDA/HIP error copying cell tallies");

      // copy emission Groups
      copy_err = cudaMemcpy(d_emission_groups, emission_groups.data(), sizeof(EmissionGroupData) * n_cells, cudaMemcpyHostToDevice);
      Insist(!copy_err, "CUDA/HIP error copying emission groups");
    }
#else
    d_emission_groups = emission_groups.data();
    d_cell_tallies = cell_tallies.data();
#endif

    if constexpr (std::is_same_v<Census_T, std::vector<Photon>>) {
      census_photons.reserve(n_user_photons);
    } else {
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

  void sync_cell_tallies() {
    #ifdef USE_GPU
      auto copy_err = cudaMemcpy(cell_tallies.data(), d_cell_tallies, cell_tallies.size()* sizeof(Cell_Tally), cudaMemcpyDeviceToHost);
      Insist(!copy_err, "CUDA/HIP error copying cell tallies");
    #endif
  }

  Census_T & get_census_photons() {return census_photons;}
  Cell const * const get_host_cells_ptr() const {return host_cells_ptr;}
  Cell *get_device_cells_ptr() const {return device_cells_ptr;}
  bool use_gpu_transporter() const {return m_use_gpu_transporter;}
  EmissionGroupData * get_emission_groups_ptr() const {return d_emission_groups;}
  const std::vector<EmissionGroupData> & get_emission_groups() const {return emission_groups;}
  Cell_Tally * get_device_cell_tallies_ptr() const {return d_cell_tallies;}
  const std::vector<Cell_Tally> & get_cell_tallies() const {return cell_tallies;}
  size_t get_n_cells() {return n_cells;}

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
  if (rank == 0) {
    std::cout << "N devices: " << n_devices << std::endl;
  }
  std::cout << "rank: " << rank << ", device: " << my_device << " bus id: ";
  std::cout << my_bus_id << std::endl;
#endif
}

  Cell const * host_cells_ptr;
  bool m_use_gpu_transporter;
  Cell * device_cells_ptr;
  size_t n_cells;
  Census_T census_photons;
  std::vector<Cell_Tally> cell_tallies;
  Cell_Tally *d_cell_tallies;
  std::vector<EmissionGroupData> emission_groups;
  EmissionGroupData *d_emission_groups;
};

#endif // gpu_setup_h_
//----------------------------------------------------------------------------//
// end of gpu_setup.h
//----------------------------------------------------------------------------//
