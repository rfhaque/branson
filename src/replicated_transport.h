#ifndef transport_replicated_h_
#define transport_replicated_h_

#include <algorithm>
#include <functional>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <vector>
#include <type_traits> // Required for std::is_same_v

#include "RNG.h"
#include "constants.h"
#include "gpu_setup.h"
#include "info.h"
#include "mesh.h"
#include "message_counter.h"
#include "post_process_functions.h"
#include "history_based_transport.h"
#include "event_based_transport.h"
#include "photon.h"
#include "photon_array.h" // Include PhotonArray

// Add function to calculate memory usage for both approaches
template<typename T>
void print_memory_footprint(const T& particle_container, const std::string& container_type) {
  size_t total_bytes = 0;
  size_t num_particles = particle_container.size();

  if (num_particles == 0) {
      std::cout << container_type << " Memory footprint: 0 particles, 0 bytes" << std::endl;
      return;
  }

  if constexpr (std::is_same_v<T, std::vector<Photon>>) {
    // AoS memory calculation
    total_bytes = num_particles * sizeof(Photon); // Simpler calculation using sizeof class
  }
  else if constexpr (std::is_same_v<T, PhotonArray>) {
    // SoA memory calculation
    total_bytes =
      num_particles * sizeof(uint32_t) +         // cell_ID vector
      num_particles * sizeof(uint32_t) +         // group vector
      num_particles * sizeof(uint32_t) +         // source_type vector
      num_particles * sizeof(unsigned char) +     // descriptors vector
      num_particles * sizeof(std::array<double, 3>) + // position vector
      num_particles * sizeof(std::array<double, 3>) + // angle vector
      num_particles * sizeof(double) +           // E vector
      num_particles * sizeof(double) +           // E0 vector
      num_particles * sizeof(double) +           // life_dx vector
      num_particles * sizeof(RNG);               // rng vector
  } else {
      std::cout << "Unknown particle container type for memory calculation." << std::endl;
      return;
  }


  double total_mb = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
  std::cout << container_type << " Memory footprint:" << std::endl;
  std::cout << "  Total particles: " << num_particles << std::endl;
  std::cout << "  Total bytes: " << total_bytes << std::endl;
  std::cout << "  Total MB: " << total_mb << std::endl;
  std::cout << "  Bytes per particle: " << static_cast<double>(total_bytes) / num_particles << std::endl;
}

template <typename Census_T>
Census_T replicated_transport(
  const Mesh& mesh, const GPU_Setup& gpu_setup, IMC_State& imc_state,
  std::vector<double>& rank_abs_E, std::vector<double>& rank_track_E,
  Census_T& all_photons, const int n_omp_threads, const uint32_t batch_size,
  const int transport_algorithm) {
  using std::cout;
  using std::endl;
  using std::vector;

  // Print total memory footprint information
  cout << "\n=== Total Particle Memory Usage ===" << endl;
  print_memory_footprint(all_photons, std::is_same_v<Census_T, std::vector<Photon>> ? "AoS (vector<Photon>)" : "SoA (PhotonArray)");

  // Print theoretical batch memory calculation once at the start (for CPU event-based)
  if (transport_algorithm == Constants::EVENT) {
      size_t batch_memory = 0;
      if constexpr (std::is_same_v<Census_T, std::vector<Photon>>) {
        batch_memory = batch_size * sizeof(Photon);
        cout << "\n=== AoS CPU Event Batch Memory Estimate ===" << endl;
      }
      else if constexpr (std::is_same_v<Census_T, PhotonArray>) {
        batch_memory =
          batch_size * sizeof(uint32_t) +          // cell_ID vector
          batch_size * sizeof(uint32_t) +          // group vector
          batch_size * sizeof(uint32_t) +          // source_type vector
          batch_size * sizeof(unsigned char) +      // descriptors vector
          batch_size * sizeof(std::array<double, 3>) + // position vector
          batch_size * sizeof(std::array<double, 3>) + // angle vector
          batch_size * sizeof(double) +            // E vector
          batch_size * sizeof(double) +            // E0 vector
          batch_size * sizeof(double) +            // life_dx vector
          batch_size * sizeof(RNG);                // rng vector
        cout << "\n=== SoA CPU Event Batch Memory Estimate ===" << endl;
      }
       if (batch_size > 0) {
            double batch_mb = static_cast<double>(batch_memory) / (1024.0 * 1024.0);
            cout << "Batch size: " << batch_size << " particles" << endl;
            cout << "Batch memory: " << batch_memory << " bytes (" << batch_mb << " MB)" << endl;
            cout << "Bytes per particle: " << static_cast<double>(batch_memory) / batch_size << endl;
       }
  }


  // is the GPU even available?
#ifdef USE_CUDA
  constexpr bool gpu_available = true;
#else
  constexpr bool gpu_available = false;
#endif

  double census_E = 0.0;
  double exit_E = 0.0;
  double next_dt = imc_state.get_next_dt(); //! Set for census photons
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // print warning message if GPU transport is requested but not available
  if (rank == 0 && gpu_setup.use_gpu_transporter() && !gpu_available) {
    std::cout << "WARNING: use_gpu_transporter set to true but GPU kernel not available,";
    std::cout << " running transport on CPU" << std::endl;
  }

  // timing
  Timer t_transport;
  t_transport.start_timer("timestep transport");

  //------------------------------------------------------------------------//
  // main transport loop
  //------------------------------------------------------------------------//

  Census_T census_list;   //! End of timestep census list
  vector<Cell_Tally> cell_tallies(mesh.get_n_local_cells()); // Initialize tallies (zeroed)
  uint32_t rank_cell_offset{ 0 }; // no offset in replicated mesh

  // Precompute emission group data for event-based transport (both CPU and GPU)
  std::vector<EmissionGroupData> emission_groups;
  if (transport_algorithm == Constants::EVENT) {
      emission_groups.resize(mesh.get_n_local_cells());
      for (size_t i = 0; i < mesh.get_n_local_cells(); ++i) {
          emission_groups[i] = precompute_emission_group_data(mesh.get_cells()[i]);
      }
  }


  // GPU mode
  if (gpu_setup.use_gpu_transporter() && gpu_available) {
    if (transport_algorithm == Constants::HISTORY) {
        if constexpr (std::is_same_v<Census_T, std::vector<Photon>>) {
            t_transport.start_timer("gpu history transport");
            gpu_transport_photons(rank_cell_offset, all_photons, gpu_setup.get_device_cells_ptr(), cell_tallies);
            t_transport.stop_timer("gpu history transport");
            cout << "GPU history transport time (AoS): " << t_transport.get_time("gpu history transport") << endl;
            // Post-process needed after GPU transport to handle census/exit
            post_process_photons(next_dt, all_photons, census_list, census_E, exit_E);
        } else {
            t_transport.start_timer("gpu history transport");
            gpu_transport_photons(rank_cell_offset, all_photons, gpu_setup.get_device_cells_ptr(), cell_tallies);
            t_transport.stop_timer("gpu history transport");
            cout << "GPU history transport time (SoA): " << t_transport.get_time("gpu history transport") << endl;
            // Post-process needed after GPU transport to handle census/exit
            post_process_photons(next_dt, all_photons, census_list, census_E, exit_E);
            // cout << "No GPU history kernel for SoA particle data structures yet" << endl;
            // exit(EXIT_FAILURE);
        }
    } else if (transport_algorithm == Constants::EVENT) {
        t_transport.start_timer("gpu event transport");
        // Call the correct overloaded gpu_event_transport_photons based on Census_T
        gpu_event_transport_photons(rank_cell_offset, all_photons, gpu_setup.get_device_cells_ptr(), cell_tallies, emission_groups);
        t_transport.stop_timer("gpu event transport");
        if constexpr (std::is_same_v<Census_T, std::vector<Photon>>) {
             cout << "GPU event transport time (AoS): " << t_transport.get_time("gpu event transport") << endl;
        } else {
             cout << "GPU event transport time (SoA): " << t_transport.get_time("gpu event transport") << endl;
        }
        // Post-process needed after GPU transport to handle census/exit
        post_process_photons(next_dt, all_photons, census_list, census_E, exit_E);
    } else {
         cout << "Transport algorithm type not supported for GPU... exiting" << endl;
         exit(EXIT_FAILURE);
    }
  }
  // CPU mode
  else {
    if (transport_algorithm == Constants::HISTORY) {
      cout << "Starting CPU history-based transport..." << endl;
      t_transport.start_timer("cpu history transport");
      // Call correct overloaded history_cpu_transport_photons
      history_cpu_transport_photons(rank_cell_offset, all_photons, mesh.get_cells(), cell_tallies, n_omp_threads);
      t_transport.stop_timer("cpu history transport");
      cout << "CPU history transport time: " << t_transport.get_time("cpu history transport") << endl;
      // Post-process needed after CPU history transport
      post_process_photons(next_dt, all_photons, census_list, census_E, exit_E);
    }
    else if (transport_algorithm == Constants::EVENT) {
      cout << "Starting CPU event-based transport (batch size: " << batch_size << ")..." << endl;
      t_transport.start_timer("cpu event transport");
      if constexpr (std::is_same_v<Census_T, std::vector<Photon>>) {
        // AoS CPU Event-Based (Batched)
        for (size_t batch_start = 0; batch_start < all_photons.size(); batch_start += batch_size) {
          size_t batch_end = std::min(batch_start + batch_size, all_photons.size());
          // Create a sub-vector for the batch (requires copy)
          std::vector<Photon> batch_photons(all_photons.begin() + batch_start, all_photons.begin() + batch_end);

          cpu_event_transport_photons(rank_cell_offset, batch_photons, mesh.get_cells(), cell_tallies, n_omp_threads, emission_groups);

          // Post process photons for this batch, account for escaped energy and add particles to census
          post_process_photons(next_dt, batch_photons, census_list, census_E, exit_E);
          // Note: Modifications in batch_photons are not reflected back into all_photons here.
          // The original CPU event code modified the main array directly.
          // For consistency with GPU/History post-processing, we process batches and collect census/exit.
        }
      }
      else if constexpr (std::is_same_v<Census_T, PhotonArray>) {
        // SoA CPU Event-Based (Batched)
        for (size_t batch_start = 0; batch_start < all_photons.size(); batch_start += batch_size) {
          size_t batch_end = std::min(batch_start + batch_size, all_photons.size());
          // Get a sub-batch (creates a copy)
          auto batch_photons = all_photons.get_sub_batch(batch_start, batch_end);

          cpu_event_transport_photons(rank_cell_offset, batch_photons, mesh.get_cells(), cell_tallies, n_omp_threads, emission_groups);

          // Post process photons for this batch
          post_process_photons(next_dt, batch_photons, census_list, census_E, exit_E);
        }
      } else {
          cout << "Unsupported particle container for CPU event transport." << endl;
          exit(EXIT_FAILURE);
      }
       t_transport.stop_timer("cpu event transport");
       cout << "CPU event transport time: " << t_transport.get_time("cpu event transport") << endl;
       // Note: Post-processing happens inside the batch loop for CPU event-based.
    }
    else {
      cout << "Transport algorithm type not supported... exiting" << endl;
      exit(EXIT_FAILURE);
    }
  }

  // copy cell tallies back out to rank_abs_E and rank_track_E
  // This should happen regardless of CPU/GPU or algorithm, using the final cell_tallies state.
  for (size_t i = 0; i < cell_tallies.size();++i) {
    rank_abs_E[i] = cell_tallies[i].get_abs_E();
    rank_track_E[i] = cell_tallies[i].get_track_E();
  }

  // record time of transport work for this rank
  t_transport.stop_timer("timestep transport");

  // wait for all ranks to finish
  MPI_Barrier(MPI_COMM_WORLD);

  // Sorting census list might be needed depending on subsequent operations
  // std::sort(census_list.begin(), census_list.end()); // If using std::vector<Photon>
  // Sorting PhotonArray census_list would require a custom sort or sorting based on one member vector.

  // set diagnostic quantities
  imc_state.set_exit_E(exit_E);
  imc_state.set_post_census_E(census_E);
  imc_state.set_census_size(census_list.size());
  imc_state.set_rank_transport_runtime(
    t_transport.get_time("timestep transport"));

  return census_list;
}

#endif // def transport_replicated_h_