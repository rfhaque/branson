#ifndef transport_mode_wrapper_h_
#define transport_mode_wrapper_h_

#include "cell_tally.h"
#include "gpu_setup.h"
#include "mesh.h"
#include "photon.h"
#include "photon_array.h"
#include "post_process_functions.h"
#include "timer.h"

//! Use one of 8 avaialbe transport algorithms in DD or REP: CPU/GPU, EVENT/HISTORY, SoA/AoS
template <typename Census_T>
std::tuple<uint64_t, double, double>
batch_transport(const double next_dt, const bool gpu_available, const GPU_Setup &gpu_setup,
                const IMC_Parameters &imc_parameters,
                const uint32_t rank_cell_offset, const Mesh &mesh,
                Census_T &all_photons, Census_T &census_list,
                std::vector<std::vector<Photon>> &phtn_send_buffer,
                std::vector<Cell_Tally> &cell_tallies, Timer &t_transport) {
  using std::cout;
  using std::endl;
  auto transport_algorithm = imc_parameters.get_transport_algorithm();
  auto n_omp_threads = imc_parameters.get_n_omp_threads();
  uint64_t n_complete = 0.0;
  double census_E{0.0};
  double exit_E{0.0};
  std::string hardware = "GPU";      // default, change to CPU if used
  std::string algorithm = "history"; // default, change to event if used
  t_transport.start_timer("batch transport");
  if (transport_algorithm == Constants::HISTORY) {
    // HISTORY: GPU
    if (gpu_setup.use_gpu_transporter() && gpu_available) {
      if constexpr (std::is_same_v<Census_T, std::vector<Photon>>) {
        gpu_transport_photons(rank_cell_offset, all_photons, gpu_setup.get_device_cells_ptr(),
                              cell_tallies);
      } else {
        gpu_transport_photons(rank_cell_offset, all_photons, gpu_setup.get_device_cells_ptr(),
                              cell_tallies);
      }
    } // HISTORY: GPU
    // HISTORY: CPU
    else {
      hardware = "CPU";
      // Call correct overloaded history_cpu_transport_photons
      history_cpu_transport_photons(rank_cell_offset, all_photons, mesh.get_cells(), cell_tallies,
                                    n_omp_threads);
    } // HISTORY: CPU
    auto [batch_complete, batch_exit_E, batch_census_E] =
        post_process_photons(next_dt, all_photons, census_list, mesh, phtn_send_buffer);
    n_complete += batch_complete;
    exit_E += batch_exit_E;
    census_E += batch_census_E;
  } // HISTORY
  else if (transport_algorithm == Constants::EVENT) {
    algorithm = "event";
    // Precompute emission group data for event-based transport (both CPU and GPU)
    std::vector<EmissionGroupData> emission_groups;
    if (transport_algorithm == Constants::EVENT) {
      emission_groups.resize(mesh.get_n_local_cells());
      for (size_t i = 0; i < mesh.get_n_local_cells(); ++i) {
        emission_groups[i] = precompute_emission_group_data(mesh.get_cells()[i]);
      }
    }
    // EVENT: GPU
    if (gpu_setup.use_gpu_transporter() && gpu_available) {
      // Call the correct overloaded gpu_event_transport_photons based on Census_T
      gpu_event_transport_photons(rank_cell_offset, all_photons, gpu_setup.get_device_cells_ptr(),
                                  cell_tallies, emission_groups);
      auto [batch_complete, batch_exit_E, batch_census_E] =
          post_process_photons(next_dt, all_photons, census_list, mesh, phtn_send_buffer);

      n_complete += batch_complete;
      exit_E += batch_exit_E;
      census_E += batch_census_E;
    } // GPU
    // EVENT: CPU
    else {
      hardware = "CPU";
      // Call correct overloaded history_cpu_transport_photons
      // Note that both SoA and AoS can use batches here
      auto event_batch_size = imc_parameters.get_batch_size();
      cout << "Starting CPU event-based transport (batch size: " << event_batch_size << ")..."
           << endl;
      for (size_t batch_start = 0; batch_start < all_photons.size();
           batch_start += event_batch_size) {
        size_t batch_end = std::min(batch_start + event_batch_size, all_photons.size());
        if constexpr (std::is_same_v<Census_T, std::vector<Photon>>) {
          // Create a sub-vector for the batch (requires copy)
          std::vector<Photon> batch_photons(all_photons.begin() + batch_start,
                                            all_photons.begin() + batch_end);
          cpu_event_transport_photons(rank_cell_offset, batch_photons, mesh.get_cells(),
                                      cell_tallies, n_omp_threads, emission_groups);
          auto [batch_complete, batch_exit_E, batch_census_E] =
              post_process_photons(next_dt, batch_photons, census_list, mesh, phtn_send_buffer);
          n_complete += batch_complete;
          exit_E += batch_exit_E;
          census_E += batch_census_E;
        } else if constexpr (std::is_same_v<Census_T, PhotonArray>) {
          // Get a sub-batch (creates a copy)
          auto batch_photons = all_photons.get_sub_batch(batch_start, batch_end);
          cpu_event_transport_photons(rank_cell_offset, batch_photons, mesh.get_cells(),
                                      cell_tallies, n_omp_threads, emission_groups);
          auto [batch_complete, batch_exit_E, batch_census_E] =
              post_process_photons(next_dt, batch_photons, census_list, mesh, phtn_send_buffer);
          n_complete += batch_complete;
          exit_E += batch_exit_E;
          census_E += batch_census_E;
        } else {
          cout << "Unsupported particle container for CPU event transport." << endl;
          exit(EXIT_FAILURE);
        }
      }
    } // EVENT: CPU
  }   // EVENT
  t_transport.stop_timer("batch transport");
  if constexpr (std::is_same_v<Census_T, std::vector<Photon>>) {
    cout << hardware << ", " << algorithm << ", AoS, transport--particles: " << n_complete
         << " time: " << t_transport.get_time("batch transport") << endl;
  } else {
    cout << hardware << ", " << algorithm << ", SoA, transport--particles: " << n_complete
         << " time: " << t_transport.get_time("batch transport") << endl;
  }
  return {n_complete, exit_E, census_E};
}
#endif // def transport_mode_wrapper_h_
