#ifndef transport_mode_wrapper_h_
#define transport_mode_wrapper_h_

#include "cell_tally.h"
#include "gpu_setup.h"
#include "mesh.h"
#include "photon.h"
#include "photon_array.h"
#include "photon_data.h"
#include "post_process_functions.h"
#include "timer.h"
#include "config.h"

//! Use one of 8 avaialbe transport algorithms in DD or REP: CPU/GPU, EVENT/HISTORY, SoA/AoS
template <typename Census_T>
std::tuple<uint64_t, double, double>
batch_transport(const double next_dt, const bool gpu_available, GPU_Setup<Census_T> &gpu_setup,
                const IMC_Parameters &imc_parameters,
                const uint32_t rank_cell_offset, const Mesh &mesh,
                Photon_Data<Census_T> &photon_data,
                size_t batch_start, size_t batch_end,
                std::vector<std::vector<Photon>> &phtn_send_buffer,
                Timer &t_transport) {
  auto transport_algorithm = imc_parameters.get_transport_algorithm();
  auto n_omp_threads = imc_parameters.get_n_omp_threads();
  uint64_t n_complete = 0;
  double census_E{0.0};
  double exit_E{0.0};
  std::string hardware = "GPU";      // default, change to CPU if used
  std::string algorithm = "history"; // default, change to event if used
  t_transport.start_timer("batch transport");
  if (transport_algorithm == Constants::HISTORY) {
    // HISTORY: GPU
    if (gpu_setup.use_gpu_transporter() && gpu_available) {
#ifdef USE_GPU
        gpu_transport_photons(rank_cell_offset, photon_data, batch_start, batch_end,  gpu_setup);
        photon_data.sync();
#endif
    } // HISTORY: GPU
    // HISTORY: CPU
    else {
      hardware = "CPU";
      // Call correct overloaded history_cpu_transport_photons
      history_cpu_transport_photons(rank_cell_offset, photon_data.photons, batch_start, batch_end, gpu_setup, n_omp_threads);
    } // HISTORY: CPU
    auto [batch_complete, batch_exit_E, batch_census_E] =
        post_process_photons(next_dt, photon_data.photons, batch_start, batch_end, mesh, phtn_send_buffer);
    n_complete += batch_complete;
    exit_E += batch_exit_E;
    census_E += batch_census_E;
  } // HISTORY
  else if (transport_algorithm == Constants::EVENT) {
    algorithm = "event";
    // EVENT: GPU
    if (gpu_setup.use_gpu_transporter() && gpu_available) {
#ifdef USE_GPU
      // Call the correct overloaded gpu_event_transport_photons based on Census_T
      gpu_event_transport_photons(rank_cell_offset, photon_data, batch_start, batch_end, gpu_setup);
      photon_data.sync();
      auto [batch_complete, batch_exit_E, batch_census_E] =
          post_process_photons(next_dt, photon_data.photons, batch_start, batch_end, mesh, phtn_send_buffer);

      n_complete += batch_complete;
      exit_E += batch_exit_E;
      census_E += batch_census_E;
#endif
    } // GPU
    // EVENT: CPU
    else {
      hardware = "CPU";
      // Call correct overloaded history_cpu_transport_photons
      // Note that both SoA and AoS can use batches here
      auto event_batch_size = imc_parameters.get_event_batch_size();
      //std::cout << "Starting CPU event-based transport (batch size: " << event_batch_size << ", batch_end: "<<batch_end<<")..."
      //     << std::endl;
      for (size_t event_batch_start = batch_start; event_batch_start < batch_end;
           event_batch_start += event_batch_size) {
        size_t event_batch_end = std::min(event_batch_start + event_batch_size, batch_end);
        //std::cout << "batch: event_batch_start: " << event_batch_start << " event_batch_end: "<<event_batch_end<<std::endl;
        cpu_event_transport_photons(rank_cell_offset, photon_data.photons, event_batch_start, event_batch_end, gpu_setup,
                                    n_omp_threads);
      }
      //std::cout<<"batches done, postprocessing..."<<std::endl;
      auto [batch_complete, batch_exit_E, batch_census_E] =
          post_process_photons(next_dt, photon_data.photons, batch_start, batch_end, mesh, phtn_send_buffer);
      n_complete += batch_complete;
      exit_E += batch_exit_E;
      census_E += batch_census_E;
    } // EVENT: CPU
  }   // EVENT
  t_transport.stop_timer("batch transport");
  /*
  if constexpr (std::is_same_v<Census_T, std::vector<Photon>>) {
    std::cout << hardware << ", " << algorithm << ", AoS, transport--particles: " << n_complete
         << " time: " << t_transport.get_time("batch transport") << std::endl;
  } else {
    std::cout << hardware << ", " << algorithm << ", SoA, transport--particles: " << n_complete
         << " time: " << t_transport.get_time("batch transport") << std::endl;
  }
  */
  return {n_complete, exit_E, census_E};
}
#endif // def transport_mode_wrapper_h_
