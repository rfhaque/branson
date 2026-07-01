//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   event_based_transport.h
 * \author Joseph Farmer, Alex Long
 * \date   August 1 2024
 * \brief  IMC transport with an event-based algorithm (CPU and GPU)
 * \note   Copyright (C) 2024 Triad National Security, LLC.
 *         All rights reserved
 */
 //---------------------------------------------------------------------------//
#ifndef event_based_transport_h_
#define event_based_transport_h_

#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <fstream>
#include <string>
#include <array>

#include "config.h"
#include "RNG.h"
#include "cell_tally.h"
#include "constants.h"
#include "photon.h"
#include "photon_array.h"
#include "sampling_functions.h"

#ifdef USE_GPU
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#endif

//----------------------------------------------------------------------------//
// GPU Specific Includes and Typedefs                                         //
//----------------------------------------------------------------------------//
#ifdef USE_GPU

enum GPUEventType { GPU_SCATTER, GPU_BOUNDARY, GPU_CENSUS, GPU_KILLED, GPU_PASS, GPU_EXIT, GPU_BOUND }; // Match Constants if possible

#endif // USE_GPU

//----------------------------------------------------------------------------//
// CPU Event-Based Transport Structures and Functions                         //
//----------------------------------------------------------------------------//

enum EventType { SCATTER, BOUNDARY, CENSUS, KILLED }; // CPU Event Type

struct Event {
  size_t photon_index;
  EventType type;
  double distance;
};

// PhotonTrackingData is specific to the  CPU event-based approach
struct PhotonTrackingData {
  double initial_angle_x;
  double final_angle_x;
  double time_in_cell;
  size_t num_interactions;
  bool entered_domain;
  bool exited_vacuum;
};

//----------------------------------------------------------------------------//
// CPU Event-Based Transport - SoA (PhotonArray)                              //
//----------------------------------------------------------------------------//
inline void precompute_data(const uint32_t rank_cell_offset,
  const uint32_t* group,
  const uint32_t* cell_ID,
  const Cell* cells,
  const size_t* active_photons,
  size_t active_count,
  std::vector<double>& sigma_s,
  std::vector<double>& sigma_a,
  std::vector<double>& f,
  std::vector<double>& total_sigma_s,
  std::vector<uint32_t>& local_cell_indices) {
#pragma omp simd
  for (size_t i = 0; i < active_count; ++i) {
    size_t photon_index = active_photons[i];
    local_cell_indices[i] = cell_ID[photon_index] - rank_cell_offset;
    const Cell* cell = &cells[local_cell_indices[i]];
    sigma_s[i] = cell->get_op_s(group[photon_index]);
    sigma_a[i] = cell->get_op_a(group[photon_index]);
    f[i] = cell->get_f();
    total_sigma_s[i] = (1.0 - f[i]) * sigma_a[i] + sigma_s[i];
  }
}

inline void calculate_distances(
  std::array<double,3> *pos,
  std::array<double,3> *angle,
  double *life_dx,
  RNG *rng,
  const Cell* cells,
  const size_t* active_photons,
  size_t active_count,
  const std::vector<double>& total_sigma_s,
  const std::vector<uint32_t>& local_cell_indices,
  std::vector<Event>& events) {
#pragma omp simd
  for (size_t i = 0; i < active_count; ++i) {
    size_t photon_index = active_photons[i];
    const Cell* cell = &cells[local_cell_indices[i]];

    double dist_to_scatter = 1.0e100;
    if (total_sigma_s[i] > 0.0) {
      double rn = rng[photon_index].generate_random_number();
      dist_to_scatter = -std::log(rn) / total_sigma_s[i];
    }

    uint32_t surface_cross = 0;
    double dist_to_boundary = cell->get_distance_to_boundary(
      pos[photon_index],
      angle[photon_index],
      surface_cross);

    double dist_to_census = life_dx[photon_index];
    double final_dist = std::min(dist_to_scatter, std::min(dist_to_boundary, dist_to_census));

    events[i].distance = final_dist;
    events[i].photon_index = photon_index;

    if (final_dist == dist_to_scatter)       events[i].type = SCATTER;
    else if (final_dist == dist_to_boundary) events[i].type = BOUNDARY;
    else                                     events[i].type = CENSUS;
  }
}

inline void process_scatter_events(
  uint32_t *group,
  unsigned char *descriptors,
  std::array<double,3> *angle,
  RNG *rng,
  const std::vector<double>& sigma_s,
  const std::vector<double>& total_sigma_s,
  const std::vector<uint32_t>& local_cell_indices,
  const Cell* cells,
  const size_t* scatter_photon_indices,
  size_t scatter_count,
  const std::vector<EmissionGroupData>& emission_groups,
  std::vector<PhotonTrackingData>& tracking_data) // Tracking data only for CPU
{
#pragma omp simd
  for (size_t i = 0; i < scatter_count; ++i) {
    size_t photon_index = scatter_photon_indices[i];
    uint32_t local_idx = local_cell_indices[i];
    tracking_data[photon_index].num_interactions += 1;
    angle[photon_index] = get_uniform_angle(rng[i]);
    descriptors[photon_index] = static_cast<unsigned char>(Constants::SCATTER);
    if (rng[i].generate_random_number() > (sigma_s[i] / total_sigma_s[i])) {
      group[photon_index] =
        sample_emission_group(rng[i], emission_groups[local_idx]);
      // chance of more intensive scatter
      if (rng[i].generate_random_number() <= Constants::intensive_scatter_fraction) {
        // get a frequency (faux multigroup so just sample from wide spectrum)
        double freq = Constants::lower_frequency_bound + static_cast<double>(group[i])/static_cast<double>(BRANSON_N_GROUPS)*Constants::delta_frequency_bounds;
        auto new_energy_angle = intensive_scatter(cells[local_idx].get_T_e(), freq, angle[photon_index], rng[i]);
        angle[photon_index] = new_energy_angle.second;
      }
    }
  }
}

inline void process_boundary_events(
  uint32_t* cell_ID,
  unsigned char *descriptors,
  std::array<double,3> *pos,
  std::array<double,3> *angle,
  const size_t* boundary_photon_indices,
  const Cell* cells,
  uint32_t rank_cell_offset,
  size_t boundary_count,
  std::vector<PhotonTrackingData>& tracking_data) { // Tracking data only for CPU
#pragma omp simd
  for (size_t i = 0; i < boundary_count; ++i) {
    size_t photon_index = boundary_photon_indices[i];
    uint32_t local_cell_idx = cell_ID[photon_index] - rank_cell_offset;
    const Cell* cell = &cells[local_cell_idx];

    uint32_t surface_cross = 0;
    cell->get_distance_to_boundary(pos[photon_index],
      angle[photon_index],
      surface_cross);
    auto boundary_type = cell->get_bc(surface_cross);

    if (boundary_type == Constants::ELEMENT) {
      cell_ID[photon_index] = cell->get_next_cell(surface_cross);
      descriptors[photon_index] = static_cast<unsigned char>(Constants::BOUND); // Still active
    }
    else if (boundary_type == Constants::PROCESSOR) {
      cell_ID[photon_index] = cell->get_next_cell(surface_cross);
      descriptors[photon_index] = static_cast<unsigned char>(Constants::PASS); // Inactive
    }
    else if (boundary_type == Constants::VACUUM ||
      boundary_type == Constants::SOURCE) {
      descriptors[photon_index] = static_cast<unsigned char>(Constants::EXIT); // Inactive
      if (boundary_type == Constants::VACUUM) {
        tracking_data[photon_index].exited_vacuum = true;
      }
    }
    else { // REFLECT
      int reflect_dim = surface_cross / 2;
      angle[photon_index][reflect_dim] =
        -angle[photon_index][reflect_dim];
      descriptors[photon_index] = static_cast<unsigned char>(Constants::BOUND); // Still active
    }
  }
}

inline void process_census_events(
  unsigned char *descriptors,
  const size_t* census_photon_indices,
  size_t census_count) {
#pragma omp simd
  for (size_t i = 0; i < census_count; ++i) {
    size_t photon_index = census_photon_indices[i];
    descriptors[photon_index] = static_cast<unsigned char>(Constants::CENSUS); // Inactive
  }
}

inline void update_photon_state(
  unsigned char *descriptors,
  std::array<double,3> *pos,
  std::array<double,3> *angle,
  double *E,
  double *E0,
  double *life_dx,
  Cell_Tally* cell_tallies,
  const std::vector<Event>& events, // Contains distance and type for active photons
  const std::vector<double>& sigma_a, // Corresponds to active photons
  const std::vector<double>& f,       // Corresponds to active photons
  const std::vector<uint32_t>& local_cell_indices, // Corresponds to active photons
  size_t active_count,
  std::vector<size_t>& scatter_indices, // Output: indices of photons that will scatter
  std::vector<size_t>& boundary_indices,// Output: indices of photons that will hit boundary
  std::vector<size_t>& census_indices,  // Output: indices of photons that will hit census
  std::vector<size_t>& killed_indices,  // Output: indices of photons that were killed
  size_t& scatter_count,                // Output count
  size_t& boundary_count,               // Output count
  size_t& census_count,                 // Output count
  size_t& killed_count,                 // Output count
  std::vector<PhotonTrackingData>& tracking_data) // Tracking data only for CPU
{
  // Reset counts
  scatter_count = 0;
  boundary_count = 0;
  census_count = 0;
  killed_count = 0;

  std::vector<double> absorbed_Es(active_count);
#pragma omp simd
  for (size_t i = 0; i < active_count; ++i) {
    double distance = events[i].distance;
    size_t photon_idx = events[i].photon_index; // Original index
    absorbed_Es[i] = E[photon_idx] *
      (1.0 - std::exp(-sigma_a[i] * f[i] * distance));
  }

  // Accumulate tallies - cannot SIMD easily
  for (size_t i = 0; i < active_count; ++i) {
    uint32_t local_idx = local_cell_indices[i];
    double track_contrib = (sigma_a[i] > 0.0 && f[i] > 0.0) ? absorbed_Es[i] / (sigma_a[i] * f[i]) : 0.0;
    cell_tallies[local_idx].accumulate_absorbed_E(absorbed_Es[i]);
    cell_tallies[local_idx].accumulate_track_E(track_contrib);
  }

#pragma omp simd
  for (size_t i = 0; i < active_count; ++i) {
    const Event& event = events[i];
    size_t photon_index = event.photon_index; // Original index
    double distance = event.distance;

    // Update energy and position
    E[photon_index] -= absorbed_Es[i];
    pos[photon_index][0] += angle[photon_index][0] * distance;
    pos[photon_index][1] += angle[photon_index][1] * distance;
    pos[photon_index][2] += angle[photon_index][2] * distance;
    life_dx[photon_index] -= distance;
    tracking_data[photon_index].time_in_cell += distance / Constants::c;

    // Check for kill based on energy cutoff
    if (E[photon_index] / E0[photon_index] < Constants::cutoff_fraction) {

      cell_tallies[local_cell_indices[i]].accumulate_absorbed_E(E[photon_index]);
      descriptors[photon_index] = static_cast<unsigned char>(Constants::KILLED);
      // This photon will be added to killed_indices below
    }

  }

   // Separate loop to fill event index vectors - cannot SIMD easily due to conditional increments
   // and potential race conditions on counts
  for (size_t i = 0; i < active_count; ++i) {
      const Event& event = events[i];
      size_t photon_index = event.photon_index; // Original index

      // Check descriptor first (set above if killed by energy cutoff)
      if (descriptors[photon_index] == static_cast<unsigned char>(Constants::KILLED)) {
          killed_indices[killed_count++] = photon_index;
      } else {
          // Assign to event lists based on the calculated event type
          switch (event.type) {
          case SCATTER:
              scatter_indices[scatter_count++] = photon_index;
              break;
          case BOUNDARY:
              boundary_indices[boundary_count++] = photon_index;
              break;
          case CENSUS:
              census_indices[census_count++] = photon_index;
              break;
          default: // Should not happen
              break;
          }
      }
  }
}

// CPU Main loop for SoA
void cpu_event_transport_photons(const uint32_t rank_cell_offset,
  Photon_Data<PhotonArray> photon_data, size_t batch_start, size_t batch_end, GPU_Setup<PhotonArray> &gpu_setup,
  int /*n_omp_threads*/ ) // n_omp_threads currently unused in this fine-grained version
{
  // ARL: NOTE, need to use batch start and end in CPU event-based loops
  const size_t maxPhotons = batch_end - batch_start;
  if (maxPhotons == 0) return;

  // Tracking data is specific to this CPU implementation
  std::vector<PhotonTrackingData> tracking_data(photon_data.n_photons);
  for (size_t i = batch_start; i < batch_end; ++i) {
    tracking_data[i].initial_angle_x = photon_data.h_angle_ptr[i][0];
    tracking_data[i].final_angle_x = photon_data.h_angle_ptr[i][0];
    tracking_data[i].time_in_cell = 0.0;
    tracking_data[i].num_interactions = 0;
    tracking_data[i].entered_domain = (photon_data.h_angle_ptr[i][0] > 0);
    tracking_data[i].exited_vacuum = false;
  }

  // Allocate vectors for event processing
  std::vector<size_t> scatter_indices(maxPhotons),
    boundary_indices(maxPhotons),
    census_indices(maxPhotons),
    killed_indices(maxPhotons),
    active_photons_indices(maxPhotons); // Stores original indices of active photons

  std::vector<Event> events(maxPhotons); // Stores event info for active photons
  std::vector<uint32_t> local_cell_indices(maxPhotons); // Stores local cell index for active photons
  std::vector<double> sigma_s(maxPhotons), sigma_a(maxPhotons), f(maxPhotons),
    total_sigma_s(maxPhotons);

  // Initialize active photons list with original indices 0 to N-1
  std::iota(active_photons_indices.begin(), active_photons_indices.end(), batch_start);
  size_t active_count = maxPhotons;

  const Cell* cells_ptr = gpu_setup.get_host_cells_ptr(); // CPU pointer here
  Cell_Tally* tallies_ptr = gpu_setup.get_device_cell_tallies_ptr(); // CPU pointer here

  while (active_count > 0) {
    // Resize intermediate vectors to current active count for efficiency
    events.resize(active_count);
    local_cell_indices.resize(active_count);
    sigma_s.resize(active_count);
    sigma_a.resize(active_count);
    f.resize(active_count);
    total_sigma_s.resize(active_count);

    size_t scatter_count = 0;
    size_t boundary_count = 0;
    size_t census_count = 0;
    size_t killed_count = 0;

    // 1. Precompute physics data for active photons
    //    active_photons_indices.data() points to the original indices
    precompute_data(rank_cell_offset, photon_data.h_cell_ID_ptr, photon_data.h_group_ptr, cells_ptr,
      active_photons_indices.data(), active_count,
      sigma_s, sigma_a, f, total_sigma_s, local_cell_indices);

    // 2. Calculate distances and event types for active photons
    calculate_distances(photon_data.h_pos_ptr, photon_data.h_angle_ptr, photon_data.h_life_dx_ptr, photon_data.h_RNG_ptr, cells_ptr,
      active_photons_indices.data(), active_count,
      total_sigma_s, local_cell_indices, events);
      // events[i] now corresponds to the photon active_photons_indices[i]

    // 3. Update photon state, tally energy, check for kills, and classify into event lists
    //    Resize event index vectors before passing them
    scatter_indices.resize(active_count);
    boundary_indices.resize(active_count);
    census_indices.resize(active_count);
    killed_indices.resize(active_count);

    update_photon_state(
      photon_data.h_descriptors_ptr,
      photon_data.h_pos_ptr,
      photon_data.h_angle_ptr,
      photon_data.h_E_ptr,
      photon_data.h_E0_ptr,
      photon_data.h_life_dx_ptr,
      tallies_ptr, events,
      sigma_a, f, local_cell_indices,
      active_count,
      scatter_indices, boundary_indices, census_indices, killed_indices, // Output lists
      scatter_count, boundary_count, census_count, killed_count, // Output counts
      tracking_data);

    // 4. Process events using the filled index vectors and counts

    if (scatter_count > 0) {
        std::vector<double> scatter_sigma_s(scatter_count);
        std::vector<double> scatter_total_sigma_s(scatter_count);
        std::vector<uint32_t> scatter_local_indices(scatter_count);
        std::vector<double> subset_sigma_s(scatter_count);
        std::vector<double> subset_total_sigma_s(scatter_count);
        std::vector<uint32_t> subset_local_indices(scatter_count);
        // Find the mapping
        std::vector<size_t> active_to_subset_map(photon_data.n_photons, photon_data.n_photons);// Map original index to position in active list
        for(size_t i=0; i<active_count; ++i) active_to_subset_map[active_photons_indices[i]] = i;

        for(size_t i=0; i<scatter_count; ++i) {
            size_t original_index = scatter_indices[i];
            size_t active_list_pos = active_to_subset_map[original_index];
            if (active_list_pos < active_count) { // Should always be true
                 subset_sigma_s[i] = sigma_s[active_list_pos];
                 subset_total_sigma_s[i] = total_sigma_s[active_list_pos];
                 subset_local_indices[i] = local_cell_indices[active_list_pos];
            } else { /* handle error */ }
        }

       process_scatter_events(
        photon_data.h_group_ptr,
        photon_data.h_descriptors_ptr,
        photon_data.h_angle_ptr,
        photon_data.h_RNG_ptr,
        subset_sigma_s, subset_total_sigma_s,
        subset_local_indices, cells_ptr,
        scatter_indices.data(), scatter_count, // Pass original indices
        gpu_setup.get_emission_groups(), tracking_data);
    }
    if (boundary_count > 0) {
      process_boundary_events(
        photon_data.h_cell_ID_ptr,
        photon_data.h_descriptors_ptr,
        photon_data.h_pos_ptr,
        photon_data.h_angle_ptr,
      boundary_indices.data(),
        cells_ptr, rank_cell_offset, boundary_count, tracking_data);
    }
    if (census_count > 0) {
      process_census_events(photon_data.h_descriptors_ptr, census_indices.data(), census_count);
    }

    // 5. Filter active photons for the next iteration
    // Create the next list of active photon indices
    std::vector<size_t> next_active_photons_indices;
    next_active_photons_indices.reserve(active_count); // Reserve space

    for (size_t i = 0; i < active_count; ++i) {
        size_t index = active_photons_indices[i]; // Get original index
        unsigned char desc = photon_data.h_descriptors_ptr[index];
        // Keep if descriptor indicates it's still active in this domain
        if (desc == static_cast<unsigned char>(Constants::BOUND) ||
            desc == static_cast<unsigned char>(Constants::SCATTER))
        {
            next_active_photons_indices.push_back(index);
        } else {
             // Update final angle on termination if needed by tracking
             tracking_data[index].final_angle_x = photon_data.h_angle_ptr[index][0];
        }
    }
    active_photons_indices = std::move(next_active_photons_indices); // Replace old list
    active_count = active_photons_indices.size(); // Update active count

  } // End while(active_count > 0)
}


//----------------------------------------------------------------------------//
// CPU Event-Based Transport - AoS (std::vector<Photon>)                      //
//----------------------------------------------------------------------------//

// Precompute data for AOS (Array of Structures)
inline void precompute_data(const uint32_t rank_cell_offset, Photon *photon_array, const Cell* cells, const size_t* active_photons, size_t active_count, std::vector<double>& sigma_s, std::vector<double>& sigma_a, std::vector<double>& f, std::vector<double>& total_sigma_s, std::vector<uint32_t>& local_cell_indices) {
#pragma omp simd
  for (size_t i = 0; i < active_count; ++i) {
    size_t photon_index = active_photons[i];
    const auto& phtn = photon_array[photon_index];
    local_cell_indices[i] = phtn.get_cell() - rank_cell_offset;
    const Cell* cell = &cells[local_cell_indices[i]];
    sigma_s[i] = cell->get_op_s(phtn.get_group());
    sigma_a[i] = cell->get_op_a(phtn.get_group());
    f[i] = cell->get_f();
    total_sigma_s[i] = (1.0 - f[i]) * sigma_a[i] + sigma_s[i];
  }
}

// Calculate distances for AOS
inline void calculate_distances(Photon *photon_array, const Cell* cells, const size_t* active_photons, size_t active_count, const std::vector<double>& total_sigma_s, const std::vector<uint32_t>& local_cell_indices, std::vector<Event>& events) {
#pragma omp simd
  for (size_t i = 0; i < active_count; ++i) {
    size_t photon_index = active_photons[i];
    auto& phtn = photon_array[photon_index];
    Cell const* cell = &cells[local_cell_indices[i]];
    auto& rng = phtn.get_rng();
    uint32_t surface_cross = 0;
    const double dist_to_scatter = (total_sigma_s[i] > 0.0) ? -log(rng.generate_random_number()) / total_sigma_s[i] : 1e100;
    const double dist_to_boundary = cell->get_distance_to_boundary(phtn.get_position(), phtn.get_angle(), surface_cross);
    const double dist_to_census = phtn.get_distance_remaining();
    events[i].distance = std::min(dist_to_scatter, std::min(dist_to_boundary, dist_to_census));
    events[i].photon_index = photon_index; // Store original index

    if (events[i].distance == dist_to_scatter)
      events[i].type = SCATTER;
    else if (events[i].distance == dist_to_boundary)
      events[i].type = BOUNDARY;
    else
      events[i].type = CENSUS;
  }
}

// Process scatter events for AOS
inline void process_scatter_events(Photon *photon_array,
    const std::vector<double>& sigma_s,
    const std::vector<double>& total_sigma_s,
    const std::vector<uint32_t>& local_cell_indices,
    const Cell* cells,
    const size_t* scatter_indices,
    size_t scatter_count,
    const std::vector<EmissionGroupData>& emission_groups,
    std::vector<PhotonTrackingData>& tracking_data)
{
#pragma omp simd
  for (size_t i = 0; i < scatter_count; ++i) {
    size_t photon_index = scatter_indices[i];
    uint32_t local_idx = local_cell_indices[i];
    auto& phtn = photon_array[photon_index];
    RNG& rng = phtn.get_rng();
    tracking_data[photon_index].num_interactions += 1;
    phtn.set_angle(get_uniform_angle(rng));
    phtn.set_descriptor(Constants::SCATTER);

    if (rng.generate_random_number() > (sigma_s[i] / total_sigma_s[i])) {
      phtn.set_group(sample_emission_group(rng, emission_groups[local_idx]));
      // chance of more intensive scatter
      if (rng.generate_random_number() <= Constants::intensive_scatter_fraction) {
        auto group = phtn.get_group();
        // get a frequency (faux multigroup so just sample from wide spectrum)
        double freq = Constants::lower_frequency_bound + static_cast<double>(group)/static_cast<double>(BRANSON_N_GROUPS)*Constants::delta_frequency_bounds;
        auto angle = phtn.get_angle();
        auto new_energy_angle = intensive_scatter(cells[local_idx].get_T_e(), freq, angle, rng);
        phtn.set_angle(new_energy_angle.second);
      }
    }
  }
}

// Process boundary events for AOS
inline void process_boundary_events(Photon *photon_array,
const size_t* boundary_indices, // Original indices
const Cell* cells,
uint32_t rank_cell_offset,
size_t boundary_count,
std::vector<PhotonTrackingData>& tracking_data)
{
#pragma omp simd
for (size_t i = 0; i < boundary_count; ++i) {
size_t photon_index = boundary_indices[i];
auto& phtn = photon_array[photon_index];
uint32_t local_cell_index = phtn.get_cell() - rank_cell_offset;
Cell const* cell = &cells[local_cell_index];
uint32_t surface_cross = 0;
// Recalculate surface_cross
cell->get_distance_to_boundary(phtn.get_position(), phtn.get_angle(), surface_cross);
auto boundary_event = cell->get_bc(surface_cross);
if (boundary_event == Constants::ELEMENT) {
  phtn.set_cell(cell->get_next_cell(surface_cross));
  phtn.set_descriptor(Constants::BOUND); // Active
}
else if (boundary_event == Constants::PROCESSOR) {
  phtn.set_cell(cell->get_next_cell(surface_cross));
  phtn.set_descriptor(Constants::PASS); // Inactive
}
else if (boundary_event == Constants::VACUUM || boundary_event == Constants::SOURCE) {
  phtn.set_descriptor(Constants::EXIT); // Inactive
   if (boundary_event == Constants::VACUUM) {
    tracking_data[photon_index].exited_vacuum = true;
  }
}
else { // REFLECT
  phtn.reflect(surface_cross);
  phtn.set_descriptor(Constants::BOUND); // Active
}
}
}

// Process census events for AOS
inline void process_census_events(Photon *photon_array,
const size_t* census_indices, // Original indices
size_t census_count)
{
#pragma omp simd // Safe for simple assignment
for (size_t i = 0; i < census_count; ++i) {
photon_array[census_indices[i]].set_descriptor(Constants::CENSUS); // Inactive
}
}

// Update photon state for AOS
inline void update_photon_state(Photon *photon_array,
Cell_Tally* cell_tallies,
const std::vector<Event>& events, // Corresponds to active photons
const std::vector<double>& sigma_a, // Corresponds to active photons
const std::vector<double>& f, // Corresponds to active photons
const std::vector<uint32_t>& local_cell_indices, // Corresponds to active photons
size_t active_count,
std::vector<size_t>& scatter_indices, // Output list of original indices
std::vector<size_t>& boundary_indices,// Output list of original indices
std::vector<size_t>& census_indices,  // Output list of original indices
std::vector<size_t>& killed_indices,  // Output list of original indices
size_t& scatter_count, size_t& boundary_count, size_t& census_count, size_t& killed_count, // Output counts
std::vector<PhotonTrackingData>& tracking_data)
{
// Reset counts
scatter_count = 0;
boundary_count = 0;
  census_count = 0;
  killed_count = 0;

  std::vector<double> absorbed_Es(active_count);
#pragma omp simd
  for (size_t i = 0; i < active_count; ++i) {
    size_t photon_idx = events[i].photon_index;
    absorbed_Es[i] = photon_array[photon_idx].get_E() * (1.0 - exp(-sigma_a[i] * f[i] * events[i].distance));
  }

  // Accumulate tallies - cannot SIMD easily
  for (size_t i = 0; i < active_count; ++i) {
    uint32_t local_idx = local_cell_indices[i];
    double track_contrib = (sigma_a[i] > 0.0 && f[i] > 0.0) ? absorbed_Es[i] / (sigma_a[i] * f[i]) : 0.0;
    cell_tallies[local_idx].accumulate_absorbed_E(absorbed_Es[i]);
    cell_tallies[local_idx].accumulate_track_E(track_contrib);
  }

#pragma omp simd
  for (size_t i = 0; i < active_count; ++i) {
    const Event& event = events[i];
    size_t photon_index = event.photon_index;
    auto& phtn = photon_array[photon_index];
    double distance = event.distance;

    phtn.set_E(phtn.get_E() - absorbed_Es[i]);
    phtn.move(distance);
    tracking_data[photon_index].time_in_cell += distance / Constants::c;

    // Check for kill based on energy cutoff FIRST
    if (phtn.below_cutoff(Constants::cutoff_fraction)) {
      cell_tallies[local_cell_indices[i]].accumulate_absorbed_E(phtn.get_E()); // Tally remaining E
      phtn.set_descriptor(Constants::KILLED);
      // This photon will be added to killed_indices below
    }
    // else: Event type determines the next state (descriptor set in process_* functions later)
  }

  // Separate loop to fill event index vectors - cannot SIMD easily
  for (size_t i = 0; i < active_count; ++i) {
      const Event& event = events[i];
      size_t photon_index = event.photon_index; // Original index
      auto& phtn = photon_array[photon_index];
      // Check descriptor first (set above if killed by energy cutoff)
      if (phtn.get_descriptor() == Constants::KILLED) {
          killed_indices[killed_count++] = photon_index;
      } else {
          // Assign to event lists based on the calculated event type
          switch (event.type) {
          case SCATTER:
              scatter_indices[scatter_count++] = photon_index;
              break;
          case BOUNDARY:
              boundary_indices[boundary_count++] = photon_index;
              break;
          case CENSUS:
              census_indices[census_count++] = photon_index;
              break;
          default: // Should not happen
              break;
          }
      }
  }
}

// Main CPU event transport function for AOS
void cpu_event_transport_photons(const uint32_t rank_cell_offset, Photon_Data<std::vector<Photon>> photon_data, size_t batch_start, size_t batch_end, GPU_Setup<std::vector<Photon>> &gpu_setup, int n_omp_threads) {

  Photon *photon_array = photon_data.h_photon_ptr;
  const size_t maxPhotons = batch_end - batch_start;
   if (maxPhotons == 0) return;

  // Tracking data specific to CPU implementation
  std::vector<PhotonTrackingData> tracking_data(photon_data.n_photons);
   for (size_t i = batch_start; i < batch_end; ++i) {
    tracking_data[i].initial_angle_x = photon_array[i].get_angle()[0];
    tracking_data[i].final_angle_x = photon_array[i].get_angle()[0]; // Initialize
    tracking_data[i].time_in_cell = 0.0;
    tracking_data[i].num_interactions = 0;
    tracking_data[i].entered_domain = (photon_array[i].get_angle()[0] > 0); // Example condition
    tracking_data[i].exited_vacuum = false;
  }

  std::vector<size_t> scatter_indices(maxPhotons), boundary_indices(maxPhotons), census_indices(maxPhotons), killed_indices(maxPhotons), active_photons_indices(maxPhotons);
  std::vector<Event> events(maxPhotons);
  std::vector<uint32_t> local_cell_indices(maxPhotons);
  std::vector<double> sigma_s(maxPhotons), sigma_a(maxPhotons), f(maxPhotons), total_sigma_s(maxPhotons);

  std::iota(active_photons_indices.begin(), active_photons_indices.end(), batch_start);
  size_t active_count = maxPhotons;

  const Cell* cells_ptr = gpu_setup.get_host_cells_ptr(); // CPU pointer here
  Cell_Tally* tallies_ptr = gpu_setup.get_device_cell_tallies_ptr(); // CPU pointer here

  while (active_count > 0) {
    // Resize intermediate vectors
    events.resize(active_count);
    local_cell_indices.resize(active_count);
    sigma_s.resize(active_count);
    sigma_a.resize(active_count);
    f.resize(active_count);
    total_sigma_s.resize(active_count);

    size_t scatter_count = 0, boundary_count = 0, census_count = 0, killed_count = 0;

    // 1. Precompute
    precompute_data(rank_cell_offset, photon_array, cells_ptr, active_photons_indices.data(), active_count, sigma_s, sigma_a, f, total_sigma_s, local_cell_indices);

    // 2. Calculate distances
    calculate_distances(photon_array, cells_ptr, active_photons_indices.data(), active_count, total_sigma_s, local_cell_indices, events);

    // 3. Update state and classify
    scatter_indices.resize(active_count);
    boundary_indices.resize(active_count);
    census_indices.resize(active_count);
    killed_indices.resize(active_count);
    update_photon_state(photon_array, tallies_ptr, events, sigma_a, f, local_cell_indices, active_count, scatter_indices, boundary_indices, census_indices, killed_indices, scatter_count, boundary_count, census_count, killed_count, tracking_data);

    // 4. Process events
    if (scatter_count > 0) {
        // Rebuild subset physics data for scatter events
        std::vector<double> subset_sigma_s(scatter_count);
        std::vector<double> subset_total_sigma_s(scatter_count);
        std::vector<uint32_t> subset_local_indices(scatter_count);
        std::vector<size_t> active_to_subset_map(photon_data.n_photons, photon_data.n_photons);
        for(size_t i=0; i<active_count; ++i) active_to_subset_map[active_photons_indices[i]] = i;
        for(size_t i=0; i<scatter_count; ++i) {
            size_t original_index = scatter_indices[i];
            size_t active_list_pos = active_to_subset_map[original_index];
            if (active_list_pos < active_count) {
                 subset_sigma_s[i] = sigma_s[active_list_pos];
                 subset_total_sigma_s[i] = total_sigma_s[active_list_pos];
                 subset_local_indices[i] = local_cell_indices[active_list_pos];
            } else { /* error */ }
        }
        process_scatter_events(photon_array, subset_sigma_s, subset_total_sigma_s, subset_local_indices, cells_ptr, scatter_indices.data(), scatter_count,  gpu_setup.get_emission_groups(), tracking_data);
    }
     if (boundary_count > 0) {
        process_boundary_events(photon_array, boundary_indices.data(), cells_ptr, rank_cell_offset, boundary_count, tracking_data);
    }
    if (census_count > 0) {
        process_census_events(photon_array, census_indices.data(), census_count);
    }

    // 5. Compact active list
    std::vector<size_t> next_active_photons_indices;
    next_active_photons_indices.reserve(active_count);
    for (size_t i = 0; i < active_count; ++i) {
        size_t index = active_photons_indices[i];
        Constants::event_type desc = photon_array[index].get_descriptor();
        if (desc == Constants::BOUND || desc == Constants::SCATTER) {
            next_active_photons_indices.push_back(index);
        } else {
            tracking_data[index].final_angle_x = photon_array[index].get_angle()[0]; // Update final angle
        }
    }
    active_photons_indices = std::move(next_active_photons_indices);
    active_count = active_photons_indices.size();
  }
}


//----------------------------------------------------------------------------//
// GPU Event-Based Transport Implementation                                   //
//----------------------------------------------------------------------------//
#ifdef USE_GPU

struct EventInfo
{
  uint32_t photon_idx;
  uint32_t active_idx;
};

// Kernel to reset atomic counters
GPU_KERNEL void reset_atomic_counters_kernel(unsigned int* counters, int num_counters) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_counters) {
        counters[idx] = 0;
    }
}

//----------------------------------------------------------------------------//
// SoA GPU Event Kernels                                                      //
//----------------------------------------------------------------------------//



// Function to perform atomic addition within a warp
// __device__ inline void warp_atomic_add(double *address, uint32_t cell_idx, double val) {
//     // Get the mask of active threads in the warp
//     const unsigned int active_mask = __activemask();
//     // Get the mask of threads with matching cell_idx
//     const unsigned int group_mask = __match_any_sync(active_mask, cell_idx);
//     // Perform warp-level reduction sum for the matching threads
//     double subgroup_sum = warp_reduce_sum(val, group_mask);
//     // Check if the current thread is the leader of the matching group
//     // Determine if the current thread is the leader of its group

//     // Calculate the lane ID within the warp (0-31)
//     unsigned int lane_id = threadIdx.x % 32;

//     // Create a mask with only the bit corresponding to this thread's lane ID set to 1
//     unsigned int thread_mask = 1u << lane_id;

//     // Check if this thread's bit is set in the group_mask
//     // If true, this thread is part of the group with matching cell_idx
//     bool is_thread_in_group = (group_mask & thread_mask) != 0;

//     // Find the lane ID of the first active thread in the group
//     // __ffs finds the 1-based index of the least significant set bit, so we subtract 1
//     unsigned int first_active_lane = __ffs(group_mask) - 1;

//     // Determine if this thread is the leader:
//     // It must be part of the group and have the lowest lane ID among group members
//     const bool is_leader = is_thread_in_group && (lane_id == first_active_lane);

//     // If the current thread is the leader, perform atomic addition
//     if (is_leader) {
//         atomicAdd(address, subgroup_sum);
//     }
// }

//----------------------------------------------------------------------------//
// SoA GPU Event Kernels                                                      //
//----------------------------------------------------------------------------//
__global__ void mark_inactive_particles_soa(
    unsigned char *descriptors,
    int32_t* active_indices,
    int32_t active_count)
{
  uint32_t active_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (active_idx >= active_count) return;

  // get the photon at this active indiex
  uint32_t photon_idx = active_indices[active_idx];
  // if photon is not marked as scattered or bound, mark it for removal from active particle queue
  if (!(descriptors[photon_idx] ==  Constants::BOUND  || descriptors[photon_idx] == Constants::SCATTER)) {
    active_indices[active_idx] = -1;
  }
}

__global__ void calculate_events_kernel_soa(
    const uint32_t rank_cell_offset,
    unsigned char *descriptors,
    const uint32_t* group,
    const uint32_t* cell_ID,
    std::array<double,3> *pos,
    std::array<double,3> *angle,
    double *E,
    double *E0,
    double *life_dx,
    RNG *rng,
    const Cell* cells,
    Cell_Tally* cell_tallies,
    int32_t* active_indices,
    int32_t* scatter_indices,  int32_t *boundary_indices, int32_t *census_indices,
    double* absorbed_E, double* track_length_E,
    int32_t active_count)
{
    uint32_t active_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (active_idx >= active_count) return;

    uint32_t photon_idx = active_indices[active_idx];
    uint32_t local_cell_index = cell_ID[photon_idx] - rank_cell_offset;
    // Add bounds check if necessary, assume valid for now
    const Cell* cell = &cells[local_cell_index];

    const double sigma_s = cell->get_op_s(group[photon_idx]);
    const double sigma_a = cell->get_op_a(group[photon_idx]);
    const double f = cell->get_f();
    const double total_sigma_s = (1.0 - f) * sigma_a + sigma_s;

    const double dist_to_scatter = (total_sigma_s > 0.0) ?
      -log(rng[photon_idx].generate_random_number()) / total_sigma_s : 1.0e100;

    uint32_t surface_cross = 0; // Output param, value not needed here
    const double dist_to_boundary = cell->get_distance_to_boundary(
        pos[photon_idx], angle[photon_idx], surface_cross);
    const double dist_to_census = life_dx[photon_idx];

    double distance = min(dist_to_scatter, min(dist_to_boundary, dist_to_census));

    // Calculate and tally absorbed energy
    double event_abs_E = E[photon_idx] * (1.0 - exp(-sigma_a * f * distance));
    absorbed_E[photon_idx] += event_abs_E;
    track_length_E[photon_idx] += (sigma_a > 0.0 && f > 0.0) ? event_abs_E / (sigma_a * f) : 0.0;

    // Update photon state
    E[photon_idx] =E[photon_idx] - event_abs_E;
    pos[photon_idx][0] += angle[photon_idx][0] * distance;
    pos[photon_idx][1] += angle[photon_idx][1] * distance;
    pos[photon_idx][2] += angle[photon_idx][2] * distance;
    life_dx[photon_idx] -= distance;

    // Check for energy kill
    if (E[photon_idx]/E0[photon_idx] <Constants::cutoff_fraction) {
      atomicAdd(&cell_tallies[local_cell_index].abs_E, E[photon_idx] + absorbed_E[photon_idx]); // Tally remaining E
      atomicAdd(&cell_tallies[local_cell_index].track_E, track_length_E[photon_idx]);
      absorbed_E[photon_idx] = 0.0;
      track_length_E[photon_idx] = 0.0;
      E[photon_idx] = 0.0; // Zero out energy
      descriptors[photon_idx] = Constants::KILLED; // Set final descriptor
      distance = -1.0; // used to exclude from all event queues below
    }

    // put real photon index in the event array if photon had event (and wasn't killed
    scatter_indices[active_idx] = (distance == dist_to_scatter) ?  photon_idx : -1;
    boundary_indices[active_idx] = (distance == dist_to_boundary) ? photon_idx : -1;
    census_indices[active_idx] = (distance == dist_to_census) ? photon_idx : -1;
}

__global__ void process_scatter_kernel_soa(
    const uint32_t* cell_ID,
    uint32_t *group,
    unsigned char *descriptors,
    std::array<double,3> *angle,
    RNG *rng,
    const Cell* cells,
    const EmissionGroupData* emission_groups,
    const int32_t* scatter_indices,
    uint32_t scatter_count,
    const uint32_t rank_cell_offset)
{
    uint32_t scatter_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (scatter_idx >= scatter_count) return;

    uint32_t photon_idx = scatter_indices[scatter_idx];

    uint32_t local_cell_index = cell_ID[photon_idx] - rank_cell_offset;
    const Cell* cell = &cells[local_cell_index];
    const EmissionGroupData* emission_data = &emission_groups[local_cell_index];

    const double sigma_s = cell->get_op_s(group[photon_idx]);
    const double sigma_a = cell->get_op_a(group[photon_idx]);
    const double f = cell->get_f();
    const double total_sigma_s = (1.0 - f) * sigma_a + sigma_s; // Recalculate needed data

    angle[photon_idx] = get_uniform_angle(rng[photon_idx]);
    if (total_sigma_s > 0.0 && rng[photon_idx].generate_random_number() > (sigma_s / total_sigma_s)) {
      group[photon_idx] = sample_emission_group(rng[photon_idx], *emission_data);
      // chance of more intensive scatter
      if (rng[photon_idx].generate_random_number() <= Constants::intensive_scatter_fraction) {
        // get a frequency (faux multigroup so just sample from wide spectrum)
        double freq = Constants::lower_frequency_bound + static_cast<double>(group[photon_idx])/static_cast<double>(BRANSON_N_GROUPS)*Constants::delta_frequency_bounds;
        auto new_energy_angle = intensive_scatter(cell->get_T_e(), freq, angle[photon_idx], rng[photon_idx]);
        angle[photon_idx] = new_energy_angle.second;
      }
    }
    descriptors[photon_idx] = Constants::SCATTER; // Mark as scattered (still active for next round)
}

__global__ void process_boundary_kernel_soa(
    unsigned char *descriptors,
    uint32_t* cell_ID,
    std::array<double,3> *pos,
    std::array<double,3> *angle,
    const Cell* cells,
    const int32_t* boundary_indices,
    uint32_t boundary_count,
    double* absorbed_E, double* track_length_E,
    Cell_Tally* cell_tallies,
    const uint32_t rank_cell_offset)
{
    uint32_t boundary_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (boundary_idx >= boundary_count) return;

    uint32_t photon_idx = boundary_indices[boundary_idx];

    uint32_t local_cell_index = cell_ID[photon_idx] - rank_cell_offset;
    const Cell* cell = &cells[local_cell_index];

    uint32_t surface_cross = 0;
    cell->get_distance_to_boundary(pos[photon_idx], angle[photon_idx], surface_cross); // Recalculate surface

    auto boundary_event = cell->get_bc(surface_cross);

    if (boundary_event == Constants::ELEMENT) {
        cell_ID[photon_idx] = cell->get_next_cell(surface_cross);
        descriptors[photon_idx] = (Constants::BOUND); // Still active
        // particle leaving cell, tally energy before leaving and reset tally
        atomicAdd(&cell_tallies[local_cell_index].abs_E, absorbed_E[photon_idx]);
        atomicAdd(&cell_tallies[local_cell_index].track_E, track_length_E[photon_idx]);
        absorbed_E[photon_idx] = 0.0;
        track_length_E[photon_idx] = 0.0;
    } else if (boundary_event == Constants::PROCESSOR) {
        cell_ID[photon_idx] = cell->get_next_cell(surface_cross); // Set global ID for post-processing
        descriptors[photon_idx] = Constants::PASS; // Inactive
        // particle leaving cell, tally energy before leaving and reset tally
        atomicAdd(&cell_tallies[local_cell_index].abs_E, absorbed_E[photon_idx]);
        atomicAdd(&cell_tallies[local_cell_index].track_E, track_length_E[photon_idx]);
    } else if (boundary_event == Constants::VACUUM || boundary_event == Constants::SOURCE) {
        descriptors[photon_idx] = Constants::EXIT; // Inactive
        // particle leaving cell, tally energy before leaving
        atomicAdd(&cell_tallies[local_cell_index].abs_E, absorbed_E[photon_idx]);
        atomicAdd(&cell_tallies[local_cell_index].track_E, track_length_E[photon_idx]);
    } else { // REFLECT
        int reflect_dim = surface_cross / 2;
        angle[photon_idx][reflect_dim] =
          -angle[photon_idx][reflect_dim];
        descriptors[photon_idx] = Constants::BOUND; // Still active
    }
}

__global__ void process_census_kernel_soa(
    unsigned char *descriptors,
    const uint32_t* cell_ID,
    const int32_t* census_indices,
    uint32_t census_count,
    double* absorbed_E, double* track_length_E,
    Cell_Tally* cell_tallies, const uint32_t rank_cell_offset)
{
    uint32_t census_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (census_idx >= census_count) return;

    uint32_t photon_idx = census_indices[census_idx];
    descriptors[photon_idx] = Constants::CENSUS;
    // particle done, tally energy before leaving and reset tally
    uint32_t local_cell_index = cell_ID[photon_idx] - rank_cell_offset;
    atomicAdd(&cell_tallies[local_cell_index].abs_E, absorbed_E[photon_idx]);
    atomicAdd(&cell_tallies[local_cell_index].track_E, track_length_E[photon_idx]);
    absorbed_E[photon_idx] = 0.0;
    track_length_E[photon_idx] = 0.0;
}


//----------------------------------------------------------------------------//
// AoS GPU Event Kernels                                                      //
//----------------------------------------------------------------------------//
__global__ void mark_inactive_particles(
    Photon* all_photons,
    int32_t* active_indices,
    int32_t active_count)
{
  uint32_t active_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (active_idx >= active_count) return;

  // get the photon at this active indiex
  uint32_t photon_idx = active_indices[active_idx];
  const Photon& phtn = all_photons[photon_idx];

  // if photon is not marked as scattered or bound, mark it for removal from active particle queue
  if (!(phtn.get_descriptor() ==  Constants::BOUND  || phtn.get_descriptor() == Constants::SCATTER)) {
    active_indices[active_idx] = -1;
  }
}

__global__ void calculate_events_kernel_aos(
    const uint32_t rank_cell_offset,
    Photon* all_photons,
    const Cell* cells,
    Cell_Tally* cell_tallies,
    int32_t* active_indices,
    int32_t* scatter_indices,  int32_t *boundary_indices, int32_t *census_indices,
    double* absorbed_E, double* track_length_E,
    int32_t active_count)
{
    uint32_t active_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (active_idx >= active_count) return;

    uint32_t photon_idx = active_indices[active_idx];
    Photon& phtn = all_photons[photon_idx];
    RNG& rng = phtn.get_rng();

    uint32_t local_cell_index = phtn.get_cell() - rank_cell_offset;
    // Add bounds check if necessary, assume valid for now
    const Cell* cell = &cells[local_cell_index];

    const double sigma_s = cell->get_op_s(phtn.get_group());
    const double sigma_a = cell->get_op_a(phtn.get_group());
    const double f = cell->get_f();
    const double total_sigma_s = (1.0 - f) * sigma_a + sigma_s;

    const double dist_to_scatter = (total_sigma_s > 0.0) ?
      -log(rng.generate_random_number()) / total_sigma_s : 1.0e100;

    uint32_t surface_cross = 0; // Output param, value not needed here
    const double dist_to_boundary = cell->get_distance_to_boundary(
        phtn.get_position(), phtn.get_angle(), surface_cross);
    const double dist_to_census = phtn.get_distance_remaining();

    double distance = min(dist_to_scatter, min(dist_to_boundary, dist_to_census));

    // Calculate and tally absorbed energy
    double event_abs_E = phtn.get_E() * (1.0 - exp(-sigma_a * f * distance));
    absorbed_E[photon_idx] += event_abs_E;
    track_length_E[photon_idx] += (sigma_a > 0.0 && f > 0.0) ? event_abs_E / (sigma_a * f) : 0.0;

    // Update photon state
    phtn.set_E(phtn.get_E() - event_abs_E);
    phtn.move(distance);

    // Check for energy kill
    if (phtn.below_cutoff(Constants::cutoff_fraction)) {
        atomicAdd(&cell_tallies[local_cell_index].abs_E, phtn.get_E() + absorbed_E[photon_idx]); // Tally remaining E
        atomicAdd(&cell_tallies[local_cell_index].track_E, track_length_E[photon_idx]);
        absorbed_E[photon_idx] = 0.0;
        track_length_E[photon_idx] = 0.0;
        phtn.set_E(0.0); // Zero out energy
        phtn.set_descriptor(Constants::KILLED); // Set final descriptor
        distance = -1.0; // used to exclude from all event queues below
    }

    // put real photon index in the event array if photon had event (and wasn't killed
    scatter_indices[active_idx] = (distance == dist_to_scatter) ?  photon_idx : -1;
    boundary_indices[active_idx] = (distance == dist_to_boundary) ? photon_idx : -1;
    census_indices[active_idx] = (distance == dist_to_census) ? photon_idx : -1;
}

__global__ void process_scatter_kernel_aos(
    Photon* all_photons,
    const Cell* cells,
    const EmissionGroupData* emission_groups,
    const int32_t* scatter_indices,
    uint32_t scatter_count,
    const uint32_t rank_cell_offset)
{
    uint32_t scatter_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (scatter_idx >= scatter_count) return;

    uint32_t photon_idx = scatter_indices[scatter_idx];
    Photon& phtn = all_photons[photon_idx];
    RNG& rng = phtn.get_rng();

    uint32_t local_cell_index = phtn.get_cell() - rank_cell_offset;
    const Cell* cell = &cells[local_cell_index];
    const EmissionGroupData* emission_data = &emission_groups[local_cell_index];

    const double sigma_s = cell->get_op_s(phtn.get_group());
    const double sigma_a = cell->get_op_a(phtn.get_group());
    const double f = cell->get_f();
    const double total_sigma_s = (1.0 - f) * sigma_a + sigma_s; // Recalculate needed data

    phtn.set_angle(get_uniform_angle(rng));
    if (total_sigma_s > 0.0 && rng.generate_random_number() > (sigma_s / total_sigma_s)) {
      phtn.set_group(sample_emission_group(rng, *emission_data));
      // chance of more intensive scatter
      if (rng.generate_random_number() <= Constants::intensive_scatter_fraction) {
        // get a frequency (faux multigroup so just sample from wide spectrum)
        double freq = Constants::lower_frequency_bound + static_cast<double>(phtn.get_group())/static_cast<double>(BRANSON_N_GROUPS)*Constants::delta_frequency_bounds;
        auto new_energy_angle = intensive_scatter(cell->get_T_e(), freq, phtn.get_angle(), rng);
        phtn.set_angle(new_energy_angle.second);
      }
    }
    phtn.set_descriptor(Constants::SCATTER); // Mark as scattered (still active for next round)
}

__global__ void process_boundary_kernel_aos(
    Photon* all_photons,
    const Cell* cells,
    const int32_t* boundary_indices,
    uint32_t boundary_count,
    double* absorbed_E, double* track_length_E,
    Cell_Tally* cell_tallies,
    const uint32_t rank_cell_offset)
{
    uint32_t boundary_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (boundary_idx >= boundary_count) return;

    uint32_t photon_idx = boundary_indices[boundary_idx];
    Photon& phtn = all_photons[photon_idx];

    uint32_t local_cell_index = phtn.get_cell() - rank_cell_offset;
    const Cell* cell = &cells[local_cell_index];

    uint32_t surface_cross = 0;
    cell->get_distance_to_boundary(phtn.get_position(), phtn.get_angle(), surface_cross); // Recalculate surface

    auto boundary_event = cell->get_bc(surface_cross);

    if (boundary_event == Constants::ELEMENT) {
        phtn.set_cell(cell->get_next_cell(surface_cross));
        phtn.set_descriptor(Constants::BOUND); // Still active
        // particle leaving cell, tally energy before leaving and reset tally
        atomicAdd(&cell_tallies[local_cell_index].abs_E, absorbed_E[photon_idx]);
        atomicAdd(&cell_tallies[local_cell_index].track_E, track_length_E[photon_idx]);
        absorbed_E[photon_idx] = 0.0;
        track_length_E[photon_idx] = 0.0;
    } else if (boundary_event == Constants::PROCESSOR) {
        phtn.set_cell(cell->get_next_cell(surface_cross)); // Set global ID for post-processing
        phtn.set_descriptor(Constants::PASS); // Inactive
        // particle leaving cell, tally energy before leaving and reset tally
        atomicAdd(&cell_tallies[local_cell_index].abs_E, absorbed_E[photon_idx]);
        atomicAdd(&cell_tallies[local_cell_index].track_E, track_length_E[photon_idx]);
    } else if (boundary_event == Constants::VACUUM || boundary_event == Constants::SOURCE) {
        phtn.set_descriptor(Constants::EXIT); // Inactive
        // particle leaving cell, tally energy before leaving
        atomicAdd(&cell_tallies[local_cell_index].abs_E, absorbed_E[photon_idx]);
        atomicAdd(&cell_tallies[local_cell_index].track_E, track_length_E[photon_idx]);
    } else { // REFLECT
        phtn.reflect(surface_cross);
        phtn.set_descriptor(Constants::BOUND); // Still active
    }
}

__global__ void process_census_kernel_aos(
    Photon* all_photons,
    const int32_t* census_indices,
    uint32_t census_count,
    double* absorbed_E, double* track_length_E,
    Cell_Tally* cell_tallies, const uint32_t rank_cell_offset)
{
    uint32_t census_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (census_idx >= census_count) return;

    uint32_t photon_idx = census_indices[census_idx];
    all_photons[photon_idx].set_descriptor(Constants::CENSUS);
    // particle done, tally energy before leaving and reset tally
    uint32_t local_cell_index = all_photons[photon_idx].get_cell() - rank_cell_offset;
    atomicAdd(&cell_tallies[local_cell_index].abs_E, absorbed_E[photon_idx]);
    atomicAdd(&cell_tallies[local_cell_index].track_E, track_length_E[photon_idx]);
    absorbed_E[photon_idx] = 0.0;
    track_length_E[photon_idx] = 0.0;
}

//----------------------------------------------------------------------------//
// GPU Transport Function - Event Based                                 //
//----------------------------------------------------------------------------//
template <typename Census_T>
void gpu_event_transport_photons(const uint32_t rank_cell_offset,
    Photon_Data<Census_T> &photon_data, size_t batch_start, size_t batch_end, GPU_Setup<Census_T> &gpu_setup)
{
  Timer t_transport;
  std::string timer_name;
  if constexpr (std::is_same_v<Census_T, std::vector<Photon>>) {
    timer_name ="aos_gpu_event_transport_photons";
  }
  else {
    timer_name ="soa_gpu_event_transport_photons";
  }


  t_transport.start_timer(timer_name);
  uint32_t n_photons = static_cast<uint32_t>(batch_end - batch_start);
  if (n_photons == 0) return;

  size_t n_cells = gpu_setup.get_n_cells();
  cudaError_t err;

  // Active Indices (initialize with 0, 1, ..., n_photons-1 on host first)
  // resets active_indices to iota, and zeros absorbed E and track length E
  photon_data.reset_event_based_data();

  // Event-Specific Index Lists
  int32_t *d_active_indices = photon_data.d_active_indices;
  int32_t *d_scatter_indices = photon_data.d_scatter_indices;
  int32_t *d_boundary_indices = photon_data.d_boundary_indices;
  int32_t *d_census_indices = photon_data.d_census_indices;
  double *d_absorbed_E = photon_data.d_absorbed_E;
  double *d_track_length_E = photon_data.d_absorbed_E;

  // --- Event Loop ---
  uint32_t current_active_count = n_photons;

  int n_threads = Constants::n_threads_per_block;

  // data from GPU_Setup
  Cell *d_cells = gpu_setup.get_device_cells_ptr();
  Cell_Tally *d_cell_tallies= gpu_setup.get_device_cell_tallies_ptr();
  EmissionGroupData *d_emission_groups = gpu_setup.get_emission_groups_ptr();

  if constexpr (std::is_same_v<Census_T, std::vector<Photon>>) {

    t_transport.start_timer("aos kernel");

    Photon *d_photons = photon_data.d_photon_ptr;

    while (current_active_count > 0) {
      int n_blocks = (current_active_count + n_threads - 1) / n_threads;

      // 1. Calculate distances, event types, attenuate, mark inactive particles
      calculate_events_kernel_aos<<<n_blocks, n_threads>>>(rank_cell_offset, d_photons, d_cells,
          d_cell_tallies, d_active_indices, d_scatter_indices, d_boundary_indices, d_census_indices,
          d_absorbed_E, d_track_length_E, current_active_count);
      err = cudaGetLastError(); Insist(!err, "GPU AoS Kernel Launch Error: calculate_events");

      // 2. Sort event indices queues, removing -1 indices
      auto scatter_end =
          thrust::remove(thrust::device, d_scatter_indices, d_scatter_indices + current_active_count, -1);
      auto boundary_end =
          thrust::remove(thrust::device, d_boundary_indices, d_boundary_indices +  current_active_count, -1);
      auto census_end =
          thrust::remove(thrust::device, d_census_indices, d_census_indices + current_active_count, -1);

      // 3. Get event counts
      uint32_t scatter_count = thrust::distance(d_scatter_indices, scatter_end);
      uint32_t boundary_count = thrust::distance(d_boundary_indices, boundary_end);
      uint32_t census_count = thrust::distance(d_census_indices, census_end);

      // 4. Process Events (launch kernels only if counts > 0)
      //std::cout<<"scatter: "<<scatter_count<< " boundary: "<<boundary_count<< " census: "<<census_count<<std::endl;
      if (scatter_count > 0) {
          int scatter_blocks = (scatter_count + n_threads - 1) / n_threads;
          process_scatter_kernel_aos<<<scatter_blocks, n_threads>>>(
              d_photons, d_cells, d_emission_groups,
              d_scatter_indices, scatter_count, rank_cell_offset);
          err = cudaGetLastError(); Insist(!err, "GPU AoS Kernel Launch Error: process_scatter");
      }
      if (boundary_count > 0) {
          int boundary_blocks = (boundary_count + n_threads - 1) / n_threads;
          process_boundary_kernel_aos<<<boundary_blocks, n_threads>>>(
              d_photons, d_cells,
              d_boundary_indices, boundary_count, d_absorbed_E, d_track_length_E, d_cell_tallies, rank_cell_offset);
          err = cudaGetLastError(); Insist(!err, "GPU AoS Kernel Launch Error: process_boundary");
      }
      if (census_count > 0) {
          int census_blocks = (census_count + n_threads - 1) / n_threads;
          process_census_kernel_aos<<<census_blocks, n_threads>>>(
              d_photons, d_census_indices, census_count, d_absorbed_E, d_track_length_E, d_cell_tallies, rank_cell_offset);
          err = cudaGetLastError(); Insist(!err, "GPU AoS Kernel Launch Error: process_census");
      }

      // 5: Update inactive particles
      mark_inactive_particles<<<n_blocks, n_threads>>>(d_photons, d_active_indices, current_active_count);
      err = cudaGetLastError(); Insist(!err, "GPU AoS Kernel Launch Error: mark_inactive_particles");

      // 6. Remove inactive indices and set new active count
      auto active_end =
          thrust::remove(thrust::device, d_active_indices, d_active_indices + current_active_count, -1);
      current_active_count = thrust::distance(d_active_indices, active_end);

    } // End while(current_active_count > 0)

    auto sync_error = cudaDeviceSynchronize(); // Ensure all kernels are finished before copy back
    t_transport.stop_timer("aos kernel");
    Insist(!sync_error, "Error in synchronize");

    t_transport.stop_timer("aos_gpu_event_transport_photons");
  }
  else {
    t_transport.start_timer("soa kernel");

    // set pointers for this batch
    uint32_t *d_cell_ID = photon_data.d_cell_ID_ptr;
    uint32_t *d_group = photon_data.d_group_ptr;
    unsigned char *d_descriptors = photon_data.d_descriptors_ptr;
    std::array<double, 3> *d_pos = photon_data.d_pos_ptr;
    std::array<double, 3> *d_angle = photon_data.d_angle_ptr;
    double *d_E = photon_data.d_E_ptr;
    double *d_E0 = photon_data.d_E0_ptr;
    double *d_life_dx = photon_data.d_life_dx_ptr;
    RNG *d_RNG = photon_data.d_RNG_ptr;

    while (current_active_count > 0) {
      int n_blocks = (current_active_count + n_threads - 1) / n_threads;

      // 1. Calculate distances, event types, attenuate, mark inactive particles
      calculate_events_kernel_soa<<<n_blocks, n_threads>>>(rank_cell_offset, d_descriptors, d_group,
          d_cell_ID, d_pos, d_angle, d_E, d_E0, d_life_dx, d_RNG,  d_cells,
          d_cell_tallies, d_active_indices, d_scatter_indices, d_boundary_indices, d_census_indices,
          d_absorbed_E, d_track_length_E, current_active_count);
      err = cudaGetLastError(); Insist(!err, "GPU AoS Kernel Launch Error: calculate_events");

      // 2. Sort event indices queues, removing -1 indices
      auto scatter_end =
          thrust::remove(thrust::device, d_scatter_indices, d_scatter_indices + current_active_count, -1);
      auto boundary_end =
          thrust::remove(thrust::device, d_boundary_indices, d_boundary_indices +  current_active_count, -1);
      auto census_end =
          thrust::remove(thrust::device, d_census_indices, d_census_indices + current_active_count, -1);

      // 3. Get event counts
      uint32_t scatter_count = thrust::distance(d_scatter_indices, scatter_end);
      uint32_t boundary_count = thrust::distance(d_boundary_indices, boundary_end);
      uint32_t census_count = thrust::distance(d_census_indices, census_end);

      // 4. Process Events (launch kernels only if counts > 0)
      if (scatter_count > 0) {
          int scatter_blocks = (scatter_count + n_threads - 1) / n_threads;
          process_scatter_kernel_soa<<<scatter_blocks, n_threads>>>(
              d_cell_ID, d_group, d_descriptors, d_angle, d_RNG, d_cells, d_emission_groups,
              d_scatter_indices, scatter_count, rank_cell_offset);
          err = cudaGetLastError(); Insist(!err, "GPU AoS Kernel Launch Error: process_scatter");
      }
      if (boundary_count > 0) {
          int boundary_blocks = (boundary_count + n_threads - 1) / n_threads;
          process_boundary_kernel_soa<<<boundary_blocks, n_threads>>>(
              d_descriptors, d_cell_ID, d_pos, d_angle, d_cells,
              d_boundary_indices, boundary_count, d_absorbed_E, d_track_length_E, d_cell_tallies, rank_cell_offset);
          err = cudaGetLastError(); Insist(!err, "GPU AoS Kernel Launch Error: process_boundary");
      }
      if (census_count > 0) {
          int census_blocks = (census_count + n_threads - 1) / n_threads;
          process_census_kernel_soa<<<census_blocks, n_threads>>>(
              d_descriptors, d_cell_ID, d_census_indices, census_count, d_absorbed_E, d_track_length_E, d_cell_tallies, rank_cell_offset);
          err = cudaGetLastError(); Insist(!err, "GPU AoS Kernel Launch Error: process_census");
      }

      // 5: Update inactive particles
      mark_inactive_particles_soa<<<n_blocks, n_threads>>>(d_descriptors, d_active_indices, current_active_count);
      err = cudaGetLastError(); Insist(!err, "GPU AoS Kernel Launch Error: mark_inactive_particles");

      // 6. Remove inactive indices and set new active count
      auto active_end =
          thrust::remove(thrust::device, d_active_indices, d_active_indices + current_active_count, -1);
      current_active_count = thrust::distance(d_active_indices, active_end);
    } // End while(current_active_count > 0)

    auto sync_error = cudaDeviceSynchronize(); // Ensure all kernels are finished before copy back
    t_transport.stop_timer("soa kernel");
    Insist(!sync_error, "Error in synchronize");

    t_transport.stop_timer("soa_gpu_event_transport_photons");
  }
}
#endif // USE_GPU

#endif // event_based_transport_h_
//----------------------------------------------------------------------------//
// end of event_based_transport.h
//----------------------------------------------------------------------------//
