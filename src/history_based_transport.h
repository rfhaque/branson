//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   history_based_transport.h
 * \author Alex Long
 * \date   December 1 2015
 * \brief  IMC transport with particle passing method
 * \note   Copyright (C) 2017 Los Alamos National Security, LLC.
 *         All rights reserved
 */
//---------------------------------------------------------------------------//

#ifndef history_based_transport_h_
#define history_based_transport_h_

#include <algorithm>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "config.h"
#include "RNG.h"
#include "cell_tally.h"
#include "constants.h"
#include "gpu_setup.h"
#include "photon.h"
#include "photon_array.h"
#include "photon_data.h"
#include "sampling_functions.h"

//----------------------------------------------------------------------------//
// AoS Transport Function (CPU Host)
//----------------------------------------------------------------------------//
//! Transport a photon (AoS) when the mesh is always available - CPU version
void transport_photon_history_aos_cpu(const uint32_t rank_cell_offset,
    Photon &phtn, const Cell *cells, Cell_Tally *cell_tallies) {

  using Constants::bc_type;
  using Constants::c;
  using std::min;

  auto &rng = phtn.get_rng();

  uint32_t surface_cross = 0;

  uint32_t local_cell_index =  phtn.get_cell() - rank_cell_offset;
  Cell const * cell = &cells[local_cell_index];
  bool active = true;

  // Use thread-local tallies for OpenMP efficiency, accumulate directly otherwise
  double thread_absorbed_E{0.0};
  double thread_track_E{0.0};

  // transport this photon
  while (active) {
    const double sigma_s = cell->get_op_s(phtn.get_group());
    const double sigma_a = cell->get_op_a(phtn.get_group());
    const double f = cell->get_f();
    const double total_sigma_s = (1.0 - f) * sigma_a + sigma_s;

    // get distance to event
    const double dist_to_scatter = (total_sigma_s > 0.0) ?
      -log(rng.generate_random_number()) / total_sigma_s : 1.0e100;

    const double dist_to_boundary = cell->get_distance_to_boundary(
        phtn.get_position(), phtn.get_angle(), surface_cross);
    const double dist_to_census = phtn.get_distance_remaining();

    // select minimum distance event
    const double dist_to_event = min(dist_to_scatter, min(dist_to_boundary, dist_to_census));

    // calculate energy absorbed by material, update photon and material energy
    // and update the path-length weighted tally for T_r
    const double absorbed_E = phtn.get_E() * (1.0 - exp(-sigma_a * f * dist_to_event));

    thread_absorbed_E += absorbed_E;
    // Avoid division by zero if sigma_a or f is zero
    if (sigma_a > 1.0e-100 && f > 1.0e-100)
        thread_track_E += absorbed_E / (sigma_a * f);

    phtn.set_E(phtn.get_E() - absorbed_E);

    // update position
    phtn.move(dist_to_event);

    // apply variance/runtime reduction
    if (phtn.below_cutoff(Constants::cutoff_fraction)) {
      thread_absorbed_E += phtn.get_E();
      // Accumulate directly into the provided tally pointer (could be thread-local or global)
      cell_tallies[local_cell_index].accumulate_absorbed_E(thread_absorbed_E);
      cell_tallies[local_cell_index].accumulate_track_E(thread_track_E);
      active = false;
      phtn.set_descriptor(Constants::KILLED);
    }
    // or apply event
    else {
      // EVENT TYPE: SCATTER
      if (dist_to_event == dist_to_scatter) {
        phtn.set_angle(get_uniform_angle(rng));
        if (rng.generate_random_number() > (sigma_s / total_sigma_s)) {
          phtn.set_group(sample_emission_group(rng, *cell));
          // chance of more intensive scatter
          if (rng.generate_random_number() <= Constants::intensive_scatter_fraction) {
            auto group = phtn.get_group();
            // get a frequency (faux multigroup so just sample from wide spectrum)
            double freq = Constants::lower_frequency_bound + static_cast<double>(group)/static_cast<double>(BRANSON_N_GROUPS)*Constants::delta_frequency_bounds;
            auto angle = phtn.get_angle();
            auto new_energy_angle = intensive_scatter(cell->get_T_e(), freq, angle, rng);
            phtn.set_angle(new_energy_angle.second);
          }
        }
        phtn.set_descriptor(Constants::SCATTER);
      }
      // EVENT TYPE: BOUNDARY CROSS
      else if (dist_to_event == dist_to_boundary) {
        auto boundary_event = cell->get_bc(surface_cross);
        if (boundary_event == Constants::ELEMENT) {
          cell_tallies[local_cell_index].accumulate_absorbed_E(thread_absorbed_E);
          cell_tallies[local_cell_index].accumulate_track_E(thread_track_E);
          phtn.set_cell(cell->get_next_cell(surface_cross));
          local_cell_index =  phtn.get_cell() - rank_cell_offset;
          cell = &cells[local_cell_index];
          phtn.set_descriptor(Constants::BOUND);
          thread_absorbed_E = 0.0;
          thread_track_E = 0.0;
        } else if (boundary_event == Constants::PROCESSOR) {
          active = false;
          phtn.set_cell(cell->get_next_cell(surface_cross));
          phtn.set_descriptor(Constants::PASS);
          cell_tallies[local_cell_index].accumulate_absorbed_E(thread_absorbed_E);
          cell_tallies[local_cell_index].accumulate_track_E(thread_track_E);
        } else if (boundary_event == Constants::VACUUM || boundary_event == Constants::SOURCE) {
          active = false;
          phtn.set_descriptor(Constants::EXIT);
          cell_tallies[local_cell_index].accumulate_absorbed_E(thread_absorbed_E);
          cell_tallies[local_cell_index].accumulate_track_E(thread_track_E);
        } else { // REFLECT
          phtn.reflect(surface_cross);
          phtn.set_descriptor(Constants::BOUND);
        }
      }
      // EVENT TYPE: REACH CENSUS
      else if (dist_to_event == dist_to_census) {
        active = false;
        phtn.set_descriptor(Constants::CENSUS);
        cell_tallies[local_cell_index].accumulate_absorbed_E(thread_absorbed_E);
        cell_tallies[local_cell_index].accumulate_track_E(thread_track_E);
      }
    } // end event loop
  } // end while alive
}


//----------------------------------------------------------------------------//
// SoA Transport Function (CPU Host)
//----------------------------------------------------------------------------//
//! Transport a photon (SoA) when the mesh is always available - CPU version
void transport_photon_history_soa_cpu(const uint32_t rank_cell_offset,
    uint32_t &cell_ID,
    uint32_t &group,
    unsigned char &descriptor,
    std::array<double, 3> &pos,
    std::array<double, 3> &angle,
    double &E,
    double E0,
    double &life_dx,
    RNG &rng,
    const Cell *cells, Cell_Tally *cell_tallies) {

  using Constants::bc_type;
  using Constants::c;
  using std::min;

  uint32_t surface_cross = 0;

  uint32_t local_cell_index =  cell_ID - rank_cell_offset;
  Cell const * cell = &cells[local_cell_index];
  bool active = true;

  double thread_absorbed_E{0.0};
  double thread_track_E{0.0};

  // transport this photon
  while (active) {
    const double sigma_s = cell->get_op_s(group);
    const double sigma_a = cell->get_op_a(group);
    const double f = cell->get_f();
    const double total_sigma_s = (1.0 - f) * sigma_a + sigma_s;

    // get distance to event
    const double dist_to_scatter = (total_sigma_s > 0.0) ?
      -log(rng.generate_random_number()) / total_sigma_s : 1.0e100;

    const double dist_to_boundary = cell->get_distance_to_boundary(
        pos, angle, surface_cross);
    const double dist_to_census = life_dx;

    // select minimum distance event
    const double dist_to_event = min(dist_to_scatter, min(dist_to_boundary, dist_to_census));

    // calculate energy absorbed by material, update photon and material energy
    // and update the path-length weighted tally for T_r
    const double absorbed_E = E * (1.0 - exp(-sigma_a * f * dist_to_event));

    thread_absorbed_E += absorbed_E;
    // Avoid division by zero if sigma_a or f is zero
    if (sigma_a > 1.0e-100 && f > 1.0e-100)
        thread_track_E += absorbed_E / (sigma_a * f);


    E = (E - absorbed_E);

    // update position
    pos[0] += angle[0] * dist_to_event;
    pos[1] += angle[1] * dist_to_event;
    pos[2] += angle[2] * dist_to_event;
    life_dx -= dist_to_event;

    // apply runtime reduction
    if (E / E0 < Constants::cutoff_fraction) {
      thread_absorbed_E += E;
      cell_tallies[local_cell_index].accumulate_absorbed_E(thread_absorbed_E);
      cell_tallies[local_cell_index].accumulate_track_E(thread_track_E);
      active = false;
      descriptor = static_cast<unsigned char>(Constants::KILLED);
    }
    // or apply event
    else {
      // EVENT TYPE: SCATTER
      if (dist_to_event == dist_to_scatter) {
        angle = get_uniform_angle(rng);
        if (rng.generate_random_number() > (sigma_s / total_sigma_s)) {
          group = sample_emission_group(rng, *cell);
          // chance of more intensive scatter
          if (rng.generate_random_number() <= Constants::intensive_scatter_fraction) {
            // get a frequency (faux multigroup so just sample from wide spectrum)
            double freq = Constants::lower_frequency_bound + static_cast<double>(group)/static_cast<double>(BRANSON_N_GROUPS)*Constants::delta_frequency_bounds;
            auto new_energy_angle = intensive_scatter(cell->get_T_e(), freq, angle, rng);
            angle = new_energy_angle.second;
          }
        }
        descriptor = Constants::SCATTER;
      }
      // EVENT TYPE: BOUNDARY CROSS
      else if (dist_to_event == dist_to_boundary) {
        auto boundary_event = cell->get_bc(surface_cross);
        if (boundary_event == Constants::ELEMENT) {
          cell_tallies[local_cell_index].accumulate_absorbed_E(thread_absorbed_E);
          cell_tallies[local_cell_index].accumulate_track_E(thread_track_E);
          cell_ID = cell->get_next_cell(surface_cross);
          local_cell_index =  cell_ID - rank_cell_offset;
          cell = &cells[local_cell_index];
          descriptor = static_cast<unsigned char>(Constants::BOUND);
          thread_absorbed_E = 0.0;
          thread_track_E = 0.0;
        } else if (boundary_event == Constants::PROCESSOR) {
          active = false;
          cell_ID = cell->get_next_cell(surface_cross);
          descriptor = static_cast<unsigned char>(Constants::PASS);
          cell_tallies[local_cell_index].accumulate_absorbed_E(thread_absorbed_E);
          cell_tallies[local_cell_index].accumulate_track_E(thread_track_E);
        } else if (boundary_event == Constants::VACUUM || boundary_event == Constants::SOURCE) {
          active = false;
          descriptor = static_cast<unsigned char>(Constants::EXIT);
          cell_tallies[local_cell_index].accumulate_absorbed_E(thread_absorbed_E);
          cell_tallies[local_cell_index].accumulate_track_E(thread_track_E);
        } else { // REFLECT
          int reflect_angle = surface_cross/2; // X -> 0, Y->1, Z->2
          angle[reflect_angle] = -angle[reflect_angle];
          descriptor = static_cast<unsigned char>(Constants::BOUND);
        }
      }
      // EVENT TYPE: REACH CENSUS
      else if (dist_to_event == dist_to_census) {
        active = false;
        descriptor = static_cast<unsigned char>(Constants::CENSUS);
        cell_tallies[local_cell_index].accumulate_absorbed_E(thread_absorbed_E);
        cell_tallies[local_cell_index].accumulate_track_E(thread_track_E);
      }
    } // end event loop
  } // end while alive
}


//----------------------------------------------------------------------------//
// AoS Transport Function (GPU Device)
//----------------------------------------------------------------------------//
//! Transport a photon (AoS) when the mesh is always available - GPU version
GPU_DEVICE
void transport_photon_history_aos_gpu(const uint32_t rank_cell_offset,
    Photon &phtn, const Cell *cells, Cell_Tally *cell_tallies) {

  using Constants::bc_type;
  using Constants::c;
  using std::min; // Use std::min for GPU compatibility if needed, or define min

  auto &rng = phtn.get_rng();

  uint32_t surface_cross = 0;

  uint32_t local_cell_index =  phtn.get_cell() - rank_cell_offset;
  Cell const * cell = &cells[local_cell_index];
  bool active = true;

  double thread_absorbed_E{0.0};
  double thread_track_E{0.0};

  // transport this photon
  while (active) {
    const double sigma_s = cell->get_op_s(phtn.get_group());
    const double sigma_a = cell->get_op_a(phtn.get_group());
    const double f = cell->get_f();
    const double total_sigma_s = (1.0 - f) * sigma_a + sigma_s;

    // get distance to event
    const double dist_to_scatter = (total_sigma_s > 0.0) ?
      -log(rng.generate_random_number()) / total_sigma_s : 1.0e100;

    const double dist_to_boundary = cell->get_distance_to_boundary(
        phtn.get_position(), phtn.get_angle(), surface_cross);
    const double dist_to_census = phtn.get_distance_remaining();

    // select minimum distance event
    const double dist_to_event = min(dist_to_scatter, min(dist_to_boundary, dist_to_census));

    // calculate energy absorbed by material, update photon and material energy
    // and update the path-length weighted tally for T_r
    const double absorbed_E = phtn.get_E() * (1.0 - exp(-sigma_a * f * dist_to_event));

    thread_absorbed_E += absorbed_E;
    // Avoid division by zero if sigma_a or f is zero
    if (sigma_a > 1.0e-100 && f > 1.0e-100)
        thread_track_E += absorbed_E / (sigma_a * f);

    phtn.set_E(phtn.get_E() - absorbed_E);

    // update position
    phtn.move(dist_to_event);

    // apply variance/runtime reduction
    if (phtn.below_cutoff(Constants::cutoff_fraction)) {
      thread_absorbed_E += phtn.get_E();
#ifdef USE_GPU// Use atomicAdd for GPU tallies
      atomicAdd(&cell_tallies[local_cell_index].abs_E, thread_absorbed_E);
      atomicAdd(&cell_tallies[local_cell_index].track_E, thread_track_E);
#else // Should not happen if GPU_DEVICE is defined correctly
      cell_tallies[local_cell_index].accumulate_absorbed_E(thread_absorbed_E);
      cell_tallies[local_cell_index].accumulate_track_E(thread_track_E);
#endif
      active = false;
      phtn.set_descriptor(Constants::KILLED);
    }
    // or apply event
    else {
      // EVENT TYPE: SCATTER
      if (dist_to_event == dist_to_scatter) {
        phtn.set_angle(get_uniform_angle(rng));
        if (rng.generate_random_number() > (sigma_s / total_sigma_s)) {
          phtn.set_group(sample_emission_group(rng, *cell));
          // chance of more intensive scatter
          if (rng.generate_random_number() <= Constants::intensive_scatter_fraction) {
            auto group = phtn.get_group();
            // get a frequency (faux multigroup so just sample from wide spectrum)
            double freq = Constants::lower_frequency_bound + static_cast<double>(group)/static_cast<double>(BRANSON_N_GROUPS)*Constants::delta_frequency_bounds;
            auto angle = phtn.get_angle();
            auto new_energy_angle = intensive_scatter(cell->get_T_e(), freq, angle, rng);
            phtn.set_angle(new_energy_angle.second);
          }
        }
        phtn.set_descriptor(Constants::SCATTER);
      }
      // EVENT TYPE: BOUNDARY CROSS
      else if (dist_to_event == dist_to_boundary) {
        auto boundary_event = cell->get_bc(surface_cross);
        if (boundary_event == Constants::ELEMENT) {
#ifdef USE_GPU
          atomicAdd(&cell_tallies[local_cell_index].abs_E, thread_absorbed_E);
          atomicAdd(&cell_tallies[local_cell_index].track_E, thread_track_E);
#else
          cell_tallies[local_cell_index].accumulate_absorbed_E(thread_absorbed_E);
          cell_tallies[local_cell_index].accumulate_track_E(thread_track_E);
#endif
          phtn.set_cell(cell->get_next_cell(surface_cross));
          local_cell_index =  phtn.get_cell() - rank_cell_offset;
          cell = &cells[local_cell_index];
          phtn.set_descriptor(Constants::BOUND);
          thread_absorbed_E = 0.0;
          thread_track_E = 0.0;
        } else if (boundary_event == Constants::PROCESSOR) {
          active = false;
          phtn.set_cell(cell->get_next_cell(surface_cross));
          phtn.set_descriptor(Constants::PASS);
#ifdef USE_GPU
          atomicAdd(&cell_tallies[local_cell_index].abs_E, thread_absorbed_E);
          atomicAdd(&cell_tallies[local_cell_index].track_E, thread_track_E);
#else
          cell_tallies[local_cell_index].accumulate_absorbed_E(thread_absorbed_E);
          cell_tallies[local_cell_index].accumulate_track_E(thread_track_E);
#endif
        } else if (boundary_event == Constants::VACUUM || boundary_event == Constants::SOURCE) {
          active = false;
          phtn.set_descriptor(Constants::EXIT);
#ifdef USE_GPU
          atomicAdd(&cell_tallies[local_cell_index].abs_E, thread_absorbed_E);
          atomicAdd(&cell_tallies[local_cell_index].track_E, thread_track_E);
#else
          cell_tallies[local_cell_index].accumulate_absorbed_E(thread_absorbed_E);
          cell_tallies[local_cell_index].accumulate_track_E(thread_track_E);
#endif
        } else { // REFLECT
          phtn.reflect(surface_cross);
          phtn.set_descriptor(Constants::BOUND);
        }
      }
      // EVENT TYPE: REACH CENSUS
      else if (dist_to_event == dist_to_census) {
        active = false;
        phtn.set_descriptor(Constants::CENSUS);
#ifdef USE_GPU
        atomicAdd(&cell_tallies[local_cell_index].abs_E, thread_absorbed_E);
        atomicAdd(&cell_tallies[local_cell_index].track_E, thread_track_E);
#else
        cell_tallies[local_cell_index].accumulate_absorbed_E(thread_absorbed_E);
        cell_tallies[local_cell_index].accumulate_track_E(thread_track_E);
#endif
      }
    } // end event loop
  } // end while alive
}
//----------------------------------------------------------------------------//
// SoA Transport Function (GPU Device)
//----------------------------------------------------------------------------//
//! Transport a photon (SoA) when the mesh is always available - GPU version
GPU_DEVICE
void transport_photon_history_soa_gpu(const uint32_t rank_cell_offset,
    const size_t i, // index of the photon
    uint32_t* cell_ID_ptr, uint32_t* group_ptr, unsigned char* descriptors_ptr,
    std::array<double, 3>* pos_ptr, std::array<double, 3>* angle_ptr,
    double* E_ptr, double* E0_ptr, double* life_dx_ptr, RNG* rng_ptr,
    const Cell *cells, Cell_Tally *cell_tallies) {

  using Constants::bc_type;
  using Constants::c;
  using std::min;

  RNG &rng = rng_ptr[i];

  uint32_t surface_cross = 0;

  uint32_t local_cell_index =  cell_ID_ptr[i] - rank_cell_offset;
  Cell const * cell = &cells[local_cell_index];
  bool active = true;

  double thread_absorbed_E{0.0};
  double thread_track_E{0.0};

  // transport this photon
  while (active) {
    const double sigma_s = cell->get_op_s(group_ptr[i]);
    const double sigma_a = cell->get_op_a(group_ptr[i]);
    const double f = cell->get_f();
    const double total_sigma_s = (1.0 - f) * sigma_a + sigma_s;

    // get distance to event
    const double dist_to_scatter = (total_sigma_s > 0.0) ?
      -log(rng.generate_random_number()) / total_sigma_s : 1.0e100;

    const double dist_to_boundary = cell->get_distance_to_boundary(
        pos_ptr[i], angle_ptr[i], surface_cross);
    const double dist_to_census = life_dx_ptr[i];

    // select minimum distance event
    const double dist_to_event = min(dist_to_scatter, min(dist_to_boundary, dist_to_census));

    // calculate energy absorbed by material, update photon and material energy
    // and update the path-length weighted tally for T_r
    const double absorbed_E = E_ptr[i] * (1.0 - exp(-sigma_a * f * dist_to_event));

    thread_absorbed_E += absorbed_E;
    // Avoid division by zero if sigma_a or f is zero
    if (sigma_a > 1.0e-100 && f > 1.0e-100)
        thread_track_E += absorbed_E / (sigma_a * f);

    E_ptr[i] = (E_ptr[i] - absorbed_E);

    // update position
    pos_ptr[i][0] += angle_ptr[i][0] * dist_to_event;
    pos_ptr[i][1] += angle_ptr[i][1] * dist_to_event;
    pos_ptr[i][2] += angle_ptr[i][2] * dist_to_event;
    life_dx_ptr[i] -= dist_to_event;

    // apply runtime reduction
    if (E_ptr[i] / E0_ptr[i] < Constants::cutoff_fraction) {
      thread_absorbed_E += E_ptr[i];
#ifdef USE_GPU // Use atomicAdd for GPU tallies
      // atomicAdd(&cell_tallies[local_cell_index].abs_E, thread_absorbed_E);
      // atomicAdd(&cell_tallies[local_cell_index].track_E, thread_track_E);
    warp_atomic_addh(&cell_tallies[local_cell_index].abs_E, local_cell_index, thread_absorbed_E); //+ (event_type == GPU_KILLED ? next_E : 0.0));
    warp_atomic_addh(&cell_tallies[local_cell_index].track_E, local_cell_index, thread_track_E);


#else
      cell_tallies[local_cell_index].accumulate_absorbed_E(thread_absorbed_E);
      cell_tallies[local_cell_index].accumulate_track_E(thread_track_E);
#endif
      active = false;
      descriptors_ptr[i] = static_cast<unsigned char>(Constants::KILLED);
    }
    // or apply event
    else {
      // EVENT TYPE: SCATTER
      if (dist_to_event == dist_to_scatter) {
        angle_ptr[i] =  get_uniform_angle(rng);
        if (rng.generate_random_number() > (sigma_s / total_sigma_s)) {
          group_ptr[i] = sample_emission_group(rng, *cell);
          // chance of more intensive scatter
          if (rng.generate_random_number() <= Constants::intensive_scatter_fraction) {
            auto group = group_ptr[i];
            // get a frequency (faux multigroup so just sample from wide spectrum)
            double freq = Constants::lower_frequency_bound + static_cast<double>(group)/static_cast<double>(BRANSON_N_GROUPS)*Constants::delta_frequency_bounds;
            auto angle = angle_ptr[i];
            auto new_energy_angle = intensive_scatter(cell->get_T_e(), freq, angle, rng);
            angle_ptr[i] = new_energy_angle.second;
          }
        }
        descriptors_ptr[i] = Constants::SCATTER;
      }
      // EVENT TYPE: BOUNDARY CROSS
      else if (dist_to_event == dist_to_boundary) {
        auto boundary_event = cell->get_bc(surface_cross);
        if (boundary_event == Constants::ELEMENT) {
#ifdef USE_GPU
          // atomicAdd(&cell_tallies[local_cell_index].abs_E, thread_absorbed_E);
          // atomicAdd(&cell_tallies[local_cell_index].track_E, thread_track_E);
          warp_atomic_addh(&cell_tallies[local_cell_index].abs_E, local_cell_index, thread_absorbed_E); //+ (event_type == GPU_KILLED ? next_E : 0.0));
          warp_atomic_addh(&cell_tallies[local_cell_index].track_E, local_cell_index, thread_track_E);
#else
          cell_tallies[local_cell_index].accumulate_absorbed_E(thread_absorbed_E);
          cell_tallies[local_cell_index].accumulate_track_E(thread_track_E);
#endif
          cell_ID_ptr[i] = cell->get_next_cell(surface_cross);
          local_cell_index =  cell_ID_ptr[i] - rank_cell_offset;
          cell = &cells[local_cell_index];
          descriptors_ptr[i] = static_cast<unsigned char>(Constants::BOUND);
          thread_absorbed_E = 0.0;
          thread_track_E = 0.0;
        } else if (boundary_event == Constants::PROCESSOR) {
          active = false;
          cell_ID_ptr[i] = cell->get_next_cell(surface_cross);
          descriptors_ptr[i] = static_cast<unsigned char>(Constants::PASS);
#ifdef USE_GPU
          // atomicAdd(&cell_tallies[local_cell_index].abs_E, thread_absorbed_E);
          // atomicAdd(&cell_tallies[local_cell_index].track_E, thread_track_E);
          warp_atomic_addh(&cell_tallies[local_cell_index].abs_E, local_cell_index, thread_absorbed_E); //+ (event_type == GPU_KILLED ? next_E : 0.0));
          warp_atomic_addh(&cell_tallies[local_cell_index].track_E, local_cell_index, thread_track_E);
#else
          cell_tallies[local_cell_index].accumulate_absorbed_E(thread_absorbed_E);
          cell_tallies[local_cell_index].accumulate_track_E(thread_track_E);
#endif
        } else if (boundary_event == Constants::VACUUM || boundary_event == Constants::SOURCE) {
          active = false;
          descriptors_ptr[i] = static_cast<unsigned char>(Constants::EXIT);
#ifdef USE_GPU
          // atomicAdd(&cell_tallies[local_cell_index].abs_E, thread_absorbed_E);
          // atomicAdd(&cell_tallies[local_cell_index].track_E, thread_track_E);
          warp_atomic_addh(&cell_tallies[local_cell_index].abs_E, local_cell_index, thread_absorbed_E); //+ (event_type == GPU_KILLED ? next_E : 0.0));
          warp_atomic_addh(&cell_tallies[local_cell_index].track_E, local_cell_index, thread_track_E);
#else
          cell_tallies[local_cell_index].accumulate_absorbed_E(thread_absorbed_E);
          cell_tallies[local_cell_index].accumulate_track_E(thread_track_E);
#endif
        } else { // REFLECT
          int reflect_angle = surface_cross/2; // X -> 0, Y->1, Z->2
          angle_ptr[i][reflect_angle] = -angle_ptr[i][reflect_angle];
          descriptors_ptr[i] = static_cast<unsigned char>(Constants::BOUND);
        }
      }
      // EVENT TYPE: REACH CENSUS
      else if (dist_to_event == dist_to_census) {
        active = false;
        descriptors_ptr[i] = static_cast<unsigned char>(Constants::CENSUS);
#ifdef USE_GPU
        // atomicAdd(&cell_tallies[local_cell_index].abs_E, thread_absorbed_E);
        // atomicAdd(&cell_tallies[local_cell_index].track_E, thread_track_E);
        warp_atomic_addh(&cell_tallies[local_cell_index].abs_E, local_cell_index, thread_absorbed_E); //+ (event_type == GPU_KILLED ? next_E : 0.0));
        warp_atomic_addh(&cell_tallies[local_cell_index].track_E, local_cell_index, thread_track_E);
#else
        cell_tallies[local_cell_index].accumulate_absorbed_E(thread_absorbed_E);
        cell_tallies[local_cell_index].accumulate_track_E(thread_track_E);
#endif
      }
    } // end event loop
  } // end while alive
}

//----------------------------------------------------------------------------//
// CPU Transport Functions (Host) - Overloads for AoS and SoA
//----------------------------------------------------------------------------//

//! Transport photons using history-based method on CPU (AoS version)
void history_cpu_transport_photons(const uint32_t rank_cell_offset,
    Photon_Data<std::vector<Photon>> photon_data, size_t batch_start, size_t batch_end, GPU_Setup<std::vector<Photon>> &gpu_setup, int n_omp_threads) {

  Photon *photons = photon_data.h_photon_ptr;
  auto cpu_cells_ptr{gpu_setup.get_host_cells_ptr()}; // CPU data here
  auto cpu_tallies_ptr{gpu_setup.get_device_cell_tallies_ptr()}; // CPU data here
  const auto n_cells = gpu_setup.get_n_cells();

#ifdef USE_OPENMP
  std::vector<std::vector<Cell_Tally>> thread_tallies(n_omp_threads);
#pragma omp parallel num_threads(n_omp_threads)
  {
    int tid = omp_get_thread_num();
    thread_tallies[tid].resize(n_cells); // Initialize with zeros
    auto thread_tally_ptr = thread_tallies[tid].data();

#pragma omp for schedule(guided)
    for (size_t i=batch_start; i<batch_end; ++i) {
      // Call the CPU version
      transport_photon_history_aos_cpu(rank_cell_offset, photons[i], cpu_cells_ptr, thread_tally_ptr);
    }
  } // end parallel region

  // reduce tallies if using openmp
  for(size_t cell=0; cell<n_cells; ++cell) {
    for(int thread =0; thread<n_omp_threads;++thread)
      cpu_tallies_ptr[cell].merge_in_tally(thread_tallies[thread][cell]);
  }
#else
  // normal serial version
  for (size_t i=batch_start; i<batch_end; ++i) {
    // Call the CPU version
    transport_photon_history_aos_cpu(rank_cell_offset, photons[i], cpu_cells_ptr, cpu_tallies_ptr);
  }
#endif
}

//! Transport photons using history-based method on CPU (SoA version)
void history_cpu_transport_photons(const uint32_t rank_cell_offset,
    Photon_Data<PhotonArray> &photon_data, size_t batch_start, size_t batch_end, GPU_Setup<PhotonArray> &gpu_setup,
    int n_omp_threads) {

  auto cpu_cells_ptr{gpu_setup.get_host_cells_ptr()}; // CPU data here
  auto cpu_tallies_ptr{gpu_setup.get_device_cell_tallies_ptr()}; // CPU data here
  const auto n_cells = gpu_setup.get_n_cells();

#ifdef USE_OPENMP
  std::vector<std::vector<Cell_Tally>> thread_tallies(n_omp_threads);
#pragma omp parallel num_threads(n_omp_threads)
  {
    int tid = omp_get_thread_num();
    thread_tallies[tid].resize(n_cells); // Initialize with zeros
    auto thread_tally_ptr = thread_tallies[tid].data();

#pragma omp for schedule(guided)
    for (size_t i=batch_start; i<batch_end; ++i) {
      // Call the CPU version
      transport_photon_history_soa_cpu(rank_cell_offset,
        photon_data.h_cell_ID_ptr[i],
        photon_data.h_group_ptr[i],
        photon_data.h_descriptors_ptr[i],
        photon_data.h_pos_ptr[i],
        photon_data.h_angle_ptr[i],
        photon_data.h_E_ptr[i],
        photon_data.h_E0_ptr[i],
        photon_data.h_life_dx_ptr[i],
        photon_data.h_RNG_ptr[i],
        cpu_cells_ptr, thread_tally_ptr);
    }
  } // end parallel region

  // reduce tallies if using openmp
  for(size_t cell=0; cell<n_cells; ++cell) {
    for(int thread =0; thread<n_omp_threads;++thread)
      cpu_tallies_ptr[cell].merge_in_tally(thread_tallies[thread][cell]);
  }
#else
  // normal serial version
  for (size_t i=batch_start; i<batch_end; ++i) {
    // Call the CPU version
      transport_photon_history_soa_cpu(rank_cell_offset,
        photon_data.h_cell_ID_ptr[i],
        photon_data.h_group_ptr[i],
        photon_data.h_descriptors_ptr[i],
        photon_data.h_pos_ptr[i],
        photon_data.h_angle_ptr[i],
        photon_data.h_E_ptr[i],
        photon_data.h_E0_ptr[i],
        photon_data.h_life_dx_ptr[i],
        photon_data.h_RNG_ptr[i],
        cpu_cells_ptr, cpu_tallies_ptr);
  }
#endif
}

//----------------------------------------------------------------------------//
// GPU Kernels
//----------------------------------------------------------------------------//

//! GPU kernel for history-based transport (AoS version)
GPU_KERNEL
void gpu_history_transport_aos_kernel(const uint32_t rank_cell_offset,
    Photon *all_photons, const Cell *cells, Cell_Tally *cell_tallies, const uint32_t n_photons) {
#ifdef USE_GPU
  int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n_photons) {
    // Call the GPU device version
    transport_photon_history_aos_gpu(rank_cell_offset, all_photons[i], cells, cell_tallies);
  }
#endif
}

//! GPU kernel for history-based transport (SoA version)
GPU_KERNEL
void gpu_history_transport_soa_kernel(const uint32_t rank_cell_offset,
    uint32_t* cell_ID_ptr, uint32_t* group_ptr, unsigned char* descriptors_ptr,
    std::array<double, 3>* pos_ptr, std::array<double, 3>* angle_ptr,
    double* E_ptr, double* E0_ptr, double* life_dx_ptr, RNG* rng_ptr,
    const Cell *cells, Cell_Tally *cell_tallies, const uint32_t n_photons) {
#ifdef USE_GPU
  int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n_photons) {
    // Call the GPU device version
    transport_photon_history_soa_gpu(rank_cell_offset, i,
        cell_ID_ptr, group_ptr, descriptors_ptr,
        pos_ptr, angle_ptr, E_ptr, E0_ptr, life_dx_ptr, rng_ptr,
        cells, cell_tallies);
  }
#endif
}

//----------------------------------------------------------------------------//
// GPU Transport Functions (Host)
//----------------------------------------------------------------------------//

//! Transport photons using history-based method on GPU (AoS version)
template <typename Census_T>
void gpu_transport_photons(const uint32_t rank_cell_offset,
    Photon_Data<Census_T> &photon_data, size_t batch_start, size_t batch_end, GPU_Setup<Census_T> &gpu_setup) {

#ifdef USE_GPU
  Timer t_transport;
  std::string timer_name;
  std::string kernel_name;

  if constexpr (std::is_same_v<Census_T, std::vector<Photon>>) {
    timer_name ="aos_gpu_transport_photons";
  }
  else {
    timer_name ="soa_gpu_transport_photons";
  }
  t_transport.start_timer(timer_name);


  size_t n_batch_photons = batch_end - batch_start;
  if (n_batch_photons == 0) return; // No work to do

  //device_debug_print(n_batch_photons, timer_name);

  // Kernel settings
  int n_threads = Constants::n_threads_per_block;
  int n_blocks = (n_batch_photons + n_threads - 1) / n_threads;

  if constexpr (std::is_same_v<Census_T, std::vector<Photon>>) {
    // Launch kernel
    gpu_history_transport_aos_kernel<<<n_blocks, n_threads>>>(
        rank_cell_offset, photon_data.d_photon_ptr + batch_start, gpu_setup.get_device_cells_ptr(), gpu_setup.get_device_cell_tallies_ptr(), n_batch_photons);
  }
  else {
    // set pointers for this batch
    uint32_t *d_cell_ID = photon_data.d_cell_ID_ptr + batch_start;
    uint32_t *d_group = photon_data.d_group_ptr + batch_start;
    unsigned char *d_descriptors = photon_data.d_descriptors_ptr + batch_start;
    std::array<double, 3> *d_pos = photon_data.d_pos_ptr + batch_start;
    std::array<double, 3> *d_angle = photon_data.d_angle_ptr + batch_start;
    double *d_E = photon_data.d_E_ptr + batch_start;
    double *d_E0 = photon_data.d_E0_ptr + batch_start;
    double *d_life_dx = photon_data.d_life_dx_ptr + batch_start;
    RNG *d_RNG = photon_data.d_RNG_ptr + batch_start;

    gpu_history_transport_soa_kernel<<<n_blocks, n_threads>>>(
        rank_cell_offset, d_cell_ID, d_group, d_descriptors, d_pos, d_angle,
        d_E, d_E0, d_life_dx, d_RNG,
        gpu_setup.get_device_cells_ptr(), gpu_setup.get_device_cell_tallies_ptr(), n_batch_photons);
  }

  auto kernel_err = cudaGetLastError();
  Insist(!kernel_err, "CUDA/HIP error in history kernel launch");
  auto sync_err = cudaDeviceSynchronize();
  Insist(!sync_err, "CUDA/HIP error synchronizing after history kernel");

  t_transport.stop_timer(timer_name);
#else
  // Provide a fallback or error if GPU is not enabled but this function is called
  std::cerr << "Warning: GPU transport called but CUDA/HIP is not enabled. Running on CPU." << std::endl;
  // Find number of threads available
  int n_omp_threads = 1;
#ifdef USE_OPENMP
  #pragma omp parallel
  { n_omp_threads = omp_get_num_threads(); }
#endif
  // Need the host cells vector if running on CPU
  std::vector<Cell> host_cells; // Placeholder - needs actual data if fallback is used
  history_cpu_transport_photons(rank_cell_offset, photon_data, batch_start, batch_end, gpu_setup, n_omp_threads);
#endif
}

#endif // def history_based_transport_h_
//----------------------------------------------------------------------------//
// end of history_based_transport.h
//----------------------------------------------------------------------------//
