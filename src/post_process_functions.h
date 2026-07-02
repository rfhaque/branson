//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   particle_pass_transport.h
 * \author Alex Long
 * \date   December 1 2015
 * \brief  IMC transport with particle passing method
 * \note   Copyright (C) 2017 Los Alamos National Security, LLC.
 *         All rights reserved
 */
//---------------------------------------------------------------------------//

#ifndef transport_photon_h_
#define transport_photon_h_

#include <algorithm>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "config.h"
#include "RNG.h"
#include "cell_tally.h"
#include "constants.h"
#include "photon.h"
#include "photon_data.h"
#include "sampling_functions.h"


template <typename Census_T>
std::tuple<uint64_t, double, double> post_process_photons(const double next_dt, Photon_Data<Census_T > &photon_data, size_t batch_start, size_t batch_end, const Mesh &mesh, std::vector<std::vector<Photon>> &send_list);

template <>
std::tuple<uint64_t, double, double> post_process_photons<std::vector<Photon>>(const double next_dt, Photon_Data<std::vector<Photon>> &photon_data, size_t batch_start, size_t batch_end, const Mesh &mesh, std::vector<std::vector<Photon>> &send_list) {
  Photon *all_photons = photon_data.h_photon_ptr;
  uint64_t n_complete = 0;
  double census_E{0.0};
  double exit_E{0.0};
  const auto adjacent_procs = mesh.get_proc_adjacency_list();
  for (size_t i =batch_start; i<batch_end; ++i) {
    auto descriptor{all_photons[i].get_descriptor()};
    switch (descriptor) {
    case Constants::event_type::KILLED:
      // note: for now killed particles go into the material so separate conservation issues here
      n_complete++;
      break;
    case Constants::event_type::EXIT:
      exit_E+=all_photons[i].get_E();
      n_complete++;
      break;
    case Constants::event_type::CENSUS:
      all_photons[i].set_distance_to_census(Constants::c*next_dt);
      all_photons[i].set_source_type(0); // 0 is census
      census_E+=all_photons[i].get_E();
      n_complete++;
      break;
    case Constants::PASS:
      auto send_rank = mesh.get_rank(all_photons[i].get_cell());
      int i_b = adjacent_procs.at(send_rank);
      send_list[i_b].push_back(all_photons[i]);
      break;
    } //switch(descriptor)
  }
  return {n_complete, exit_E, census_E};
}

template <>
std::tuple<uint64_t, double, double>  post_process_photons<PhotonArray>(const double next_dt, Photon_Data<PhotonArray> &photon_data, size_t batch_start, size_t batch_end,  const Mesh &mesh, std::vector<std::vector<Photon>> &send_list) {
  uint64_t n_complete = 0.0;
  double census_E{0.0};
  double exit_E{0.0};
  size_t census_count = 0;
  const auto adjacent_procs = mesh.get_proc_adjacency_list();
  for (size_t i=batch_start; i<batch_end;++i) {
    auto descriptor{photon_data.h_descriptors_ptr[i]};
    switch (descriptor) {
    case Constants::event_type::KILLED:
      // note: for now killed particles go into the material so separate conservation issues here
      n_complete++;
      break;
    case Constants::event_type::EXIT:
      exit_E+=photon_data.h_E_ptr[i];
      n_complete++;
      break;
    case Constants::event_type::CENSUS:
      photon_data.h_life_dx_ptr[i] = Constants::c*next_dt;
      photon_data.h_source_type_ptr[i] = 0; // 0 is census
      census_E+=photon_data.h_E_ptr[i];
      census_count++;
      n_complete++;
      break;
    case Constants::PASS:
      auto send_rank = mesh.get_rank(photon_data.h_cell_ID_ptr[i]);
      int i_b = adjacent_procs.at(send_rank);
      send_list[i_b].push_back(Photon(
        photon_data.h_cell_ID_ptr[i],
        photon_data.h_group_ptr[i],
        photon_data.h_source_type_ptr[i],
        static_cast<Constants::event_type>(photon_data.h_descriptors_ptr[i]),
        photon_data.h_pos_ptr[i],
        photon_data.h_angle_ptr[i],
        photon_data.h_E_ptr[i],
        photon_data.h_E0_ptr[i],
        photon_data.h_life_dx_ptr[i],
        photon_data.h_RNG_ptr[i]));
      break;
    } // switch(descriptor)
  }
  return {n_complete, exit_E, census_E};
}

#endif // def transport_photon_h_
//----------------------------------------------------------------------------//
// end of transport_photon.h
//----------------------------------------------------------------------------//
