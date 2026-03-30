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
#include "sampling_functions.h"


template <typename Census_T>
std::tuple<uint64_t, double, double> post_process_photons(const double next_dt, Census_T &all_photons, const Mesh &mesh, std::vector<std::vector<Photon>> &send_list);

template <>
std::tuple<uint64_t, double, double> post_process_photons<std::vector<Photon>>(const double next_dt, std::vector<Photon> &all_photons, const Mesh &mesh, std::vector<std::vector<Photon>> &send_list) {
  uint64_t n_complete = 0;
  double census_E{0.0};
  double exit_E{0.0};
  const auto adjacent_procs = mesh.get_proc_adjacency_list();
  for (auto &phtn : all_photons) {
    auto descriptor{phtn.get_descriptor()};
    switch (descriptor) {
    case Constants::event_type::KILLED:
      // note: for now killed particles go into the material so separate conservation issues here
      n_complete++;
      break;
    case Constants::event_type::EXIT:
      exit_E+=phtn.get_E();
      n_complete++;
      break;
    case Constants::event_type::CENSUS:
      phtn.set_distance_to_census(Constants::c*next_dt);
      phtn.set_source_type(0); // 0 is census
      census_E+=phtn.get_E();
      n_complete++;
      break;
    case Constants::PASS:
      auto send_rank = mesh.get_rank(phtn.get_cell());
      int i_b = adjacent_procs.at(send_rank);
      send_list[i_b].push_back(phtn);
      break;
    } //switch(descriptor)
  } // phtn : all_photons
  return {n_complete, exit_E, census_E};
}

template <>
std::tuple<uint64_t, double, double>  post_process_photons<PhotonArray>(const double next_dt, PhotonArray &all_photons, const Mesh &mesh, std::vector<std::vector<Photon>> &send_list) {
  uint64_t n_complete = 0.0;
  double census_E{0.0};
  double exit_E{0.0};
  size_t census_count = 0;
  const auto adjacent_procs = mesh.get_proc_adjacency_list();
  for (size_t i=0; i<all_photons.size();++i) {
    auto descriptor{all_photons.descriptors[i]};
    switch (descriptor) {
    case Constants::event_type::KILLED:
      // note: for now killed particles go into the material so separate conservation issues here
      n_complete++;
      break;
    case Constants::event_type::EXIT:
      exit_E+=all_photons.E[i];
      n_complete++;
      break;
    case Constants::event_type::CENSUS:
      all_photons.life_dx[i] = Constants::c*next_dt;
      all_photons.source_type[i] = 0; // 0 is census
      census_E+=all_photons.E[i];
      census_count++;
      n_complete++;
      break;
    case Constants::PASS:
      auto send_rank = mesh.get_rank(all_photons.cell_ID[i]);
      int i_b = adjacent_procs.at(send_rank);
      send_list[i_b].push_back( all_photons.get_photon(i));
      break;
    } // switch(descriptor)
  }
  return {n_complete, exit_E, census_E};
}

#endif // def transport_photon_h_
//----------------------------------------------------------------------------//
// end of transport_photon.h
//----------------------------------------------------------------------------//
