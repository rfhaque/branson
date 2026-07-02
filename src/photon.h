//----------------------------------*-C++-*-----------------------------------//
/*!
 * \file   photon.h
 * \author Alex Long
 * \date   July 18 2014
 * \brief  Holds values and functions needed for transporting photon
 * \note   Copyright (C) 2017 Los Alamos National Security, LLC.
 *         All rights reserved
 */
//----------------------------------------------------------------------------//

#ifndef photon_h_
#define photon_h_

#include <cmath>
#include <iostream>
#include <vector>
#include <array>

#include "constants.h"
#include "config.h"
#include "RNG.h"

//==============================================================================
/*!
 * \class Photon
 * \brief Contains position, direction, cell ID and energy for transport.
 *
 * Holds all of the internal state of a photon and provides functions for
 * sorting photons based on census and global cell ID.
 */
//==============================================================================
class Photon {
public:
  //! Constructor
  Photon() {}

  //! Constructor
  GPU_HOST_DEVICE Photon(const uint32_t cell_ID, uint32_t group, uint32_t source_type, Constants::event_type descriptor, std::array<double,3> pos,  std::array<double,3> angle, double E0, double life_dx, RNG rng) :
    m_cell_ID(cell_ID), m_group(group), m_source_type(source_type), descriptors{descriptor, 0, 0,0}, m_pos(pos), m_angle(angle), m_E(E0), m_E0(E0), m_life_dx(life_dx), m_rng(rng)
  {}

  GPU_HOST_DEVICE Photon(const uint32_t cell_ID, uint32_t group, uint32_t source_type, Constants::event_type descriptor, std::array<double,3> pos,  std::array<double,3> angle, double E, double E0, double life_dx, RNG rng) :
    m_cell_ID(cell_ID), m_group(group), m_source_type(source_type), descriptors{descriptor, 0, 0,0}, m_pos(pos), m_angle(angle), m_E(E), m_E0(E0), m_life_dx(life_dx), m_rng(rng)
  {}


  //! Destructor
  //~Photon() {}

  //--------------------------------------------------------------------------//
  // const functions                                                          //
  //--------------------------------------------------------------------------//

  //! Check to see if photon energy weight is below cutoff fraction
  GPU_HOST_DEVICE
  bool below_cutoff(const double cutoff_fraction) const {
    // Handle potential division by zero or very small E0
    if (m_E0 <= 1.0e-100) return true; // Consider it below cutoff if initial energy is effectively zero
    return (m_E / m_E0 < cutoff_fraction);
  }

  GPU_HOST_DEVICE
  inline double get_fraction() const {
      // Handle potential division by zero or very small E0
      if (m_E0 <= 1.0e-100) return 0.0;
      return m_E/ m_E0;
  }

  //! Return global cell ID
  GPU_HOST_DEVICE
  inline uint32_t get_cell(void) const { return m_cell_ID; }

  //! Return photon group
  GPU_HOST_DEVICE
  inline uint32_t get_group(void) const { return m_group; }

  //! Return a constant pointer to the start of the particle position array
  GPU_HOST_DEVICE
  inline std::array<double,3> get_position(void) const { return m_pos; }

  //! Return a constant pointer to the start of the particle direction array
  GPU_HOST_DEVICE
  inline std::array<double,3> get_angle(void) const { return m_angle; }

  //! Get the particle's energy-weight
  GPU_HOST_DEVICE
  inline double get_E(void) const { return m_E; }

  //! Get the partice's initial energy-weight
  GPU_HOST_DEVICE
  inline double get_E0(void) const { return m_E0; }

  //! Get the distance to census (cm)
  GPU_HOST_DEVICE
  inline double get_distance_remaining(void) const { return m_life_dx; }

  //! Print particle information
  void print_info(const uint32_t &rank) const {
    using std::cout;
    using std::endl;
    cout << "----Photon Info----\n";
    cout << rank << " " << m_pos[0] << " " << m_pos[1] << " " << m_pos[2]
         << endl;
    cout << "angle: " << m_angle[0] << " " << m_angle[1] << " " << m_angle[2]
         << endl;
    cout << "Energy: " << m_E << " , Initial energy: " << m_E0 << endl;
    cout << "Cell ID: " << m_cell_ID << endl;
  }

  //! Override great than operator to sort
  bool operator<(const Photon &compare) const {
    return m_cell_ID < compare.get_cell();
  }

  GPU_HOST_DEVICE
  Constants::event_type get_descriptor() const {return static_cast<Constants::event_type>(descriptors[0]);}

  //--------------------------------------------------------------------------//
  // non-const functions                                                      //
  //--------------------------------------------------------------------------//

  //! Update particle position by moving it a distance
  GPU_HOST_DEVICE
  inline void move(const double distance) {
    m_pos[0] += m_angle[0] * distance;
    m_pos[1] += m_angle[1] * distance;
    m_pos[2] += m_angle[2] * distance;
    m_life_dx -= distance;
  }

  inline uint32_t get_source_type() const {return m_source_type;}
  inline void set_source_type(uint32_t source_type_in) {m_source_type = source_type_in;}

  //! Set the global cell ID
  GPU_HOST_DEVICE
  inline void set_cell(const uint32_t new_cell) { m_cell_ID = new_cell; }

  //! Set the group of the photon
  GPU_HOST_DEVICE
  inline void set_group(const uint32_t new_group) { m_group = new_group; }

  //! Set the initial energy-weight
  inline void set_E0(const double E) {
    m_E0 = E;
    m_E = E;
  }

  //! Set the current energy-weight
  GPU_HOST_DEVICE
  inline void set_E(const double E) { m_E = E; }

  //! Set the distance to census (cm)
  GPU_HOST_DEVICE
  inline void set_distance_to_census(const double dist_remain) { m_life_dx = dist_remain; }

  //! Set the angle of the photon
  GPU_HOST_DEVICE
  inline void set_angle(const std::array<double,3> &new_angle) { m_angle = new_angle;}

  //! Set the spatial position of the photon
  GPU_HOST_DEVICE
  inline void set_position(const std::array<double, 3> &new_pos) { m_pos = new_pos;}

  //! Reflect a photon about a plane aligned with the X, Y, or Z axes
  GPU_HOST_DEVICE
  inline void reflect(const uint32_t surface_cross) {
    // reflect the photon over the surface it was crossing
    int reflect_angle = surface_cross/2; // X -> 0, Y->1, Z->2
    m_angle[reflect_angle] = -m_angle[reflect_angle];
  }

  GPU_HOST_DEVICE
  void set_descriptor(const Constants::event_type descriptor) { descriptors[0] = static_cast<unsigned char>(descriptor);}

  GPU_HOST_DEVICE
  RNG get_rng() const {return m_rng;}

  GPU_HOST_DEVICE
  RNG &get_rng() {return m_rng;}

  void set_rng(const RNG &rng) { m_rng = rng;}

  //--------------------------------------------------------------------------//
  // member data                                                              //
  //--------------------------------------------------------------------------//
private:
  uint32_t m_cell_ID; //!< Cell ID
  uint32_t m_group;     //!< Group of photon
  uint32_t m_source_type; //!< CENSUS, EMISSION, or SOURCE
  std::array<unsigned char, 4> descriptors; //!< Only using one, but it fills out the padding
  std::array<double,3> m_pos;    //!< photon position
  std::array<double,3> m_angle;  //!< photon angle array
  double m_E;         //!< current photon energy
  double m_E0;        //!< photon energy at creation
  double m_life_dx;   //!< Distance remaining this time step
  RNG m_rng;          //!< Member RNG

};

#endif // photon_h_
