//----------------------------------*-C++-*-----------------------------------//
/*!
 * \file   photon_array.h
 * \author Alex Long
 * \date   July 18 2014, Modified for SoA
 * \brief  Holds values and functions needed for transporting photons (SoA)
 * \note   Copyright (C) 2017 Los Alamos National Security, LLC.
 *         All rights reserved
 */
//----------------------------------------------------------------------------//

#ifndef photon_array_h_
#define photon_array_h_

#include <cmath>
#include <iostream>
#include <vector>
#include <array>
#include <stdexcept> // For std::out_of_range

#include "constants.h"
#include "config.h"
#include "RNG.h"
#include "photon.h" // Include Photon definition for push_back

// Structure of arrays to store photon attributes
class PhotonArray {
public:
  std::vector<uint32_t> cell_ID;
  std::vector<uint32_t> group;
  std::vector<uint32_t> source_type; // CENSUS, EMISSION, SOURCE (as uint32_t)
  std::vector<unsigned char> descriptors; // Event type (as uchar)
  std::vector<std::array<double, 3>> pos;
  std::vector<std::array<double, 3>> angle;
  std::vector<double> E;
  std::vector<double> E0;
  std::vector<double> life_dx;
  std::vector<RNG> rng;

  // Add a photon from a Photon object (AoS -> SoA)
  void push_back(const Photon& photon) {
    cell_ID.push_back(photon.get_cell());
    group.push_back(photon.get_group());
    source_type.push_back(photon.get_source_type());
    // Ensure descriptor is stored as unsigned char
    descriptors.push_back(static_cast<unsigned char>(photon.get_descriptor()));
    pos.push_back(photon.get_position());
    angle.push_back(photon.get_angle());
    E.push_back(photon.get_E());
    E0.push_back(photon.get_E0());
    life_dx.push_back(photon.get_distance_remaining());
    rng.push_back(photon.get_rng()); // Assumes RNG is copyable
  }

  // Resize all internal vectors
  void resize(size_t size)
  {
    cell_ID.resize(size);
    group.resize(size);
    source_type.resize(size);
    descriptors.resize(size);
    pos.resize(size);
    angle.resize(size);
    E.resize(size);
    E0.resize(size);
    life_dx.resize(size);
    rng.resize(size);
  }

   // Reserve capacity for all internal vectors
   void reserve(size_t capacity)
   {
     cell_ID.reserve(capacity);
     group.reserve(capacity);
     source_type.reserve(capacity);
     descriptors.reserve(capacity);
     pos.reserve(capacity);
     angle.reserve(capacity);
     E.reserve(capacity);
     E0.reserve(capacity);
     life_dx.reserve(capacity);
     rng.reserve(capacity);
   }

   // Clear all internal vectors
   void clear()
   {
       cell_ID.clear();
       group.clear();
       source_type.clear();
       descriptors.clear();
       pos.clear();
       angle.clear();
       E.clear();
       E0.clear();
       life_dx.clear();
       rng.clear();
   }


  // Add a photon from another PhotonArray at a specific index
  void add_photon(const PhotonArray &source, size_t index) {
     // Check bounds
     if (index >= source.size()) {
         throw std::out_of_range("Index out of range in PhotonArray::add_photon");
     }
    // No need to get size, push_back handles appending
    cell_ID.push_back(source.cell_ID[index]);
    group.push_back(source.group[index]);
    source_type.push_back(source.source_type[index]);
    descriptors.push_back(source.descriptors[index]);
    pos.push_back(source.pos[index]);
    angle.push_back(source.angle[index]);
    E.push_back(source.E[index]);
    E0.push_back(source.E0[index]);
    life_dx.push_back(source.life_dx[index]);
    rng.push_back(source.rng[index]); // Assumes RNG is copyable
  }

  // Check if the array is empty (based on one representative vector)
  bool empty() const {return cell_ID.empty();}

  // Get the number of photons (based on one representative vector)
  size_t size() const {return cell_ID.size();}

  // Get a Photon object representing the data at index i (SoA -> AoS)
  // Note: This creates a temporary Photon object (copy)
  Photon get_photon(size_t i) const {
     // Check bounds
     if (i >= size()) {
         throw std::out_of_range("Index out of range in PhotonArray::get_photon");
     }
    Photon return_photon;
    // Use setters to populate the Photon object
    return_photon.set_cell(cell_ID[i]);
    return_photon.set_group(group[i]);
    return_photon.set_source_type(source_type[i]);
    return_photon.set_descriptor(static_cast<Constants::event_type>(descriptors[i]));
    return_photon.set_position(pos[i]);
    return_photon.set_angle(angle[i]);
    // Need to set E0 first if set_E depends on it, or use a different setter
    return_photon.set_E0(E0[i]); // Sets both E0 and E initially
    return_photon.set_E(E[i]);   // Correct the current E
    return_photon.set_distance_to_census(life_dx[i]);
    return_photon.set_rng(rng[i]); // Assumes RNG is copyable
    return return_photon;
  }

  // Insert photons from another PhotonArray at the end
  void insert(const PhotonArray &photons_to_add) {
     size_t original_size = size();
     size_t num_to_add = photons_to_add.size();
     if (num_to_add == 0) return;

     size_t new_size = original_size + num_to_add;
     resize(new_size); // Resize all vectors first

    // Copy data element by element
    // Consider using std::copy for potentially better performance if vectors are contiguous
    for (size_t i = 0; i < num_to_add; ++i) {
      cell_ID[original_size+i] = photons_to_add.cell_ID[i];
      group[original_size+i] = photons_to_add.group[i];
      source_type[original_size+i] = photons_to_add.source_type[i];
      descriptors[original_size+i] = photons_to_add.descriptors[i];
      pos[original_size+i] = photons_to_add.pos[i];
      angle[original_size+i] = photons_to_add.angle[i];
      E[original_size+i] = photons_to_add.E[i];
      E0[original_size+i] = photons_to_add.E0[i];
      life_dx[original_size+i] = photons_to_add.life_dx[i];
      rng[original_size+i] = photons_to_add.rng[i]; // Assumes RNG is copyable
    }
  }

  // Create a *copy* of a sub-range of this PhotonArray
  PhotonArray get_sub_batch(const size_t batch_start, const size_t batch_end) const {
    // Validate range
    if (batch_start >= size() || batch_end > size() || batch_start >= batch_end) {
        // Return empty or throw error
        // Returning empty is safer if this can happen (e.g., last batch)
        if (batch_start == batch_end && batch_start <= size()) return PhotonArray(); // Allow empty batch
        throw std::out_of_range("Invalid range in PhotonArray::get_sub_batch");
    }

    size_t batch_size = batch_end - batch_start;
    PhotonArray batch_photons;
    batch_photons.resize(batch_size); // Pre-allocate size

    // Copy data for the sub-range
    // Using std::copy for potentially better performance
    std::copy(cell_ID.begin() + batch_start, cell_ID.begin() + batch_end, batch_photons.cell_ID.begin());
    std::copy(group.begin() + batch_start, group.begin() + batch_end, batch_photons.group.begin());
    std::copy(source_type.begin() + batch_start, source_type.begin() + batch_end, batch_photons.source_type.begin());
    std::copy(descriptors.begin() + batch_start, descriptors.begin() + batch_end, batch_photons.descriptors.begin());
    std::copy(pos.begin() + batch_start, pos.begin() + batch_end, batch_photons.pos.begin());
    std::copy(angle.begin() + batch_start, angle.begin() + batch_end, batch_photons.angle.begin());
    std::copy(E.begin() + batch_start, E.begin() + batch_end, batch_photons.E.begin());
    std::copy(E0.begin() + batch_start, E0.begin() + batch_end, batch_photons.E0.begin());
    std::copy(life_dx.begin() + batch_start, life_dx.begin() + batch_end, batch_photons.life_dx.begin());
    std::copy(rng.begin() + batch_start, rng.begin() + batch_end, batch_photons.rng.begin()); // Assumes RNG is copyable

    return batch_photons;
  }

   // Update this PhotonArray from a (potentially modified) sub-batch copy
   void update_from_sub_batch(const PhotonArray& sub_batch, const size_t batch_start) {
       size_t sub_batch_size = sub_batch.size();
       if (sub_batch_size == 0) return;

       size_t batch_end = batch_start + sub_batch_size;

       // Validate range
       if (batch_start >= size() || batch_end > size()) {
           throw std::out_of_range("Invalid range in PhotonArray::update_from_sub_batch");
       }

       // Copy data back from the sub-batch into the original array
       std::copy(sub_batch.cell_ID.begin(), sub_batch.cell_ID.end(), cell_ID.begin() + batch_start);
       std::copy(sub_batch.group.begin(), sub_batch.group.end(), group.begin() + batch_start);
       std::copy(sub_batch.source_type.begin(), sub_batch.source_type.end(), source_type.begin() + batch_start);
       std::copy(sub_batch.descriptors.begin(), sub_batch.descriptors.end(), descriptors.begin() + batch_start);
       std::copy(sub_batch.pos.begin(), sub_batch.pos.end(), pos.begin() + batch_start);
       std::copy(sub_batch.angle.begin(), sub_batch.angle.end(), angle.begin() + batch_start);
       std::copy(sub_batch.E.begin(), sub_batch.E.end(), E.begin() + batch_start);
       std::copy(sub_batch.E0.begin(), sub_batch.E0.end(), E0.begin() + batch_start); // Usually E0 doesn't change, but copy for completeness
       std::copy(sub_batch.life_dx.begin(), sub_batch.life_dx.end(), life_dx.begin() + batch_start);
       std::copy(sub_batch.rng.begin(), sub_batch.rng.end(), rng.begin() + batch_start); // Copy back RNG state
   }

  // remove particles that did not reach census with simple partition sort
  void remove_inactive_particles() {
    //std::cout<<"Removing inactive particles, pre size: "<<descriptors.size()<<std::endl;
    const size_t new_census_size =  std::count_if(descriptors.begin(), descriptors.end(), [] (const auto idesc) {return idesc == Constants::CENSUS;});
    //std::cout<<"New size should be: "<<new_census_size<<std::endl;
    size_t i = 0;
    size_t j = descriptors.size()- 1;
    while (i < j) {
      while (i < j && descriptors[i] == Constants::CENSUS) ++i; // find not CENSUS from the left
      while (i < j && descriptors[j] != Constants::CENSUS) --j; // find CENSUS from the right
      if (i < j) {
        std::swap(descriptors[i], descriptors[j]);
        std::swap(cell_ID[i], cell_ID[j]);
        std::swap(group[i], group[j]);
        std::swap(source_type[i], source_type[j]);
        std::swap(pos[i], pos[j]);
        std::swap(E[i], E[j]);
        std::swap(E0[i], E0[j]);
        std::swap(life_dx[i], life_dx[j]);
        std::swap(rng[i], rng[j]);
        i++;
        j--;
      }
    }
    descriptors.erase(descriptors.begin() + new_census_size, descriptors.end());
    cell_ID.erase(cell_ID.begin() + new_census_size, cell_ID.end());
    group.erase(group.begin() + new_census_size, group.end());
    source_type.erase(source_type.begin() + new_census_size, source_type.end());
    pos.erase(pos.begin() + new_census_size, pos.end());
    angle.erase(angle.begin() + new_census_size, angle.end());
    E.erase(E.begin() + new_census_size, E.end());
    E0.erase(E0.begin() + new_census_size, E0.end());
    life_dx.erase(life_dx.begin() + new_census_size, life_dx.end());
    rng.erase(rng.begin() + new_census_size, rng.end());
    //std::cout<<"Double check new pos size: "<<pos.size()<<std::endl;
  }

};

// Remove the old Photon class definition that wrapped PhotonArray

#endif // photon_array_h_
