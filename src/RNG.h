//--------------------------------------------*-C++-*---------------------------------------------//
/*!
 * \file   rng/Counter_RNG.hh
 * \author Peter Ahrens
 * \date   Fri Aug 3 16:53:23 2012
 * \brief  Declaration of class Counter_RNG.
 * \note   Copyright (C) 2012-2024 Triad National Security, LLC., All rights reserved. */
//------------------------------------------------------------------------------------------------//

#ifndef Counter_RNG_hh
#define Counter_RNG_hh

#include "config.h"

#ifdef _MSC_FULL_VER
// - 4267: Conversion from size_t to unsigned int, possible loss of data.
// - 4521: Engines have multiple copy constructors, quite legal C++, disable MSVC complaint.
// - 4244: possible loss of data when converting between int types.
// - 4204: nonstandard extension used - non-constant aggregate initializer
// - 4127: conditional expression is constant
#pragma warning(push)
#pragma warning(disable : 4267 4521 4244 4204 4127)
#endif

#if defined(__ICC)
// Suppress Intel's "unrecognized preprocessor directive" warning, triggered by use of #warning in
// Random123/features/sse.h.
#pragma warning disable 11
#endif

#if defined(__GNUC__) && !defined(__clang__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wexpansion-to-defined"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif

#if defined(__clang__) && !defined(__ibmxl__)
// Also use these for defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wexpansion-to-defined"
#pragma clang diagnostic ignored "-Wreserved-id-macro"
#pragma clang diagnostic ignored "-Wimplicit-int-conversion"
#pragma clang diagnostic ignored "-Wshorten-64-to-32"
#pragma clang diagnostic ignored "-Wextra-semi"
#if defined(__clang_major__) && __clang_major__ > 12
#pragma clang diagnostic ignored "-Wreserved-identifier"
#endif
#endif

#ifdef __NVCOMPILER
#pragma diag_suppress 550 // set_but_not_used
#endif

#include "random123/threefry.h"
#include "random123/uniform.hpp"

#ifdef __NVCOMPILER
#pragma diag_warning 550 // set_but_not_used
#endif

#if defined(__clang__) && !defined(__ibmxl__)
// Restore clang diagnostics to previous state.
#pragma clang diagnostic pop
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#ifdef _MSC_FULL_VER
#pragma warning(pop)
#endif

#include <algorithm>

namespace rtt_rng {

// Forward declaration.
class Counter_RNG;

// Select a particular counter-based random number generator from Random123.
using CBRNG = r123::Threefry2x64;

// Counter and key types.
using ctr_type = CBRNG::ctr_type;
using key_type = CBRNG::key_type;

//----------------------------------------------------------------------------//
/*! \brief Generate a random double.
 *
 * Given a pointer to RNG state data, this function returns a random double in
 * the open interval (0, 1)---i.e., excluding the endpoints.
 */
GPU_HOST_DEVICE
inline double _ran(ctr_type::value_type *const data) {
  CBRNG rng;

  // Assemble a counter from the first two elements in data.
  ctr_type ctr = {{data[0], data[1]}};

  // Assemble a key from the last two elements in data.
  const key_type key = {{data[2], data[3]}};

  // Invoke the counter-based rng.
  const ctr_type result = rng(ctr, key);

  // Increment the counter.
  ctr.incr();

  // Copy the updated counter back into data.
  data[0] = ctr[0];
  data[1] = ctr[1];

  // Convert the first 64 bits of the RNG output into a double-precision value
  // in the open interval (0, 1) and return it.
  return r123::u01fixedpt<double, ctr_type::value_type>(result[0]);
}

} // namespace

//===========================================================================//
/*!
 * \class RNG
 * \brief A counter-based random-number generator.
 *
 * RNG provides an interface to a counter-based random number generator
 * from the Random123 library from D. E. Shaw Research
 * (http://www.deshawresearch.com/resources_random123.html).
 *
 * Similarly, Rnd_Control is a friend of RNG because initializing a
 * generator requires access to private data that should not be exposed through
 * the public interface.  Rnd_Control takes no responsibility for instantiating
 * RNGs itself, and since copying RNGs is disabled (via a
 * private copy constructor), an Rnd_Control must be able to initialize a
 * generator that was instantiated outside of its control.
 */
//===========================================================================//
class RNG {

public:
  typedef rtt_rng::ctr_type::const_iterator const_iterator;

  /*! \brief Default constructor.
   *
   * This default constructor is invoked when a client wants to create a
   * RNG
   */
  RNG() {}

  RNG(const uint32_t seed, const uint64_t streamnum) {
  // Low bits of the counter.
  data[0] = 0;

  // High bits of the counter; used for the seed.
  data[1] = static_cast<uint64_t>(seed) << 32;

  // Low bits of the key; used for the stream number.
  data[2] = streamnum;

  // High bits of the key; used as a spawn counter.
  data[3] = 0;
  }

  //! Return a random double in the interval (0, 1).
  GPU_HOST_DEVICE
  double generate_random_number() const { return rtt_rng::_ran(data); }

  //! Return the stream number.
  uint64_t get_num() const { return data[2]; }

private:
  mutable rtt_rng::ctr_type::value_type data[4];

};

//---------------------------------------------------------------------------//
// Implementation
//---------------------------------------------------------------------------//

#endif
//----------------------------------------------------------------------------//
// end of RNG.h
//----------------------------------------------------------------------------//
