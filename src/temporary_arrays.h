#ifndef temporary_arrays_h_
#define temporary_arrays_h_

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "config.h"

class Temporary_Arrays {
public:
  Temporary_Arrays(const uint32_t n_source_cells, const uint32_t n_tally_cells)
      : m_n_source_cells(n_source_cells), m_n_tally_cells(n_tally_cells) {
    allocate_array(&source_census_count, m_n_source_cells);
    allocate_array(&source_emission_count, m_n_source_cells);
    allocate_array(&source_boundary_count, m_n_source_cells);
    allocate_array(&source_offset, m_n_source_cells);
    allocate_array(&abs_E, m_n_tally_cells);
    allocate_array(&track_E, m_n_tally_cells);
    reset_source_arrays();
    reset_tally_arrays();
  }

  ~Temporary_Arrays() {
    hostFree(source_census_count);
    hostFree(source_emission_count);
    hostFree(source_boundary_count);
    hostFree(source_offset);
    hostFree(abs_E);
    hostFree(track_E);
  }

  Temporary_Arrays(const Temporary_Arrays &) = delete;
  Temporary_Arrays &operator=(const Temporary_Arrays &) = delete;

  void reset_source_arrays() {
    std::fill(source_census_count, source_census_count + m_n_source_cells, 0U);
    std::fill(source_emission_count, source_emission_count + m_n_source_cells, 0U);
    std::fill(source_boundary_count, source_boundary_count + m_n_source_cells, 0U);
    std::fill(source_offset, source_offset + m_n_source_cells, 0UL);
  }

  void reset_tally_arrays() {
    std::fill(abs_E, abs_E + m_n_tally_cells, 0.0);
    std::fill(track_E, track_E + m_n_tally_cells, 0.0);
  }

  uint32_t get_n_source_cells() const { return m_n_source_cells; }
  uint32_t get_n_tally_cells() const { return m_n_tally_cells; }

  uint32_t *source_census_count{nullptr};
  uint32_t *source_emission_count{nullptr};
  uint32_t *source_boundary_count{nullptr};
  uint64_t *source_offset{nullptr};
  double *abs_E{nullptr};
  double *track_E{nullptr};

private:
  template <typename T>
  void allocate_array(T **ptr, const uint32_t count) {
    if (count == 0) {
      *ptr = nullptr;
      return;
    }

    hostMalloc(ptr, static_cast<size_t>(count) * sizeof(T));
    Insist(*ptr != nullptr, "Failed to allocate temporary array");
  }

  uint32_t m_n_source_cells;
  uint32_t m_n_tally_cells;
};

#endif // temporary_arrays_h_
