//----------------------------------*-C++-*-----------------------------------//
/*!
 * \file   malloc_vector.h
 * \author Alex Long
 * \date   June 28 2026
 * \brief  Simple malloc-backed vector with manual element lifetime management
 */
//----------------------------------------------------------------------------//

#ifndef malloc_vector_h_
#define malloc_vector_h_

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>

template <typename T> class MallocVector {
public:
  using value_type = T;
  using size_type = size_t;
  using iterator = T *;
  using const_iterator = const T *;

  MallocVector() = default;

  MallocVector(const MallocVector &other) { copy_from(other); }

  MallocVector(MallocVector &&other) noexcept { swap(other); }

  MallocVector &operator=(const MallocVector &other) {
    if (this != &other) {
      MallocVector copy(other);
      swap(copy);
    }
    return *this;
  }

  MallocVector &operator=(MallocVector &&other) noexcept {
    if (this != &other) {
      reset();
      swap(other);
    }
    return *this;
  }

  ~MallocVector() { reset(); }

  iterator begin() noexcept { return data_; }
  const_iterator begin() const noexcept { return data_; }
  iterator end() noexcept { return iterator_at(size_); }
  const_iterator end() const noexcept { return iterator_at(size_); }

  T *data() noexcept { return data_; }
  const T *data() const noexcept { return data_; }

  bool empty() const noexcept { return size_ == 0; }
  size_type size() const noexcept { return size_; }

  T &operator[](size_type index) noexcept { return data_[index]; }
  const T &operator[](size_type index) const noexcept { return data_[index]; }

  void clear() noexcept {
    destroy_range(0, size_);
    size_ = 0;
  }

  void reserve(size_type new_capacity) {
    if (new_capacity <= capacity_) {
      return;
    }

    T *new_data = allocate_raw(new_capacity);
    size_type i = 0;
    try {
      for (; i < size_; ++i) {
        new (new_data + i) T(std::move_if_noexcept(data_[i]));
      }
    } catch (...) {
      for (size_type j = 0; j < i; ++j) {
        std::destroy_at(new_data + j);
      }
      std::free(new_data);
      throw;
    }

    destroy_range(0, size_);
    std::free(data_);
    data_ = new_data;
    capacity_ = new_capacity;
  }

  void resize(size_type new_size) {
    if (new_size < size_) {
      destroy_range(new_size, size_);
      size_ = new_size;
      return;
    }

    if (new_size > capacity_) {
      reserve(growth_capacity(new_size));
    }

    std::uninitialized_default_construct_n(data_ + size_, new_size - size_);
    size_ = new_size;
  }

  void push_back(const T &value) {
    if (size_ == capacity_) {
      reserve(growth_capacity(size_ + 1));
    }
    new (data_ + size_) T(value);
    ++size_;
  }

  void push_back(T &&value) {
    if (size_ == capacity_) {
      reserve(growth_capacity(size_ + 1));
    }
    new (data_ + size_) T(std::move(value));
    ++size_;
  }

  iterator erase(const_iterator first, const_iterator last) {
    const size_type first_index = static_cast<size_type>(first - begin());
    const size_type last_index = static_cast<size_type>(last - begin());

    if (first_index > last_index || last_index > size_) {
      throw std::out_of_range("Invalid range in MallocVector::erase");
    }

    const size_type removed = last_index - first_index;
    if (removed == 0) {
      return iterator_at(first_index);
    }

    for (size_type i = first_index; i + removed < size_; ++i) {
      data_[i] = std::move(data_[i + removed]);
    }

    destroy_range(size_ - removed, size_);
    size_ -= removed;
    return iterator_at(first_index);
  }

private:
  T *data_{nullptr};
  size_type size_{0};
  size_type capacity_{0};

  static T *allocate_raw(size_type count) {
    if (count == 0) {
      return nullptr;
    }

    T *raw = static_cast<T *>(std::malloc(count * sizeof(T)));
    if (raw == nullptr) {
      throw std::bad_alloc();
    }
    return raw;
  }

  static size_type growth_capacity(size_type required) noexcept {
    size_type new_capacity = 1;
    while (new_capacity < required) {
      new_capacity *= 2;
    }
    return new_capacity;
  }

  void destroy_range(size_type first, size_type last) noexcept {
    for (size_type i = first; i < last; ++i) {
      std::destroy_at(data_ + i);
    }
  }

  iterator iterator_at(size_type index) noexcept {
    return data_ == nullptr ? nullptr : data_ + index;
  }

  const_iterator iterator_at(size_type index) const noexcept {
    return data_ == nullptr ? nullptr : data_ + index;
  }

  void reset() noexcept {
    clear();
    std::free(data_);
    data_ = nullptr;
    capacity_ = 0;
  }

  void copy_from(const MallocVector &other) {
    if (other.size_ == 0) {
      return;
    }

    data_ = allocate_raw(other.size_);
    capacity_ = other.size_;

    size_type i = 0;
    try {
      for (; i < other.size_; ++i) {
        new (data_ + i) T(other.data_[i]);
      }
      size_ = other.size_;
    } catch (...) {
      for (size_type j = 0; j < i; ++j) {
        std::destroy_at(data_ + j);
      }
      std::free(data_);
      data_ = nullptr;
      capacity_ = 0;
      throw;
    }
  }

  void swap(MallocVector &other) noexcept {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
    std::swap(capacity_, other.capacity_);
  }
};

#endif // malloc_vector_h_
//----------------------------------------------------------------------------//
// end of malloc_vector.h
//----------------------------------------------------------------------------//
