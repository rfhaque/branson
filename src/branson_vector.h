//----------------------------------*-C++-*-----------------------------------//
/*!
 * \file   branson_vector.h
 * \brief  Minimal vector-like container backed by malloc/free storage.
 */
//----------------------------------------------------------------------------//

#ifndef branson_vector_h_
#define branson_vector_h_

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "config.h"

namespace branson {

template <typename T> class vector {
public:
  using value_type = T;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using reference = T &;
  using const_reference = const T &;
  using pointer = T *;
  using const_pointer = const T *;
  using iterator = T *;
  using const_iterator = const T *;

  vector() = default;

  explicit vector(const size_type count) {
    resize(count);
    std::fill(begin(), end(), T{});
  }

  vector(const size_type count, const T &value) {
    resize(count);
    std::fill(begin(), end(), value);
  }

  template <typename InputIt,
            typename = std::enable_if_t<!std::is_integral_v<InputIt>>>
  vector(InputIt first, InputIt last) {
    assign(first, last);
  }

  vector(std::initializer_list<T> init) { assign(init.begin(), init.end()); }

  vector(const vector &other) { copy_from(other); }

  vector(vector &&other) noexcept { move_from(std::move(other)); }

  vector &operator=(const vector &other) {
    if (this != &other) {
      vector tmp(other);
      swap(tmp);
    }
    return *this;
  }

  vector &operator=(vector &&other) noexcept {
    if (this != &other) {
      destroy_and_deallocate();
      move_from(std::move(other));
    }
    return *this;
  }

  ~vector() { destroy_and_deallocate(); }

  reference operator[](const size_type index) { return data_[index]; }
  const_reference operator[](const size_type index) const {
    return data_[index];
  }

  reference front() { return data_[0]; }
  const_reference front() const { return data_[0]; }

  reference back() { return data_[size_ - 1]; }
  const_reference back() const { return data_[size_ - 1]; }

  pointer data() noexcept { return data_; }
  const_pointer data() const noexcept { return data_; }

  iterator begin() noexcept { return data_; }
  const_iterator begin() const noexcept { return data_; }
  const_iterator cbegin() const noexcept { return data_; }

  iterator end() noexcept { return data_ == nullptr ? nullptr : data_ + size_; }
  const_iterator end() const noexcept {
    return data_ == nullptr ? nullptr : data_ + size_;
  }
  const_iterator cend() const noexcept {
    return data_ == nullptr ? nullptr : data_ + size_;
  }

  bool empty() const noexcept { return size_ == 0; }
  size_type size() const noexcept { return size_; }
  size_type capacity() const noexcept { return capacity_; }

  void clear() noexcept { size_ = 0; }

  void reserve(const size_type new_capacity) {
    if (new_capacity <= capacity_) {
      return;
    }

    const size_type old_size = size_;
    pointer new_data = allocate_and_default_construct(new_capacity);
    try {
      if (old_size != 0) {
        std::copy(begin(), end(), new_data);
      }
    } catch (...) {
      destroy_range(new_data, new_capacity);
      deallocate_storage(new_data);
      throw;
    }

    destroy_and_deallocate();
    data_ = new_data;
    size_ = old_size;
    capacity_ = new_capacity;
  }

  void resize(const size_type new_size) {
    if (new_size > capacity_) {
      reserve(new_size);
    }

    if (new_size > size_) {
      std::fill(data_ + size_, data_ + new_size, T{});
    }

    size_ = new_size;
  }

  void resize(const size_type new_size, const T &value) {
    if (new_size > capacity_) {
      reserve(new_size);
    }

    if (new_size > size_) {
      std::fill(data_ + size_, data_ + new_size, value);
    }

    size_ = new_size;
  }

  void assign(const size_type count, const T &value) {
    if (count == 0) {
      size_ = 0;
      return;
    }
    if (count > capacity_) {
      reserve(count);
    }
    std::fill(data_, data_ + count, value);
    size_ = count;
  }

  template <typename InputIt,
            typename = std::enable_if_t<!std::is_integral_v<InputIt>>>
  void assign(InputIt first, InputIt last) {
    const size_type count = static_cast<size_type>(std::distance(first, last));
    if (count == 0) {
      size_ = 0;
      return;
    }
    if (count > capacity_) {
      reserve(count);
    }
    std::copy(first, last, data_);
    size_ = count;
  }

  void push_back(const T &value) {
    if (size_ == capacity_) {
      reserve(growth_capacity(capacity_, size_ + 1));
    }
    data_[size_] = value;
    ++size_;
  }

  iterator insert(const_iterator pos, const T &value) {
    return insert_fill(index_from(pos), 1, value);
  }

  iterator insert(const_iterator pos, const size_type count, const T &value) {
    return insert_fill(index_from(pos), count, value);
  }

  template <typename InputIt,
            typename = std::enable_if_t<!std::is_integral_v<InputIt>>>
  iterator insert(const_iterator pos, InputIt first, InputIt last) {
    const size_type index = index_from(pos);
    const size_type count = static_cast<size_type>(std::distance(first, last));
    if (count == 0) {
      return ptr_at(index);
    }

    if (size_ + count > capacity_) {
      reserve(growth_capacity(capacity_, size_ + count));
    }

    std::move_backward(begin() + index, end(), end() + count);
    std::copy(first, last, begin() + index);
    size_ += count;
    return begin() + index;
  }

  iterator erase(const_iterator pos) {
    const size_type index = index_from(pos);
    return erase(pos, const_ptr_at(index + 1));
  }

  iterator erase(const_iterator first, const_iterator last) {
    const size_type first_index = index_from(first);
    const size_type last_index = index_from(last);
    if (first_index == last_index) {
      return ptr_at(first_index);
    }

    std::move(begin() + last_index, end(), begin() + first_index);
    size_ -= (last_index - first_index);
    return ptr_at(first_index);
  }

  void swap(vector &other) noexcept {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
    std::swap(capacity_, other.capacity_);
  }

private:
  static_assert(std::is_default_constructible<T>::value,
                "vector requires default-constructible value types.");
  static_assert(std::is_copy_assignable<T>::value,
                "vector requires copy-assignable value types.");

  pointer data_ = nullptr;
  size_type size_ = 0;
  size_type capacity_ = 0;

  static pointer allocate_and_default_construct(const size_type count) {
    if (count == 0) {
      return nullptr;
    }

    if (count > std::numeric_limits<size_type>::max() / sizeof(T)) {
      throw std::bad_array_new_length();
    }

    void *raw = allocate_storage(count * sizeof(T));
    if (raw == nullptr) {
      throw std::bad_alloc();
    }

    pointer data = static_cast<pointer>(raw);
    try {
      std::uninitialized_fill_n(data, count, T{});
    } catch (...) {
      deallocate_storage(data);
      throw;
    }

    return data;
  }

  static void destroy_range(pointer data, const size_type count) noexcept {
    if (data == nullptr) {
      return;
    }
    std::destroy_n(data, count);
  }

  static void *allocate_storage(const size_t bytes) {
#ifdef USE_UMPIRE
    void* data;
    umpireHostMalloc(&data, bytes);
    return data;
#else
    return std::malloc(bytes);
#endif
  }

  static void deallocate_storage(void *ptr) noexcept {
#ifdef USE_UMPIRE
    if (ptr != nullptr) {
      umpireHostFree(ptr);
    }
#else
    std::free(ptr);
#endif
  }

  static size_type growth_capacity(const size_type current_capacity,
                                   const size_type min_capacity) {
    size_type new_capacity = std::max(current_capacity, capacity_floor());
    while (new_capacity < min_capacity) {
      if (new_capacity > std::numeric_limits<size_type>::max() / 2) {
        return min_capacity;
      }
      new_capacity *= 2;
    }
    return new_capacity;
  }

  static constexpr size_type capacity_floor() noexcept { return 1; }

  size_type index_from(const_iterator pos) const noexcept {
    return data_ == nullptr ? 0 : static_cast<size_type>(pos - cbegin());
  }

  iterator ptr_at(const size_type index) noexcept {
    return data_ == nullptr ? nullptr : data_ + index;
  }

  const_iterator const_ptr_at(const size_type index) const noexcept {
    return data_ == nullptr ? nullptr : data_ + index;
  }

  iterator insert_fill(const size_type index, const size_type count,
                       const T &value) {
    if (count == 0) {
      return ptr_at(index);
    }

    if (size_ + count > capacity_) {
      reserve(growth_capacity(capacity_, size_ + count));
    }

    std::move_backward(begin() + index, end(), end() + count);
    std::fill(begin() + index, begin() + index + count, value);
    size_ += count;
    return ptr_at(index);
  }

  void copy_from(const vector &other) {
    pointer new_data = allocate_and_default_construct(other.capacity_);
    try {
      if (other.size_ != 0) {
        std::copy(other.begin(), other.end(), new_data);
      }
    } catch (...) {
      destroy_range(new_data, other.capacity_);
      deallocate_storage(new_data);
      throw;
    }

    data_ = new_data;
    size_ = other.size_;
    capacity_ = other.capacity_;
  }

  void move_from(vector &&other) noexcept {
    data_ = other.data_;
    size_ = other.size_;
    capacity_ = other.capacity_;
    other.data_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
  }

  void destroy_and_deallocate() noexcept {
    destroy_range(data_, capacity_);
    deallocate_storage(data_);
    data_ = nullptr;
    size_ = 0;
    capacity_ = 0;
  }
}; // end class vector

} // end namespace branson
#endif // branson_vector_h_
