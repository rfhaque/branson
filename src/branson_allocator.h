//----------------------------------*-C++-*-----------------------------------//
/*!
 * \file   branson_allocator.h
 * \brief  Host allocation helpers used by branson containers.
 */
//----------------------------------------------------------------------------//

#ifndef branson_allocator_h_
#define branson_allocator_h_

#include <cstddef>
#include <cstdlib>

#ifdef USE_UMPIRE
#include <umpire/Umpire.hpp>
#include <umpire/strategy/QuickPool.hpp>
#endif

#ifdef USE_UMPIRE
inline void makeUmpireHostPool() {
  auto &rm = umpire::ResourceManager::getInstance();
  const char *allocator_name = "BRANSON_HOST_POOL";
  if (!rm.isAllocator(allocator_name)) {
    size_t umpire_host_pool_size = static_cast<size_t>(8) << 30;
    size_t umpire_host_block_size = 512;
    auto host_pool_allocator = rm.makeAllocator<umpire::strategy::QuickPool>(
        allocator_name, rm.getAllocator("HOST"), umpire_host_pool_size,
        umpire_host_block_size);
    void *tmp = host_pool_allocator.allocate(100);
    host_pool_allocator.deallocate(tmp);
  }
}

inline void umpireHostMalloc(void **ptr, size_t size) {
  makeUmpireHostPool();
  auto &rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("BRANSON_HOST_POOL");
  *ptr = allocator.allocate(size);
}

inline void umpireHostFree(void *ptr) {
  makeUmpireHostPool();
  auto &rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("BRANSON_HOST_POOL");
  allocator.deallocate(ptr);
}
#else
inline void makeUmpireHostPool() {}
#endif

#endif // branson_allocator_h_
