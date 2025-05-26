################################################################################
# NOTES ON COMPILATION
################################################################################

# Load the modules for each libary and the environment variables will be set
# correctly (load cmake, compiler, mpi, metis, and optionally
# hdf5 and silo)

# Enbable debug mode by passing -DCMAKE_BUILD_TYPE=Debug to CMake, default is
# Release

cmake_minimum_required (VERSION 3.23)

project (BRANSON
  VERSION 0.8
  DESCRIPTION "Branson can be used to study different algorithms for parallel Monte Carlo transport. Currently it contains particle passing and mesh passing methods for domain decomposition."
)

## Fail if someone tries to config an in-source build.
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
   message(FATAL_ERROR "In-source builds are not supported. Please remove "
                       "CMakeCache.txt from the 'src' dir and configure an "
                       "out-of-source build in another directory.")
endif()

# Set the build type to Release by default if not set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
string(REPLACE " " ";" CMAKE_BRANSON_CXXFLAGS "${CMAKE_CXX_FLAGS}")

option(ENABLE_CUDA "Use CUDA" FALSE)
option(ENABLE_HIP "Use HIP" FALSE)
option(ENABLE_CALIPER "Enable Caliper" FALSE)
option(ENABLE_OPENMP "Enable OpenMP" FALSE)
option(ENABLE_METIS "Enable METIS" FALSE)
option(ENABLE_VIZ "Enable VIZ" FALSE)

set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(MPI REQUIRED)
list(APPEND branson_deps
  MPI::MPI_CXX)

##############################################################################
# OpenMP
##############################################################################
if(ENABLE_OPENMP)
  message(STATUS "Looking for OpenMP...")
  find_package(OpenMP REQUIRED)
  if(OpenMP_FOUND)
    message(STATUS "Looking for openmp......found ${OpenMP_CXX_FLAGS}")
    set(USE_OPENMP ON)
    list(APPEND branson_deps
      OpenMP::OpenMP_CXX
    )
  endif()
else()
  message(STATUS "Disabling OpenMP...")
  set(USE_OPENMP OFF)
endif()

if(ENABLE_CUDA)
  enable_language(CXX)

  find_package(CUDAToolkit REQUIRED)
  list(APPEND branson_deps
    CUDA::cudart)

  include(CheckLanguage)
  check_language(CUDA)

  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "Unable to find the nvcc compiler. Please use"
                        "CMAKE_CUDA_COMPILER to provide the nvcc compiler.")
  endif()

  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH}")
  set_source_files_properties("main.cc" PROPERTIES LANGUAGE CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")
  set(CMAKE_CUDA_FLAGS "-g --expt-relaxed-constexpr ${CMAKE_CUDA_FLAGS}")
  set(CMAKE_CUDA_FLAGS "-expt-extended-lambda ${CMAKE_CUDA_FLAGS}")
  list(APPEND CMAKE_BRANSON_CXXFLAGS -DUSE_GPU)
  list(APPEND CMAKE_BRANSON_CXXFLAGS -DHAS_GPU)

  set(GPU_DBS_STRING "CUDA" CACHE STRING "If CUDA is available, this variable is 'CUDA'")

  message("Making GPU(CUDA) BRANSON")

elseif(ENABLE_HIP)
  if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
      set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
      set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
  endif()

  if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which ROCM has been installed")
    elseif(DEFINED ENV{HIP_PATH})
      set(ROCM_PATH "$ENV{HIP_PATH}/.." CACHE PATH "Path to which ROCM has been installed")
    else()
      set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which ROCM has been installed")
    endif()
  endif()

  if(NOT DEFINED HCC_PATH)
    if(DEFINED ENV{HCC_PATH})
      set(HCC_PATH $ENV{HCC_PATH} CACHE PATH "Path to which HCC has been installed")
    else()
      set(HCC_PATH "${ROCM_PATH}/hcc" CACHE PATH "Path to which HCC has been installed")
    endif()
      set(HCC_HOME "${HCC_PATH}")
  endif()

  if(NOT DEFINED HIP_CLANG_PATH)
    if(NOT DEFINED ENV{HIP_CLANG_PATH})
      set(HIP_CLANG_PATH "${ROCM_PATH}/llvm/bin" CACHE PATH "Path to which HIP compatible clang binaries have been installed")
    else()
      set(HIP_CLANG_PATH $ENV{HIP_CLANG_PATH} CACHE PATH "Path to which HIP compatible clang binaries have been installed")
    endif()
  endif()

  set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
  list(APPEND CMAKE_PREFIX_PATH
    "${HIP_PATH}/lib/cmake"
    "${HIP_PATH}/../lib/cmake" # hopefully catches all extra HIP dependencies
  )

  find_package(HIP QUIET)
  if(HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})
  else()
    message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
  endif()
  find_package(hip REQUIRED)

  # For ROCm >=3.5, wipe hip-clang specific interface options which are propagated
  set_target_properties(hip::device PROPERTIES INTERFACE_COMPILE_OPTIONS "-fPIC")
  set_target_properties(hip::device PROPERTIES INTERFACE_LINK_LIBRARIES "hip::host")

  list(APPEND branson_deps
    "-fgpu-rdc"
    "--hip-link"
    "--offload-arch=${HIP_ARCH}"
    "--rocm-path=${ROCM_PATH}"
    hip::host
    hip::device)
  list(APPEND CMAKE_BRANSON_CXXFLAGS -DUSE_GPU)
  list(APPEND CMAKE_BRANSON_CXXFLAGS -DHAS_GPU)
  list(APPEND CMAKE_BRANSON_CXXFLAGS -x hip)
  list(APPEND CMAKE_BRANSON_CXXFLAGS "--offload-arch=${HIP_ARCH}")
  list(APPEND CMAKE_BRANSON_CXXFLAGS "-fgpu-rdc")
  list(APPEND CMAKE_BRANSON_CXXFLAGS "-Wno-unused-result")

  #set_source_files_properties("main.cc" PROPERTIES LANGUAGE HIP)

  set(GPU_DBS_STRING "HIP" CACHE STRING "If HIP is available, this variable is 'HIP'")

  message("Making GPU(HIP) BRANSON")

else()
  message("Making CPU BRANSON")

endif()

message("GPU_DBS_STRING: ${GPU_DBS_STRING}")

# Build system support files are located here.
set( CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/config ${CMAKE_MODULE_PATH})

message( "
This is ${CMAKE_PROJECT_NAME} v. ${PROJECT_VERSION}.
")

#------------------------------------------------------------------------------#
# User configurable options

set (BRANSON_N_GROUPS ${N_GROUPS})
if (NOT BRANSON_N_GROUPS)
  message(STATUS
    "HARDCODED NUMBER OF GROUPS NOT SET, DEFAULTING TO ONE GROUP (GRAY)
     use '-DN_GROUPS=<int>' when running cmake to select multigroup.\n")
  set(BRANSON_N_GROUPS "1")
else ()
  message(STATUS
    "HARDCODED NUMBER OF GROUPS SET TO ${BRANSON_N_GROUPS} (FAUX MULTIGROUP)\n")
endif()
set(BRANSON_N_GROUPS ${BRANSON_N_GROUPS} CACHE STRING "Number of energy groups." FORCE)

set(ENABLE_VERBOSE_GPU_TRANSPORT ${ENABLE_VERBOSE_GPU_TRANSPORT} CACHE STRING "Flag for verbose GPU output." FORCE)

#------------------------------------------------------------------------------#
# Record the site name

site_name( SITENAME )
string( REGEX REPLACE "([A-z0-9]+).*" "\\1" SITENAME ${SITENAME} )
include (lanl-setup)
set( SITENAME ${SITENAME} CACHE "STRING" "Name of the current machine" FORCE)

# GCC options for address or undefined sanitizing
#set(GCC_SANITIZE_COMPILE_FLAGS "-fsanitize=address")
#set(GCC_SANITIZE_LINK_FLAGS "-fsanitize=address")
#set(GCC_SANITIZE_COMPILE_FLAGS "-fsanitize=undefined")
#set(GCC_SANITIZE_LINK_FLAGS "-fsanitize=undefined")

# add compiler flags
if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -ffp-contract=off")
  set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -O0")
  #set( CMAKE_CXX_FLAGS_RELEASE "-march=native  -O3 -funroll-loops -fno-var-tracking-assignments")
  set( CMAKE_CXX_FLAGS_RELEASE "-g -O3 -funroll-loops -fno-var-tracking-assignments")
  # optionally add sanitize flags
  #set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${GCC_SANITIZE_COMPILE_FLAGS}")
  #set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${GCC_SANITIZE_LINK_FLAGS}")
elseif( CMAKE_CXX_COMPILER_ID STREQUAL "Intel" )
  set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -O0")
  # note that the "fp-model=precise" flag significantly limits vectorization and slows down
  # branson but it makes it the hardware counter data match Jayenne more closely (Jayenne uses
  # this flag)
  #set( CMAKE_CXX_FLAGS_RELEASE "-g -O3 -fp-speculation=fast -xhost -qno-opt-dynamic-align")
	set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -xhost -qopenmp -qopt-zmm-usage=high -fp-speculation=fast -qopt-report=3 -qopt-report-file=runAVX512.optrpt -no-vec") # "-no-vec" to disable vectorization
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -Wall -O0 -xhost -qopenmp")
	set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -xhost -qopenmp -qopt-zmm-usage=high -fp-speculation=fast -qopt-report=3 -qopt-report-file=runAVX512.optrpt") # "-no-vec" to disable vectorization
elseif( CMAKE_CXX_COMPILER_ID STREQUAL "Cray" )
  set( CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -g -DR123_USE_GNU_UINT128=0")
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -DR123_USE_GNU_UINT128=0")
endif()

if(ENABLE_CUDA)
  string(JOIN " " CMAKE_BRANSON_CUDAFLAGS ${CMAKE_BRANSON_CXXFLAGS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler ${CMAKE_BRANSON_CUDAFLAGS}")
else()
  add_compile_options(${CMAKE_BRANSON_CXXFLAGS})
endif()

#------------------------------------------------------------------------------#
# Look for Third Party Libraries (metis, etc.)
add_subdirectory(pugixml)

#------------------------------------------------------------------------------#
# Report build configuration

# Summary of TPLs
include(FeatureSummary)
message(" ")
feature_summary( WHAT ALL
  INCLUDE_QUIET_PACKAGES
  FATAL_ON_MISSING_REQUIRED_PACKAGES
  QUIET_ON_EMPTY )

# Summary of user-selectable build options
message( "\nBuild Summary:\n")
message( STATUS "Machine name    : ${SITENAME}")
report_lanl_hpc_features()
if( ${BRANSON_N_GROUPS} GREATER 1 )
  message( STATUS "Energy groups   : ${BRANSON_N_GROUPS}")
else()
  message( STATUS "Energy groups   : Gray (1-group)")
endif()
message( STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message( STATUS "Compiler        : ${CMAKE_CXX_COMPILER}")
string( REPLACE ";" " " CMAKE_BRANSON_CXXFLAGS_STR "${CMAKE_BRANSON_CXXFLAGS}")
if( CMAKE_CONFIGURATION_TYPES )
  message( STATUS "Compiler Flags (All)    : ${CMAKE_CXX_FLAGS} ${CMAKE_BRANSON_CXXFLAGS_STR}")
  message( STATUS "Compiler Flags (Debug)  : ${CMAKE_CXX_FLAGS_DEBUG}")
  message( STATUS "Compiler Flags (Release): ${CMAKE_CXX_FLAGS_RELEASE}")
else()
  string( TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UPPER )
  message( STATUS "Compiler Flags  : ${CMAKE_CXX_FLAGS} ${CMAKE_BRANSON_CXXFLAGS_STR} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}}")
endif()
message("\n")

list(APPEND branson_deps pugixml)

##############################################################################
# caliper/adiak
##############################################################################
if(ENABLE_CALIPER)
  message(STATUS "Looking for caliper...")
  find_package(caliper REQUIRED)
  if(caliper_FOUND)
    message(STATUS "Looking for caliper......found ${caliper_DIR}")
    list(APPEND branson_deps
      caliper)
  endif()

  message(STATUS "Looking for adiak...")
  find_package(adiak REQUIRED)
  if(adiak_FOUND)
    message(STATUS "Looking for adiak......found ${adiak_DIR}")
    list(APPEND branson_deps
      adiak::adiak)
  endif()
else()
  message(STATUS "Disabling caliper...")
endif()

##############################################################################
# metis
##############################################################################
if(ENABLE_METIS)
  message(STATUS "Looking for METIS...")
  find_package(METIS REQUIRED)
  if(METIS_FOUND)
    message(STATUS "Looking for METIS......found ${METIS_LIBRARIES}")
    list(APPEND branson_deps
      METIS::metis)
  endif()
else()
  message(STATUS "Disabling METIS...")
endif()

##############################################################################
# Silo and HDF5 libraries
##############################################################################
if(ENABLE_VIZ)
  message(STATUS "Looking for HDF5...")
  find_package(HDF5 REQUIRED)
  if(HDF5_FOUND)
    list(GET HDF5_LIBRARIES 0 hdf5lib)
    message(STATUS "Looking for HDF5..found ${hdf5lib}")
    unset(hdf5lib)
    message(STATUS "Looking for Silo...")
    find_package(Silo REQUIRED)
    if(Silo_FOUND)
      message(STATUS "Looking for Silo..found ${Silo_LIBRARY}")
    endif()
    list(APPEND branson_deps
      Silo::silo
      ${HDF5_LIBRARIES}
    )
  endif()
else ()
  message(STATUS "Disabling visualization libraries...")
endif()

option(BUILD_TESTING "Should we compile the tests?" ON)
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(test)
  message(STATUS
    "Building tests enabled (default), disable tests with BUILD_TESTING=FALSE")
else()
  message(STATUS
    "Building tests disabled, set BUILD_TESTING=TRUE or don't set BUILD_TESTING to enable test builds")
endif()

#------------------------------------------------------------------------------#
# Generate config.h

configure_file(config.h.in ${PROJECT_BINARY_DIR}/config.h)
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Targets
file(GLOB headers *.h)
add_executable(BRANSON main.cc ${headers})
target_link_libraries(BRANSON PUBLIC ${branson_deps})
target_include_directories(BRANSON PRIVATE
  ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR}/pugixml/src/ ${HIP_INCLUDE_DIRS})

if(ENABLE_CUDA)
  set_target_properties(BRANSON
    PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

if(ENABLE_HIP)
  set_target_properties(BRANSON
    PROPERTIES
      HIP_SEPARABLE_COMPILATION ON
      HIP_RESOLVE_DEVICE_SYMBOLS ON)
endif()

#------------------------------------------------------------------------------#
# Targets for installation

install(TARGETS BRANSON DESTINATION ${CMAKE_INSTALL_BINDIR})

#------------------------------------------------------------------------------#
# End src/CMakeLists.txt
#------------------------------------------------------------------------------#
