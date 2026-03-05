#!/bin/bash

set -euo pipefail

function clone() {
  cd $HOME
  mkdir -p ~/install

  # BRANSON
  if [ -d "$HOME/branson" ]; then
    echo "Branson directory already exists"
  else
    git clone git@github.com:JDTruj2018/branson.git
    cd branson
    git checkout FCR
    cd ..
  fi

  # Umpire
  if [ -d "$HOME/Umpire" ]; then
    echo "Umpire directory already exists"
  else
    git clone git@github.com:llnl/Umpire.git
    cd Umpire  
    git checkout v2025.12.0
    git submodule init
    git submodule update
    cd ..
  fi

  # Adiak
  if [ -d "$HOME/Adiak" ]; then
    echo "Adiak directory already exists"
  else
    git clone git@github.com:llnl/Adiak.git
    cd Adiak
    git checkout v0.5.0
    git submodule init
    git submodule update
    cd ..
  fi

  # Caliper
  if [ -d "$HOME/Caliper" ]; then
    echo "Caliper directory already exists"
  else
    git clone git@github.com:llnl/Caliper.git
    cd Caliper
    git checkout v2.14.0
    cd ..
  fi

  # Spack (Metis)
  if [ -d "$HOME/spack" ]; then
    echo "Spack directory already exists"
  else
    git clone --depth=2 https://github.com/spack/spack.git
  fi
}

function build() {
  module load PrgEnv-gnu
  module load rocm/6.4.3

  cd /var/tmp/$USER

  mkdir -p tmp-build
  cd tmp-build

  # Umpire
  mkdir -p Umpire
  cd Umpire
  cmake -DENABLE_HIP=ON -DUMPIRE_ENABLE_C=ON -DUMPIRE_ENABLE_NUMA=ON -DUMPIRE_ENABLE_FORTRAN=ON -DUMPIRE_ENABLE_MPI=ON -DUMPIRE_ENABLE_OPENMP=ON -DUMPIRE_ENABLE_TESTS=OFF -DENABLE_TESTS=OFF -DCMAKE_CXX_COMPILER=amdclang++ -DCMAKE_C_COMPILER=hipcc -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/install/ $HOME/Umpire
  make -j
  make install
  cd ..
  rm -rf Umpire

  # Adiak
  mkdir -p Adiak
  cd Adiak
  cmake -DBUILD_SHARED_LIBS=ON -DENABLE_MPI=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/install $HOME/Adiak
  make -j
  make install
  cd ..
  rm -rf Adiak

  # Caliper
  mkdir -p Caliper
  cd Caliper
  cmake -DWITH_ADIAK=ON -DWITH_MPI=ON -DWITH_PAPI=ON -DWITH_ARCH=gfx942 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/install/ $HOME/Caliper
  make -j
  make install
  cd ..
  rm -rf Caliper

  # Metis
  . $HOME/spack/share/spack/setup-env.sh
  spack env activate --create branson
  if ! spack find --silent "metis" >/dev/null 2>&1; then
    spack compiler find
    spack external find
    spack external find --not-buildable hip hipcc llvm-amdgpu hsa-rocr-dev rocm-cmake rocm rocm-core rccl rocdecode rocinfo rocprofiler-sdk
    spack add metis
    spack concretize -f
    spack install metis
  else
    echo "metis is already installed"
  fi

  spack load metis

  # Branson
  mkdir -p Branson
  cd Branson
  export ROCM_PATH=$(dirname "$(dirname "$(which hipcc)")")
  cmake -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON -DUSE_GPU=ON -DUSE_UMPIRE=ON -DUSE_CALIPER=ON -DN_GROUPS=30 -DCMAKE_PREFIX_PATH=$HOME/install/ -DCMAKE_INSTALL_PREFIX=$HOME/install/ $HOME/branson/src/
  make -j
  make install
  cd ..
  rm -rf Branson 

  cd ~
}

clone
build
