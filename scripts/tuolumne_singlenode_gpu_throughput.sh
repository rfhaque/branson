#!/bin/bash
# flux: -N 1
# flux: -g 4
# flux: -t 2h
# flux: -q pbatch
# flux: --exclusive
# flux: --setattr=hugepages=512GB
# flux: --setattr=gpumode=CPX
# flux: --job-name=branson_1node_gpu_throughpu
# flux: --output=branson_throughput_%j.out
# flux: --error=branson_throughput_%j.err

set -euo pipefail

echo "=== Allocation resources ==="
flux resource list
echo

module load rocm

export RUN_NAME=branson-tuolumne-singlenode-gpu-throughput-release

function clone() {
  cd ~

  mkdir -p ~/binaries
  mkdir -p ~/binaries/$RUN_NAME

  git clone git@github.com:lanl/branson.git
  cd branson
  #git checkout FCR
}

function build() {
  cd /var/tmp/$USER

  mkdir -p $RUN_NAME-build
  cd $RUN_NAME-build

  export ROCM_PATH=$(dirname "$(dirname "$(which hipcc)")")
  cmake -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON -DUSE_GPU=ON -DCMAKE_INSTALL_PREFIX=$HOME/binaries/$RUN_NAME $HOME/branson/src/

  make -j
  ctest -j32

  make install
}

function results() {
  echo "$1,$2,$(awk -F': ' 'END{print $2}' $3)" >> results.txt
}

function run() {
  export BRANSON_BIN=$HOME/binaries/$RUN_NAME/bin/BRANSON
  export BRANSON_INPUT=$HOME/branson/inputs/3D_hohlraum_single_node.xml

  cd /p/lustre5/$USER

  mkdir -p $RUN_NAME
  cd $RUN_NAME

  cp $BRANSON_INPUT .

  echo "Ranks,Particles,FOM" >> results.txt

  for p in 100000 200000 300000 400000 500000 600000 700000 800000 900000 1000000 2000000 3000000 4000000 5000000 6600000 10000000 13300000 20000000 50000000 100000000 200000000 300000000 400000000 500000000 600000000 700000000 800000000 900000000; do 
    sed "s|<photons>.*</photons>|<photons>$p</photons>|" 3D_hohlraum_single_node.xml > 3D_hohlraum_single_node_${p}.xml; 
    flux run -N 1 -n 48 -g 0 --setopt=mpibind=verbose:1 --exclusive /g/g14/jered/binaries/branson-tuolumne-singlenode-gpu-release/bin/BRANSON 3D_hohlraum_single_node_${p}.xml > ${p}.txt 2>&1;
    results $p ${p}.txt
  done
}

clone
build
run

