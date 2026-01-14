#!/bin/bash
# flux: -N 1
# flux: -g 4
# flux: -t 2h
# flux: -q pbatch
# flux: --exclusive
# flux: --setattr=hugepages=512GB
# flux: --setattr=gpumode=CPX
# flux: --job-name=branson_1node_gpu
# flux: --output=branson_%j.out
# flux: --error=branson_%j.err
# flux: --mail-user=jereddt@lanl.gov
# flux: --mail-type=BEGIN,END,FAIL

set -euo pipefail

echo "=== Allocation resources ==="
flux resource list
echo

module load rocm

export RUN_NAME=branson-tuolumne-singlenode-gpu-release

function clone() {
  cd ~

  mkdir -p ~/binaries
  mkdir ~/binaries/$RUN_NAME

  git clone git@github.com:lanl/branson.git
  cd branson
  #git checkout FCR
}

function build() {
  cd /var/tmp/$USER

  mkdir $RUN_NAME-build
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

  mkdir $RUN_NAME
  cd $RUN_NAME

  echo "Ranks,GPU-per-Rank,FOM" >> results.txt

  for ranks in 1 2 4 8 16 24; do
    flux run -N 1 -n $ranks -g 1 --setopt=mpibind=verbose:1 --exclusive ${BRANSON_BIN} ${BRANSON_INPUT} > ${ranks}_1.txt 2>&1
    results $ranks 1 ${ranks}_1.txt
  done

  for ranks in 1 2 4 8 16 24 32 48; do
    flux run -N 1 -n $ranks -g 0 --setopt=mpibind=verbose:1 --exclusive ${BRANSON_BIN} ${BRANSON_INPUT} > ${ranks}_0.txt 2>&1
    results $ranks 0 ${ranks}_0.txt
  done
}

clone
build
run








