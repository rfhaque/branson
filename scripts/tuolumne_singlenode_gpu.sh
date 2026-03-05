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

set -euo pipefail

echo "=== Allocation resources ==="
flux resource list
echo

function usage() {
  cat <<EOF
Usage: ${0} [-n RUN_NAME]

Options:
  -n    RUN_NAME Identified of this run
EOF
}

while getopts "n:" flag; do
  case "$flag" in
    n)
      RUN_NAME=$OPTARG
      ;;
    *)
      exit 1
      ;;
  esac
done

if [ -z "$RUN_NAME" ]; then
  echo "Error: Missing mandatory flag"
  usage
  exit 1
fi

function results() {
  echo "$1,$2,$(awk -F': ' 'END{print $2}' $3)" >> results.txt
}

function run() {
  export BRANSON_BIN=$HOME/install/bin/BRANSON
  export BRANSON_INPUT=$HOME/branson/inputs/3D_hohlraum_single_node.xml

  cd /p/lustre5/$USER

  mkdir -p $RUN_NAME
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

module load PrgEnv-gnu
module load rocm/6.4.3

run

