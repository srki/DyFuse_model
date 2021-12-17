#!/bin/bash
NUM_THREADS=${NUM_THREADS:-"-j 12"}

root=$PWD

# Determine compiler
if [[ -n "$NERSC_HOST" ]]; then
  export CC=${CC:-cc}
  export CXX=${CXX:-CC}
elif [[ "$HOSTNAME" ==  "ancilla" ]]; then
  export CC=${CC:-clang}
  export CXX=${CXX:-clang++}
else
#  export CC=${CC:-gcc}
#  export CXX=${CXX:-g++}
  export CC=${CC:-clang}
  export CXX=${CXX:-clang++}
fi

# Check if hwloc is installed
if type -p "hwloc-info"; then
  HWLOC=
else
  echo "hwloc not found"
  exit
  HWLOC=--without-hwloc
fi

# StarPU config flgas
SPU_FLAGS="--disable-fortran --disable-build-examples --disable-build-tests --enable-maxcpus=256 --enable-max-sched-ctxs=128" #--enable-cluster

# Build and install starpu
if [[ ! -d src ]]; then
  mkdir src
fi

#(
#  cd src || exit
#
#  # Download StarPU  for if was not downloaded
#  if [[ ! -f starpu-1.3.7.tar.gz ]]; then
#    if [[ $(uname) == 'Darwin' ]]; then
#      curl -O http://starpu.gforge.inria.fr/files/starpu-1.3.7/starpu-1.3.7.tar.gz
#    else
#      wget http://starpu.gforge.inria.fr/files/starpu-1.3.7/starpu-1.3.7.tar.gz
#    fi
#  fi
#
#  if [[ ! -f starpu-1.3.7.tar.gz ]]; then
#    echo "Cannot Download StarPU"
#    exit
#  fi
#
#  # debug
#  tar xzf starpu-1.3.7.tar.gz
#  (
#    cd starpu-1.3.7 || exit
#    ./configure --prefix=$root/starpu/debug ${HWLOC} ${SPU_FLAGS} --enable-debug
#    make -j ${NUM_THREADS}
#    make install
#  )
#  rm -rf starpu-1.3.7
#
#  # debug-fxt
##  tar xzf starpu-1.3.7.tar.gz
##  (
##    cd starpu-1.3.7 || exit
##    ./configure --prefix=$root/starpu/debug-fxt "$HWLOC"  ${HWLOC} ${SPU_FLAGS} --enable-debug --with-fxt
##    make -j ${NUM_THREADS}
##    make install
##  )
##  rm -rf starpu-1.3.7
#
#  # release
#  tar xzf starpu-1.3.7.tar.gz
#  (
#    cd starpu-1.3.7 || exit
#    ./configure --prefix=$root/starpu/release ${HWLOC} ${SPU_FLAGS}
#    make -j ${NUM_THREADS}
#    make install
#  )
#  rm -rf starpu-1.3.7
#
#  # release-fxt
##  tar xzf starpu-1.3.7.tar.gz
##  (
##    cd starpu-1.3.7 || exit
##    ./configure --prefix=$root/starpu/release-fxt ${HWLOC} ${SPU_FLAGS} --with-fxt
##    make -j ${NUM_THREADS}
##    make install
##  )
##  rm -rf starpu-1.3.7
#)


(
  cd src || exit

  # Download StarPU fork for if was not downloaded
  if [[ ! -f starpu-srki-fork.tar.gz ]]; then
    if [[ $(uname) == 'Darwin' ]]; then
       curl -O https://github.com/srki/starpu/archive/master.zip
    else
       wget https://github.com/srki/starpu/archive/master.zip
    fi

    unzip -q master.zip
    rm master.zip
    mv starpu-master starpu-srki-fork
    tar -czf starpu-srki-fork.tar.gz starpu-srki-fork
  fi

  if [[ ! -f starpu-srki-fork.tar.gz ]]; then
    echo "Cannot Download StarPU srki fork"
    exit
  fi

  # fork-debug
  rm -rf starpu-srki-fork-debug
  mkdir starpu-srki-fork-debug
  tar -xf starpu-srki-fork.tar.gz -C starpu-srki-fork-debug --strip-components=1
  (
    cd starpu-srki-fork-debug || exit
    ./configure --prefix=$root/starpu/debug-srki-fork ${HWLOC} ${SPU_FLAGS} --enable-debug
    if [[ "$HOSTNAME" ==  "ancilla" ]] || [[ -n "$NERSC_HOST" ]]; then
      autoreconf -f -i
    fi
    make -j ${NUM_THREADS}
    make install
  )

  # fork-release
  rm -rf starpu-srki-fork-release
  mkdir starpu-srki-fork-release
  tar -xf starpu-srki-fork.tar.gz -C starpu-srki-fork-release --strip-components=1
  (
    cd starpu-srki-fork-release || exit
    ./configure --prefix=$root/starpu/release-srki-fork ${HWLOC} ${SPU_FLAGS}
    if [[ "$HOSTNAME" ==  "ancilla" ]] || [[ -n "$NERSC_HOST" ]]; then
      autoreconf -f -i
    fi
    make -j ${NUM_THREADS}
    make install
  )
)