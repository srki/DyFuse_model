#!/usr/bin/env bash
NUM_THREADS=${NUM_THREADS:-"-j 12"}

root=$PWD

version="5.1.5"

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
if [[ ! -d src ]]; then
  mkdir src
fi

(
  cd src || exit

  if [[ ! -f v${version}.tar.gz ]]; then
    if [[ $(uname) == 'Darwin' ]]; then
#      curl -O https://github.com/DrTimothyAldenDavis/GraphBLAS/archive/v${version}.tar.gz
#      curl -O https://github.com/DrTimothyAldenDavis/GraphBLAS/archive/refs/tags/v5.1.5.tar.gz
      curl -O https://github.com/srki/GraphBLAS/archive/stable.zip
    else
#      wget https://github.com/DrTimothyAldenDavis/GraphBLAS/archive/v${version}.tar.gz
#      wget https://github.com/DrTimothyAldenDavis/GraphBLAS/archive/refs/tags/v5.1.5.tar.gz
       wget https://github.com/srki/GraphBLAS/archive/stable.zip
    fi

    unzip -q stable.zip
    rm stable.zip
    mv GraphBLAS-stable GraphBLAS-${version}
    tar -czf v${version}.tar.gz GraphBLAS-${version}
  fi

 # # Debug
 # rm -rf GraphBLAS-${version}-debug
 # tar -xf v${version}.tar.gz
 # mv GraphBLAS-${version} GraphBLAS-${version}-debug
 # (
 #   cd GraphBLAS-${version}-debug || exit
 #   cp CMakeLists.txt CMakeLists.txt.original

 #   sed 's/set *(* GB_BURBLE false *)*/set ( GB_BURBLE true )/'  CMakeLists.txt.original > CMakeLists.txt

 #   (
 #     cd build || exit
 #     cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=${root}/GraphBLAS/debug -DCMAKE_EXE_LINKER_FLAGS=-dynamic
 #     make -j ${NUM_THREADS}
 #     make install
 #   )
 # )

 # # Debug serial
 # rm -rf GraphBLAS-${version}-debug-serial
 # tar -xf v${version}.tar.gz
 # mv GraphBLAS-${version} GraphBLAS-${version}-debug-serial
 # (
 #   cd GraphBLAS-${version}-debug-serial || exit
 #   cp CMakeLists.txt CMakeLists.txt.original

 #   sed 's/include *( *FindOpenMP *)/#include ( FindOpenMP )/' CMakeLists.txt.original > CMakeLists.txt.1
 #   sed 's/set *(* GB_BURBLE false *)*/set ( GB_BURBLE true )/'  CMakeLists.txt.1 > CMakeLists.txt

 #   (
 #     cd build || exit
 #     cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=${root}/GraphBLAS/debug-serial -DCMAKE_EXE_LINKER_FLAGS=-dynamic
 #     make -j ${NUM_THREADS}
 #     make install
 #   )
 # )

 # # RelWithDebInfo
 # rm -rf GraphBLAS-${version}-relwithdebinfo
 # tar -xf v${version}.tar.gz
 # mv GraphBLAS-${version} GraphBLAS-${version}-relwithdebinfo
 # (
 #   cd GraphBLAS-${version}-relwithdebinfo || exit
 #   cp CMakeLists.txt CMakeLists.txt.original

 #   (
 #     cd build || exit
 #     cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=${root}/GraphBLAS/relwithdebinfo -DCMAKE_EXE_LINKER_FLAGS=-dynamic
 #     make -j ${NUM_THREADS}
 #     make install
 #   )
 # )

 # # RelWithDebInfo serial
 # rm -rf GraphBLAS-${version}-relwithdebinfo-serial
 # tar -xf v${version}.tar.gz
 # mv GraphBLAS-${version} GraphBLAS-${version}-relwithdebinfo-serial
 # (
 #   cd GraphBLAS-${version}-relwithdebinfo-serial || exit
 #   cp CMakeLists.txt CMakeLists.txt.original

 #   sed 's/include *( *FindOpenMP *)/#include ( FindOpenMP )/' CMakeLists.txt.original > CMakeLists.txt

 #   (
 #     cd build || exit
 #     cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=${root}/GraphBLAS/relwithdebinfo-serial -DCMAKE_EXE_LINKER_FLAGS=-dynamic
 #     make -j ${NUM_THREADS}
 #     make install
 #   )
 # )

  # Release
  rm -rf GraphBLAS-${version}-release
  tar -xf v${version}.tar.gz
  mv GraphBLAS-${version} GraphBLAS-${version}-release
  (
    cd GraphBLAS-${version}-release || exit
    cp CMakeLists.txt CMakeLists.txt.original

    (
      cd build || exit
      cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${root}/GraphBLAS/release -DCMAKE_EXE_LINKER_FLAGS=-dynamic
      make -j ${NUM_THREADS}
      make install
    )
  )

#  # Release serial
#  rm -rf GraphBLAS-${version}-release-serial
#  tar -xf v${version}.tar.gz
#  mv GraphBLAS-${version} GraphBLAS-${version}-release-serial
#  (
#    cd GraphBLAS-${version}-release-serial || exit
#    cp CMakeLists.txt CMakeLists.txt.original
#
#    sed 's/include *( *FindOpenMP *)/#include ( FindOpenMP )/' CMakeLists.txt.original > CMakeLists.txt
#
#    (
#      cd build || exit
#      cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${root}/GraphBLAS/release-serial -DCMAKE_EXE_LINKER_FLAGS=-dynamic
#      make -j ${NUM_THREADS}
#      make install
#    )
#  )

#  # Release BURBLE
#  rm -rf GraphBLAS-${version}-release-burble
#  tar -xf v${version}.tar.gz
#  mv GraphBLAS-${version} GraphBLAS-${version}-release-burble
#  (
#    cd GraphBLAS-${version}-release-burble || exit
#    cp CMakeLists.txt CMakeLists.txt.original
#
#    sed 's/set *(* GB_BURBLE false *)*/set ( GB_BURBLE true )/'  CMakeLists.txt.original > CMakeLists.txt
#
#    (
#      cd build || exit
#      cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${root}/GraphBLAS/release-burble -DCMAKE_EXE_LINKER_FLAGS=-dynamic
#      make -j ${NUM_THREADS}
#      make install
#    )
#  )
#
#  # Serial Release BURBLE
#  rm -rf GraphBLAS-${version}-release-burble-serial
#  tar -xf v${version}.tar.gz
#  mv GraphBLAS-${version} GraphBLAS-${version}-release-burble-serial
#  (
#    cd GraphBLAS-${version}-release-burble-serial || exit
#    cp CMakeLists.txt CMakeLists.txt.original
#
#    sed 's/include *( *FindOpenMP *)/#include ( FindOpenMP )/' CMakeLists.txt.original > CMakeLists.txt.1
#    sed 's/set *(* GB_BURBLE false *)*/set ( GB_BURBLE true )/'  CMakeLists.txt.1 > CMakeLists.txt
#
#    (
#      cd build || exit
#      cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${root}/GraphBLAS/release-burble-serial -DCMAKE_EXE_LINKER_FLAGS=-dynamic
#      make -j ${NUM_THREADS}
#      make install
#    )
#  )
#
#  for build_type in "debug" "debug-serial" "relwithdebinfo" "relwithdebinfo-serial" "release" "release-serial" "release-burble" "release-burble-serial"; do
#    if [[ -e ${root}/GraphBLAS/${build_type}/lib64 ]]; then
#      mv ${root}/GraphBLAS/${build_type}/lib64 ${root}/GraphBLAS/${build_type}/lib
#    fi
#  done
)