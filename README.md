
## Build instructions
``` bash
source cori_modules.sh
cd external/
chmod +x install-*
NUM_THREADS=64 ./install-starpu.sh
NUM_THREADS=64 ./install-GraphBLAS.sh
cd ..
mkdir build
cd build/
mkdir release
cd release/
CC=clang CXX=clang++ cmake ../.. -DCMAKE_BUILD_TYPE=Release -DSTARPU_HOME=$PWD/../../external/starpu/release-srki-fork
make -j
```

above are forked from https://github.com/srki/grb-fusion.git
# DyFuse_model
# run with scheduler:
build simple_model


#model
1. process train data (optional... oringal files are too large to git here)
*  cd model
*  python process_traindata_for_MLP.py
*  ls data/ancilla_classification_5/
data/ancilla_classification_5/1D_thres_faster_base64_class_16_train_data_tri.csv
data/ancilla_classification_5/1D_thres_faster_base64_class_1_train_data_tri.csv
data/ancilla_classification_5/1D_thres_faster_base64_class_32_train_data_tri.csv
data/ancilla_classification_5/1D_thres_faster_base64_class_64_train_data_tri.csv
data/ancilla_classification_5/1D_thres_faster_base64_class_8_train_data_tri.csv

2. python tricnt_mxm_w_mask.py (This includes KNN, regression, and searching for schedule order)
Regression models are saved in kernel/ancilla/linear*_1D_thres_faster_base64_class_*_train_data_tri.csv.pkl

3. ouputs are under model/res/ancilla/
each matrix has two files, e.g.,
1. cat model/res/ancilla/new_tri_block_roadNet-CA.mtx_1013.csv 
1,2 // means using 2 input blocks. 1D blocking. 
2. cat model/res/ancilla/new_tri_roadNet-CA.mtx_1013.csv // 2 input blocks => 4 output blocks
0,32 // block 0 uses 32 threads
3,32  // block 3 uses 32 threads
99999,99999 // scheduler wait for all, and then split worker
1,1 // block 1 uses 1 thread


4. the above two files are the input for DyFuse, e.g., apps/simple_model/mxm_schedule.cpp
