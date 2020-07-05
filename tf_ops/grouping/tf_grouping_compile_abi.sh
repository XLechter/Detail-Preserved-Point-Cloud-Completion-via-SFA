#!/usr/bin/env bash
nvcc=/usr/local/cuda-10.0/bin/nvcc
cudainc=/usr/local/cuda-10.0/include/
cudalib=/usr/local/cuda-10.0/lib64/
TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

g++ tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -std=c++11 -shared -fPIC -I $TF_INC \
-I$TF_INC/external/nsync/public -I $cudainc -L$TF_LIB -ltensorflow_framework -lcudart -L $cudalib -O2 -D_GLIBCXX_USE_CXX11_ABI=1
