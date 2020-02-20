export TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
export TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')


cd interpolation_3d/

g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I$TF_INC -I/usr/local/cuda-10.0/include -I$TF_INC/external/nsync/public -lcudart -L/usr/local/cuda-10.0/lib64/ -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0


cd ..

cd grouping/

/usr/local/cuda-10.0/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I$TF_INC -I/usr/local/cuda-10.0/include -I$TF_INC/external/nsync/public -lcudart -L/usr/local/cuda-10.0/lib64/ -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0


cd ..

cd sampling/

/usr/local/cuda-10.0/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I$TF_INC -I/usr/local/cuda-10.0/include -I$TF_INC/external/nsync/public -lcudart -L/usr/local/cuda-10.0/lib64/ -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
