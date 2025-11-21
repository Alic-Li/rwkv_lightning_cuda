nvcc -o test_wkv test/test_wkv.cpp src/wkv/wkv.cu \
    -I./src \
    -O3 \
    --use_fast_math \
    -Xptxas -O3 \
    --extra-device-vectorization \
    -DHEAD_SIZE=64 \
    -arch=sm_89 \
    -std=c++17

nvcc -o test_linear test/test_linear.cpp src/module/linear_cublas.cu \
    -I./src \
    -O3 \
    --use_fast_math \
    -Xptxas -O3 \
    -arch=sm_89 \
    -std=c++17 \
    -lcublas \
    --extra-device-vectorization

nvcc -o test_tensor_utils test/test_tensor_utils.cpp src/module/tensor_utils.cu \
    -I./src \
    -O3 \
    --use_fast_math \
    -Xptxas -O3 \
    --extra-device-vectorization \
    -arch=sm_89 \
    -std=c++17

nvcc -o test_load_model test/test_load_model.cpp src/utils/load_model.cu \
    -I./src \
    -O3 \
    --use_fast_math \
    -Xptxas -O3 \
    --extra-device-vectorization \
    -arch=sm_89 \
    -std=c++17

nvcc -o test_sampler test/test_sampler.cpp src/utils/sampler.cu \
    -I./src \
    -O3 \
    --use_fast_math \
    -Xptxas -O3 \
    --extra-device-vectorization \
    -arch=sm_89 \
    -std=c++17

nvcc -o test_norm test/test_norm.cpp src/module/norm.cu \
    -I./src \
    -O3 \
    --use_fast_math \
    -Xptxas -O3 \
    --extra-device-vectorization \
    -arch=sm_89 \
    -std=c++17

nvcc -o test_tokenizer test/test_tokenizer.cpp src/utils/tokenizer.cu \
    -I./src \
    -O3 \
    -std=c++17

nvcc -o test_Tmix test/test_Tmix.cpp \
    src/Tmix/Tmix.cu \
    src/module/linear_cublas.cu \
    src/module/norm.cu \
    src/module/tensor_utils.cu \
    src/wkv/wkv.cu \
    -I./src \
    -O3 \
    --use_fast_math \
    -Xptxas -O3 \
    --extra-device-vectorization \
    -DHEAD_SIZE=64 \
    -arch=sm_89 \
    -std=c++17 \
    -lcublas

nvcc -o test_Cmix test/test_Cmix.cpp \
    src/Cmix/Cmix.cu \
    src/module/linear_cublas.cu \
    src/spmv/spmv.cu \
    -I./src \
    -O3 \
    --use_fast_math \
    -Xptxas -O3 \
    --extra-device-vectorization \
    -arch=sm_89 \
    -std=c++17 \
    -lcublas

nvcc -o benchmark_rwkv benchmark/benchmark.cpp \
    src/rwkv7.cu \
    src/Tmix/Tmix.cu \
    src/Cmix/Cmix.cu \
    src/module/linear_cublas.cu \
    src/module/norm.cu \
    src/module/tensor_utils.cu \
    src/wkv/wkv.cu \
    src/spmv/spmv.cu \
    src/utils/load_model.cu \
    src/utils/sampler.cu \
    src/utils/tokenizer.cu \
    -I./src \
    -O3 \
    --use_fast_math \
    -Xptxas -O3 \
    --extra-device-vectorization \
    -DHEAD_SIZE=64 \
    -arch=sm_89 \
    -std=c++17 \
    -lcublas
