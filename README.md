## Build

```bash
cmake -S . -B ./build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;87;89;90;100;120" \
  -DRWKV7_FAST_V4_STATIC_LINK=ON

cmake --build ./build -j
```
Only Compile Benchmark Compile

```bash
cmake --build ./build -j --target benchmark
```
Only Compile Server Backend

```bash
cmake --build ./build -j --target rwkv_lighting_cuda
```
If Want to Compile GUI

```bash
mkdir ./third_party; cd ./third_party; git clone https://github.com/ocornut/imgui.git;
cmake --build ./build -j
```
## Run

Run benchmark

```bash
./build/benchmark \
  --model /dev/shm/rwkv7-g1f-7.2b-20260414-ctx8192.pth \
  --model-forward \
  --cases '1x1,1x2,1x4,1x8,1x16,1x32,1x64,1x128,1x256,2x1,4x1,8x1,16x1,32x1,64x1,128x1,256x1,2x2,4x4,8x8,16x16' \
  --graph-bench \
  --warmup 3 \
  --iters 10
```

Run server

```bash
./build/rwkv_lighting_cuda \
  --model-path /path/to/model.pth \
  --vocab-path /path/to/rwkv_vocab_v20230424.txt \
  --port 8000
```
