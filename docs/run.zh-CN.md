# 运行服务端

[English](run.md) | 简体中文 | [返回 README](../README_zh.md)

## 运行服务端

```bash
./build/rwkv_lighting_cuda \
  --model-path /path/to/model.pth \
  --vocab-path /path/to/rwkv_vocab_v20230424.txt \
  --host 127.0.0.1 \
  --port 8000 \
  --chunk-size 128 \
  --chunk-load
```

`--chunk-size` 控制 prompt prefill 的分块大小，缺省为 `128`。
`--state-db-path` 缺省为当前工作目录下的 `rwkv_sessions.db`。
W8A16 调优缓存属于服务实例本地的状态：除非显式传入 `--tune-cache`，否则它默认与
state 数据库存放在同一目录（同样是当前工作目录）。部署时请把缓存和 state 数据库放在
一起，或显式指定缓存路径。
`--chunk-load` 避免在 CUDA 上传前把完整 `.pth` 文件或完整大张量一次性读进主机内存。
它使用常驻的模型文件流和两块可复用的 32 MiB pinned 缓冲，让磁盘读取、CUDA 拷贝和
预处理三者重叠；每四个完整 transformer 层作为一个批次上传并完成定稿，减少加载期的
同步等待。在 Linux 上，每次读取后已消费的文件缓存页会被标记为可回收。省略该 flag 则
保持原来整文件加载的行为。
生成请求进入 FIFO 准入队列。服务端会根据空闲显存动态刷新可用的 prefill 批大小上限，
并在有容量时放行请求。`/v1/server/status` 会报告 `prefill_queue` 和所有
`active_requests`，同时保留 `active_request` 字段以兼容旧客户端。

服务端默认绑定 `127.0.0.1`。仅在你明确希望监听所有 IPv4 接口时才使用
`--host 0.0.0.0`。

## Windows 运行

```powershell
cd .\build\bundle\rwkv_lighting_cuda
$env:PATH = "$PWD\lib;$env:PATH"
.\rwkv_lighting_cuda.exe `
  --model-path C:\models\model.pth `
  --vocab-path .\rwkv_vocab_v20230424.txt `
  --host 127.0.0.1 `
  --port 8000
```

## 动态模型加载

默认情况下 `--model-path` 仍指向单个 `.pth` 或 `.rwkvq` 文件，原有单模型启动行为
不变。若要按需加载模型，把它改为目录并加上 `--enable-dynamic-loading`：

```bash
./build/rwkv_lighting_cuda \
  --model-path /path/to/models \
  --enable-dynamic-loading \
  --chunk-load \
  --vocab-path /path/to/rwkv_vocab_v20230424.txt
```

目录顶层 `.pth` 与 `.rwkvq` 文件会通过 `GET /v1/models` 暴露，去掉扩展名的文件名就是
模型 ID。响应会标出当前 `loaded` 的模型与所有 `available` 模型。显式加载或切换模型：

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/model/load" \
  -H "Content-Type: application/json" \
  --data '{"model":"rwkv7-g1i-7.2b-20260805-ctx16384"}'
```

推理请求始终使用已加载的模型：请求中的 `model` 字段仅为兼容 OpenAI 而保留，不会触发
加载或切换。并发推理共享该已加载模型。不同加载请求按 FIFO 排队；切换会等待在途推理
结束，从显存释放旧模型，然后加载所选模型。

## 独立 state 调优（CUDA）

在 `RWKV7_STATE_TUNING=ON`（CUDA 构建默认开启）时，独立的 `rwkv_state_tune` 二进制
只训练 `blocks.N.att.time_state`，输入是形如 `{"text":"..."}` 的 JSONL 行：

```bash
./build/rwkv_state_tune \
  --model /path/to/model.pth \
  --data /path/to/train.jsonl \
  --output ./state_output \
  --ctx 512 \
  --chunk 128 \
  --epochs 1 \
  --max-steps 10000 \
  --lr 0.0005 \
  --lr-final 0.0005 \
  --warmup-steps 10 \
  --save-every 100 \
  --batch-size 8
```

这个以正确性为先的版本使用现有 BF16 PTH 加载器及其 FP16 运行时权重，不支持 INT8
训练。样本会在 `--ctx` 处截断；`--chunk` 控制 checkpoint/recompute 长度并做反向 state
梯度传播。`--batch-size N` 在每次优化器更新中累积 N 条变长样本。checkpoint 只包含
state 张量，可直接上传到现有推理后端使用。实现细节见
[状态微调文档](../src/state_tuning/README.zh-CN.md)。

## MiSS adapter 训练与推理

同一个构建还会生成 `rwkv_miss_tune`。它冻结基模，只训练所选 attention 和 FFN
Linear 对应的 MiSS `D[out,rank]`，支持分块 state passing、梯度累积、可续训的周期
checkpoint，以及与训练 checkpoint 分离的推理导出：

```bash
./build/rwkv_miss_tune \
  --model /path/to/model.pth \
  --data /path/to/train.jsonl \
  --output ./miss_output \
  --vocab ./assets/rwkv_vocab_v20230424.txt \
  --rank 16 --alpha 16 --targets all \
  --ctx 4096 --chunk 1024 --batch-size 8 --epochs 1 \
  --lr 0.0001 --lr-final 0.00001 --warmup-steps 10 \
  --save-every 100 --wkv_tape
```

通过 `POST /v1/adapters` 上传或注册 `miss_output/adapter-final.pth`
（也可使用 checkpoint 的 `training.pth`），然后在生成请求中传入
`adapter_id`，并按需指定 `adapter_version` 或 `adapter_scale`。注册阶段只在 CPU
RAM 中校验和缓存；第一次请求时才把整个 adapter 一次性上传到 GPU。checkpoint
续训、推理包身份、缓存预算、性能分析和验收说明见
[MiSS 文档](../src/miss/README.zh-CN.md)。
