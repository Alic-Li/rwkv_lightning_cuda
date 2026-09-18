# RWKV Lightning CUDA 路由器

[English](README.md) | 简体中文

这是一个 HTTP 反向代理，按正在处理的批大小而非请求数，对 RWKV 推理服务器进行负载均衡。
所有方法、请求头、响应状态和 SSE 流均原样转发。

## 运行

```bash
cp config.example.toml config.toml
go run . -config config.toml
```

用 `go build .` 构建二进制文件。

`config.toml` 必须包含 `listen` 地址和一个或多个 `[[backends]]`。
每个后端的 `url` 为 HTTP/HTTPS URL，`weight` 表示相对容量，默认为 1。

## 调度

对每个代理请求，路由器从第一个适用字段确定 bsz：先依次检查 `contents`、`text_list`、
`prompts`、`inputs` 数组，再检查数值 `bsz` 或 `batch_size`，否则取 1。
选择以下值最小的健康后端：

```
in_flight_bsz / weight
```

数值相同时公平轮转。在复制普通响应或 SSE 流期间，包括后端排队时间，bsz 始终保持占用；
完成、上游出错或客户端取消时释放。这样小请求可以利用剩余容量，同时避免一个大批次
压垮单个 GPU。

传输失败只会将受影响后端标记为不可用，持续 `failure_cooldown_seconds`。
HTTP 错误响应会直接转发，不会据此标记后端不健康，因为它们可能是正常的请求错误。

对于请求体中有 `session_id` 或带 `X-RWKV-Session-Id` 请求头的 `/state/*` 请求，
路由器会在进程生命周期内尽力维护内存中的会话亲和映射。除非所有后端共享状态存储，
否则不能指望有状态流量在路由器重启后仍保持会话。

`/v1/state/upload`、`/v1/state/list` 和 `/v1/state/delete` 会并发发送到所有配置的后端，
路由器等待全部响应。这样上传会在每个工作进程创建相同的、基于文件名的 `state_id`，
后续携带此 ID 的推理请求仍可安全地负载均衡。如果某个后端失败，或后端返回的状态/ID
不一致，路由器会返回错误，不会宣称状态已同步。示例请求体上限为 513 MiB，
用于代理后端允许的 512 MiB 状态上传及 multipart 封装开销。
接口用法见 [HTTP API 中文文档](../docs/http-api.zh-CN.md#初始状态文件上传列表与删除)。

路由器不会重试 POST 请求：上游写入结果不明确时重试，可能导致生成执行两次。
客户端可以安全重试未建立连接的失败。

## 批量负载测试

附带命令在每个指定的请求并发度下发送非流式 `/v1/batch/completions` 请求，报告最高
持续提示词（样本）吞吐量。通过参数或对应的 `RWKV_CF_ACCESS_CLIENT_ID` 和
`RWKV_CF_ACCESS_CLIENT_SECRET` 环境变量提供 Cloudflare Access 凭据：

```bash
go run ./cmd/rwkv-batch-loadtest \
  -url 'https://api-7b.rwkvos.com/v1/batch/completions' \
  -cf-client-id 'XXX' \
  -cf-client-secret 'XXX' \
  -batch-size 8 \
  -concurrency '1,2,4,8,16,32' \
  -duration 30s \
  -max-tokens 128
```

`samples/s` 为每秒完成的提示词数（`成功请求数/s × batch-size`）。
应使用与目标工作负载相同的提示词、批大小和 `max_tokens`，否则得到的上限不可比较。
命令不会打印凭据。
