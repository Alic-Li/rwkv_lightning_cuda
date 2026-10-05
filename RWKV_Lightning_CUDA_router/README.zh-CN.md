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
路由器等待全部响应。上传时由网关生成一个 UUID-v7，通过
`X-RWKV-State-Upload-UUID` 传给所有推理进程，统一创建 `<原文件名去扩展名>-<UUID>` ID，
后续携带此 ID 的推理请求仍可安全地负载均衡。如果某个后端失败，或后端返回的状态/ID
不一致，路由器会返回错误，不会宣称状态已同步。列表会比对各进程的 ID、字节大小
与张量结构（时间和顺序允许不同）。广播上传不是分布式事务，失败时成功进程可能留有
文件。网关与所有推理进程需一起升级；直连推理层会独立生成 UUID，因此负载均衡场景
应通过网关上传。示例请求体上限为 513 MiB，
用于代理后端允许的 512 MiB 状态上传及 multipart 封装开销。
接口用法见 [HTTP API 中文文档](../docs/http-api.zh-CN.md#初始状态文件上传列表与删除)。

推理 POST 不会自动重试，避免生成执行两次。State 上传是专门的幂等例外：

- 默认最多 3 次尝试，每次最长 120 秒；失败节点按指数退避并加抖动重试，基础等待
  250 ms、最大 2000 ms，遵守 `Retry-After`，受总操作期限和客户端取消约束。
  配置见 `state_upload_*`；成功节点不重复上传。
- 重试限于连接/读响应失败、超时、HTTP 408/429/500/502/503/504。
  非法 PTH、认证等确定性 4xx 不重试。
- 同一次上传始终使用相同 UUID 和相同内容。推理层用 SHA-256 校验：同 UUID、
  同原文件名、同大小且同内容返回原记录（原上传时间不变），不同内容拒绝覆盖。
- 调度额外记录每个 State 在每个节点是否已确认就绪。上传中、上传失败或响应不确定
  的节点不接收对应 State 的推理，即使连接冷却已结束或会话原先绑定在该节点。
  已确认的健康副本可继续使用；没有可用副本时返回 HTTP 503，不回退为零 State。
- 重试耗尽后的部分成功返回 HTTP 502 JSON，含 `state_id`、`ready_backends` 和
  `failed_backends`（节点名和最终 HTTP 状态，0 表示未取得状态码）。全部节点明确
  拒绝时保留推理层的验证/认证错误；全部成功仍返回原有上传响应。
- 当前重试只在上传请求生命周期内执行，不在返回错误后无限后台补传。失败节点必须
  经同次上传重试成功或后续 list 确认已有该文件才恢复对应 State 流量。
  未知 ID / 网关重启后的首次 State 请求会以最多 5 秒查询各节点 list，按确认的副本
  路由。list 依然要求所有节点一致才向客户端返回成功。
- 删除开始后停止新请求使用该 State；不确定的删除结果保持禁用。全部节点明确返回
  验证/认证拒绝时恢复原就绪记录；过期的列表响应不会覆盖上传或删除的结果。

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
