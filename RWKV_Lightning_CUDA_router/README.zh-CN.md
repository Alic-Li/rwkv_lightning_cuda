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

## Adapter 全后端同步

`/v1/adapters` 的 `POST`（multipart 上传或 JSON 路径注册）、`GET` 和 `DELETE`
均并发发送到每个配置后端，包括处于连接冷却中的节点。只有所有后端注册成功，且返回
相同的 `adapter_id` 和内容版本 `version`，上传才返回成功。JSON 的 `path` 必须在每个
后端机器上都可读；不同机器推荐直接 multipart 上传，同一份 PTH 和可选的旧格式
`checkpoint.json` 会原样发给所有后端：

```bash
curl -sS http://127.0.0.1:8080/v1/adapters \
  -F 'adapter_id=miss-r16-lr3e-5' \
  -F 'file=@./rwkv7-g1k-7b-r16-lr3e-5-e2-miss.pth'
```

需要认证时添加 `-H 'Authorization: Bearer ...'`。上传后在生成请求中使用该
`adapter_id`；推荐同时带上上传响应中的 `version`，作为 `adapter_version`。
Router 会将省略版本的请求固定到它确认的当前内容版本，避免后端各自的 latest 不一致。
`adapter_scale`、`stop`、SSE 的 `matched_stop` 等请求和响应字段正常透传。

- 上传默认最多尝试 3 次，每次最长 120 秒；只重试失败节点，重发相同 ID 和相同请求体。
  连接/响应读取失败、超时和 HTTP 408/429/500/502/503/504 可重试，确定性 4xx 不重试。
  通过 `adapter_upload_*` 独立配置，支持 `Retry-After`、总请求期限和客户端取消。
- 部分失败返回 HTTP 502 JSON，含 `adapter_id`、已确认的 `version`、`ready_backends`
  和 `failed_backends`。成功副本不回滚；推理仅发往已确认有所需版本的健康节点。
  同时使用 State 时，必须在同一节点同时具备该 State 和 adapter；没有满足条件的节点
  返回 HTTP 503，不会去掉 adapter/State 后继续推理。推理请求本身不会自动重试。
- 同一 ID 的上传/删除期间暂不接收该 ID 的新推理，同 ID 的并发修改返回 HTTP 409。
  删除广播到所有后端；不确定的删除结果保持对应 ID/版本禁用，避免残留副本重新接收流量。
  所有后端明确拒绝修改（例如认证失败）时恢复原有就绪记录。
- 列表比较所有后端的 `(id, version, manifest)`，忽略条目顺序和各节点不同的缓存统计；
  不一致时返回 HTTP 502，不会随机展示某个节点的列表。成功响应的统计来自第一个后端，
  不是全局合计。过期列表不能覆盖正在发生的上传/删除。
- Router 重启或遇到未知 adapter 后，会用最多 5 秒查询所有后端列表并恢复版本就绪记录。
  旧后端列表不提供 latest 信息；同一 ID 仅有一个已发现版本时可自动使用，有多个版本时
  请显式提供 `adapter_version`，或通过 Router 重新注册所需版本。

广播不是分布式事务，也不在返回错误后无限后台补传。需要重新同步时，用相同 ID 重新上传，
随后再次检查 Router 列表。Cloudflare 在 Router 外层返回的 502/超时仍可能发生在实际上传
成功之后，应通过 Router 的全后端列表确认结果，不能仅凭客户端收到错误断定文件未上传。
State 也必须通过上述 `/v1/state/*` 广播接口上传，旧 Router 已只上传到某个节点的内容
不会自动复制到其他机器。

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
