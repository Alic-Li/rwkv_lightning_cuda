# State 上传与权重兼容性

## 请求路径

- 浏览器文件：Client `/api/v1/backends/{id}/v1/state/upload` → Agent 代理 → C++ 推理层。
- 节点文件：Agent `/api/v1/runtime/state/import` 校验文件系统白名单、`.pth`、非空与
  512 MiB 上限，流式封装 multipart，再发到本地运行时；运行时原始状态码和元信息直接返回。
- 多进程网关：`RWKV_Lightning_CUDA_router` 把 upload/list/delete 广播到所有推理进程。
  UUID 必须在广播前生成一次，各进程使用同一个 `X-RWKV-State-Upload-UUID`。
  直连推理层上传则由 C++ 用 OpenSSL 随机源独立生成 UUID-v7。

UUID-v7 按 [RFC 9562 §5.7](https://www.rfc-editor.org/rfc/rfc9562.html#section-5.7)
组合 48 位 Unix 毫秒时间、版本/variant 位和 74 位随机数；同毫秒不承诺单调顺序。
文件实际存储名与 API ID 一致，不保留 `.pth` 后缀；格式识别依赖内容。
`original_filename` 保存原 basename，不把客户端路径带入存储目录。
例如 `state01.pth` → `state01-01a1086e-6d3a-744f-b79d-7e67754de680`。

## 列表字段

| 字段 | 含义 |
|---|---|
| `state_id` / `filename` | 唯一 ID / 实际存储文件名 |
| `original_filename` | 上传时的原文件名 |
| `size_bytes` | PTH 文件大小，字节 |
| `tensor_count` / `layers` | 有效 `blocks.N.att.time_state` 张量数 / 层数 |
| `heads` / `head_size` | 每层张量形状 `[heads, head_size, head_size]` |
| `created` | 兼容旧客户端的 Unix 秒 |
| `created_ms` | Unix 毫秒，列表从新到旧排序 |
| `uploaded_at` | RFC 3339 UTC 时间字符串，便于直接阅读 |

网关检查各进程的列表 ID、大小与结构一致才返回列表；时间戳允许不同。
前端展示原文件名、独立 ID、文件大小、张量数和本地化上传时间。

## 与权重怎么对应

**上传的初始 State 没有绑定训练时的权重哈希。结构通过只表示可加载，不保证效果。**

上传先校验 PTH 结构：层编号从 0 连续、没有重复层、形状一致且为正的三维方阵、
dtype 为 BF16 或 FP32、存储大小与 stride 范围合法。其他非 `time_state` 张量会忽略。
上传阶段不依赖当前模型，也不把 State 声称为“匹配当前权重”。

实际推理时，`create_request_state` 用返回的 `state_id` 找到文件，再调用
`ModelBackend::load_state_from_pth` 对照当前加载权重的维度（CUDA / HIP 实现一致）：

1. 恰好具有当前模型的全部层 `blocks.0` 到 `blocks.(layers-1)`，不能出现额外层。
2. 每层形状必须为 `[dims.heads, dims.head_size, dims.head_size]`。
3. dtype 必须为 BF16 或 FP32；加载时转为配置的 WKV FP16/FP32，转置最后两维，
   并复制到请求中每个 batch 的状态。shift 等其他初始状态仍由 `create_state` 初始化。

同结构但不同训练 checkpoint 的权重仍可能加载此 State。要确认训练来源，应保存并
核对原始权重 SHA-256；当前上传协议/State 导出没有提供强制的来源哈希绑定。
BF16 权重与由它量化产生的权重在相同结构下可以通过以上检查，量化后的实际效果需验证。

另一个机制是**生成后的会话缓存身份**：模型运行身份、MiSS adapter、初始 State 文件
指纹和 WKV 精度会参与 `effective_key`，已推进的缓存遇到身份变化时拒绝复用。
这用于避免缓存串用，不能反过来证明一个新上传的 State 由当前权重训练。
MiSS adapter 自身还会校验 `base_fingerprint`，其约束比初始 State 的结构检查更强。

## 生命周期

上传 State 登记在进程内的 map、文件保存在临时目录，尚未持久化到数据库；进程退出
会清理。SQLite 会话缓存是另一条存储路径。UUID 不改变上述生命周期。
网关广播也不是分布式事务；一个进程失败时其他进程可能已接收，list 会报告进程间差异。网关对失败节点进行有限
次数的指数退避重试，同 UUID / 同内容由推理层 SHA-256 校验并幂等返回原记录；不同
内容拒绝覆盖。带 State 的推理只发往已确认就绪的健康副本；冷却结束不会自动解除
State 未确认状态。部分成功返回 502 并携带 `state_id` 与节点结果，没有就绪副本时
推理返回 503。网关重启或遇到未知 ID 时先查询节点 list，恢复副本确认记录。
没有错误返回后的无限后台补传；详细配置见 [网关说明](../RWKV_Lightning_CUDA_router/README.zh-CN.md)。
所有运行时需与新网关一起升级。
