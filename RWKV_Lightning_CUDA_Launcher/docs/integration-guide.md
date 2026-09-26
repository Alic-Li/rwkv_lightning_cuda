# 前端开发与第三方 API 联调指南

本文描述当前 Launcher 的可用接口；新前端应从这里开始。完整控制面字段见 [控制面 API](control-plane-api.md)，原生推理字段见 [原生 API 参考](../../rwkv_lightning_api_doc.md) 与 [curl 示例](../../docs/http-api.zh-CN.md)。工程启动、Bun 和 Tailwind CSS 见 [前端开发指南](frontend-development.md)。

## 1. 选择接入方式

| 接入方 | Base URL 示例 | 鉴权 | 可用接口 |
| --- | --- | --- | --- |
| WebUI / 浏览器 | 同源相对地址 `/api/v1/backends/{id}` | 远端 token 由 Client 注入；浏览器不持久化它 | 目标节点的 `/api/v1/*` 与 `/v1/*` |
| 第三方脚本管理 Agent | `http://server:18766` | `Authorization: Bearer <agent-token>` | Agent `/api/v1/*`、代理的 `/v1/*` |
| 第三方经本机 Client 管理远端 | `http://127.0.0.1:10721/api/v1/backends/{id}` | 通常无需本机凭证；Client 配置了 token 时须提供它 | 同第一行 |
| 第三方直连 C++ server | `http://127.0.0.1:8000` | runtime 的 `--password`，建议使用 Bearer 头 | 原生推理 API，没有 Launcher 进程管理 API |

浏览器请求链路：浏览器 → 本机 Client → 远端 Agent → 同机 C++ server。Client 与 Agent 是同一个 Go 程序的启动形态。一个 Agent 管一个 runtime；多个 GPU runtime 应启动多个 Agent，各占独立端口。

**Base URL 约定不同**：注册后端时填写根地址，不带 `/api` 或 `/v1`；OpenAI SDK 的 `base_url` 则必须包含 `/v1`。不要把 SDK 地址直接存入后端注册表。

Launcher 转发当前只支持 `/v1/*`、`/api/v1/*` 及有限旧控制路径。原生 `/state/*` 和 `/translate/v1/*` 不在转发白名单内，也没有本机 Launcher 代理；这些接口需要直连原生服务或使用支持它们的 router。新 WebUI 的翻译使用 `/v1/batch/completions`。

## 2. 最小联调流程

以下命令使用 Bash。Agent 需与 runtime / 训练 / 量化二进制一起部署；纯 Client 不需要 GPU 或原生二进制。

服务器：

```bash
./rwkv_launcher --listen 0.0.0.0:18766 --token example-agent-token
```

操作者电脑：

```bash
./rwkv_launcher --client
```

注册远端（示例 token 请替换）：

```bash
CLIENT_URL=http://127.0.0.1:10721
curl -sS "$CLIENT_URL/api/v1/backends" \
  -H 'Content-Type: application/json' \
  --data '{"name":"GPU server","base_url":"http://server:18766","token":"example-agent-token"}'
```

返回 HTTP 200 和单个 `BackendView`，例如：

```json
{
  "id": "a1b2c3",
  "name": "GPU server",
  "base_url": "http://server:18766",
  "has_token": true,
  "kind": "agent",
  "capabilities": ["runtime", "tuning_state", "quantization", "metrics", "fs"],
  "reachable": true,
  "last_probe": 1790000000,
  "probe_error": ""
}
```

`capabilities` 随服务器实际工具变化。**注册成功不等于探测成功**：连接或认证失败时仍可能 HTTP 200，并保存后端；必须检查 `reachable` 和 `probe_error`。`last_probe` 为 Unix 秒，`0` 表示尚未探测；重启 Client 后探测结果不持久化，`kind` 可能为空，应重新 probe，不能直接显示“已离线”。

将返回的真实 ID 填入下面变量：

```bash
BACKEND_ID=a1b2c3
BACKEND_URL="$CLIENT_URL/api/v1/backends/$BACKEND_ID"
curl -sS "$CLIENT_URL/api/v1/backends"
curl -sS -X POST "$CLIENT_URL/api/v1/backends/$BACKEND_ID/probe"
curl -sS "$BACKEND_URL/api/v1/node"
curl -sS "$BACKEND_URL/api/v1/runtime"
curl -sS "$BACKEND_URL/api/v1/node/metrics"
```

本机全套形态可使用保留 ID `local`；纯 Client 没有 `local`。当前没有 PATCH/PUT 编辑后端接口：修改地址、名字或 token，需要删除再添加，ID 会变化。

### 启动与停止推理

路径属于 **Agent 所在机器**，不是浏览器电脑。以下启动操作需替换成服务器真实路径；全部 JSON 字段见控制面参考。

```bash
curl -sS "$BACKEND_URL/api/v1/runtime/start" \
  -H 'Content-Type: application/json' \
  --data '{"model_path":"/data/models/model.pth","vocab_path":"/data/rwkv_vocab_v20230424.txt","port":"8000","password":"","use_wkv32":false,"chunk_load":false,"enable_dynamic_loading":false,"chunk_size":128,"state_db_path":"rwkv_sessions.db","tune_cache":"","visible_devices":"0"}'
```

成功返回 `{"ok":true}` 仅表示启动动作成功。轮询 `GET /api/v1/runtime`，等待 `status:"ready"`，再检查 `backend.model` 或 `/v1/server/status` 是否已加载模型。动态加载模式需先调用 `/v1/models`，再显式 `POST /v1/model/load`，请求体为 `{"model":"模型ID"}`；推理请求中的 `model` 字段不会自动切换模型。

```bash
curl -sS -N "$BACKEND_URL/api/v1/runtime/logs"
# 在另一终端执行；restart 使用实际运行配置，不读取新表单：
curl -sS -X POST "$BACKEND_URL/api/v1/runtime/restart"
curl -sS -X POST "$BACKEND_URL/api/v1/runtime/stop"
```

### Chat 与原始续写

```bash
curl -sS -N "$BACKEND_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data '{"messages":[{"role":"user","content":"你好"}],"stream":true,"max_tokens":128,"think_type":"fast"}'

curl -sS "$BACKEND_URL/v1/batch/completions" \
  -H 'Content-Type: application/json' \
  --data '{"contents":["English: Hello\n\nChinese:"],"stream":false,"max_tokens":128,"temperature":1,"top_k":1,"top_p":0,"stop_tokens":[0]}'
```

Chat 使用 `messages` 并由服务端套聊天模板；batch 使用 `contents[]` 做原始续写。客户端按返回的 `choices[].index` 对应输入；非流式文本见 `choices[].message.content`。不要向远端 Chat 发送 `contents` 来假设它会变成 raw continuation。

Launcher 曾为旧 WebUI 把 Chat 上的 `contents` 改写成 batch；旧 WebUI 移除后该改写已删除。客户端必须自己选对端点：原始续写发 `/v1/batch/completions`，聊天发 `/v1/chat/completions`。

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://server:18766/v1",
    api_key="example-agent-token",  # Agent token，不是 runtime password
    timeout=300.0,
)
for chunk in client.chat.completions.create(
    model="rwkv",
    messages=[{"role": "user", "content": "你好"}],
    stream=True,
    max_tokens=128,
    extra_body={"think_type": "fast"},
):
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
```

经本机 Client 时，SDK `base_url` 为 `http://127.0.0.1:10721/api/v1/backends/{id}/v1`。未配置本机 token 时 SDK 可填非空占位 key；转发层会替换为注册表里的远端凭证。直接访问 C++ server 时 key 才是 runtime password。

OpenAI 兼容主要面向 Chat Completions；不要假设 Responses、工具调用或所有 OpenAI 字段都受支持。完整原生请求字段以原生 API 文档为准。

## 3. 浏览器接入规则

- 从 `/api/v1/backends` 获取列表，以 backend ID 保存选中项。业务请求统一加 `/api/v1/backends/${encodeURIComponent(id)}` 前缀，禁止浏览器直接 fetch 后端 `base_url`。
- runtime / jobs / 模型 / state / adapter 缓存按 backend ID 隔离。切换节点后取消旧轮询和流；迟到响应不能覆盖新节点状态。运行中的任务应固定其目标节点。
- 根据能力位展示功能；未知字符串忽略。`kind:"inference_only"` 的节点只使用推理 API，不能起停进程、训练、量化、浏览目录或读取 Agent GPU 指标。
- `reachable` 表示最近一次探测的节点可达性，不等于 runtime ready，更不等于模型已加载。
- 文件浏览用 `POST /api/v1/node/fs`；`{}` 返回 `{"roots":[...]}`，`{"path":"..."}` 返回目录内容。返回的路径要原样回传，不用浏览器平台的路径规则重写 Windows 路径。用户也可手动填写远端路径；白名单外目录不能直接浏览。
- 只有明确支持 `host_dialog` 的本机节点才调用原生文件选择器。远程服务器应使用目录浏览；原生对话框弹在服务器上没有意义。
- Agent token 只在添加表单中短暂输入并交给 Client，不写 localStorage；列表只显示 `has_token`。runtime `config.password` 总是脱敏为空，不能拿空串冒充原密码重新启动。
- `visible_devices` 是字符串：缺省表示沿用 Agent 的配置，空字符串表示显式不注入。不要把它转换成数字数组或默认发送空字符串。注意 runtime start 会在已保存配置上解码：此前指定过选卡时，省略字段会保留该值，详见 API 参考。
- runtime / jobs 是单例，不是任务队列。`jobs/{id}` 只接受 `tuning`、`quantization`；能力探测供 UI 展示，真正的启动与显卡互斥校验由 Agent 执行。

## 4. SSE、返回值与错误处理

| 接口类别 | 格式 | 完成方式 |
| --- | --- | --- |
| runtime / jobs 日志 | `data: 一行原始日志\n\n`，不是 JSON | 连接持续保持；进程结束不会自动关闭流 |
| Chat / batch 推理流 | `data: {JSON}\n\n` | 消费 `choices[].delta`，以 `data: [DONE]` 结束 |
| 普通控制面 | JSON；部分成功停止接口为空体 | 先检查 HTTP 状态，再决定是否解析 JSON |

日志 SSE 没有 event ID、重放游标或 `[DONE]`；重连会再次发送缓冲日志，不保证 exactly-once。每约 300ms 检查一次日志，空闲时仍 flush。不要用 Chat 的 JSON SSE parser 解析控制面日志。

直接连接带 token 的 Agent 时用 `fetch` 读取流，因为原生 `EventSource` 无法设置 Authorization；token 不支持 query 参数。经不带本机 token 的同源 Client 时可以用 EventSource。推理流断开但未收到 `[DONE]` 应保留已收到的文本并标记中断。

控制面大多数失败为 `{"error":"说明"}`，有时还带 `reason`。代理传回的错误保留上游格式；网络代理错误可能是纯文本，不能假设所有错误都有 JSON body。

| 状态码 | 当前常见含义 |
| --- | --- |
| 200 | 成功；注册/probe 还要检查 `reachable`；任务停止成功可能空 body |
| 400 | JSON/参数校验失败、进程冲突、磁盘写入失败、原生对话框不适用 |
| 401 | Agent 或 Client token 错误；也可能是上游推理鉴权失败 |
| 403 | Host/Origin 检查失败或目录超出白名单 |
| 404 | 新控制面路径/方法不匹配、未知 backend/job、Client 上无本地 Agent |
| 405 | 旧 `/api/*` alias 的方法不匹配 |
| 501 | 旧 Agent 不支持对应的新控制能力 |
| 502 | 代理无法连接目标或 runtime；body 可能不是 JSON |

新控制面 JSON 解码限制为 1 MiB，拒绝未知字段和尾随 JSON；这个上限不是 `/v1/state/upload` 的 multipart 限制。注册表暂不支持分页或编辑接口。模型加载可能等待现有推理结束，不要用状态轮询的 4 秒超时去限制它；不要对启动、上传、生成等操作做无条件自动重试。

## 5. 联调验收建议

1. Client 空列表正常；添加正确/错误 token 后分别显示可达/认证失败；重启后列表仍在并能重新探测。
2. 通过 backend 前缀完成 runtime start → ready → stop；日志流实时可见；切换节点不串日志或模型信息。
3. Chat 流式完成与主动中止、batch 原始续写、模型显式加载、state 上传和 adapter 注册均使用同一目标节点。
4. 裸推理节点及旧 Agent 能正确降级；未知 capability 不导致页面报错。
5. 文件浏览覆盖根列表、正常目录、越界 403；指标不可用时显示原因；选卡区分缺省、空串和 `"0"`。

自动化基线：Launcher 目录内 `bun run test`、`bun run lint`、`bun run build`、`go test -race .`、`go vet .`。GPU 实机推理/训练、AMD 字段和跨平台部署需另做验收；mock 与本地单测不代表这些项目已完成。
