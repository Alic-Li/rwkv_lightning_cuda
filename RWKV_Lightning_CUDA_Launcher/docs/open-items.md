# 未决事项

一次 launcher review 之后留下的待办。已修的不在这里——那些在 git log 里。
每条都记了**为什么还没做**：有的是要权衡，有的是缺实机环境，有的是改动风险
大于收益。开头先记已经做出的决定，免得日后有人把它们当成"忘了做"。

## 已经决定的

**混版本不再支持。** Client 与 Agent 必须同版本部署，这是一次干净起步的取舍。
据此已经移除：pre-v1 的 `/api/*` alias 全套、`legacy.go` 的旧协议转发适配、
`probeResult` / `backendView` 的 `legacy` 字段，探测阶梯从三级降到两级
（`/api/v1/node` → `/v1/server/status`）。

留下的后果，日后要回退得先掂量：

- **只说 pre-v1 协议的主机不再被识别成节点**，会落到"不可达"。这是有意的——
  它不是这个 Client 能驱动的东西。
- **`DisallowUnknownFields` 保持严格**。同版本部署下这是优点：字段拼写错误立刻
  报错，而不是被静默忽略。真要重新支持混版本，这一条得先松开，而且**加字段
  会立刻打破新 Client → 旧 Agent**，没有缓冲。已知计划里的统一版本号如果落地，
  应该在那时一起决定放宽策略。

**tune-cache 只提示不迁移。** `deviceTuneCache()` 把 W8A16 缓存钉成
`<模型名>.dev-<tag>.w8a16.tune`，和 C++ 侧「模型名 + GPU 名」的默认路径对不上，
所以换卡后首次启动必然重新调优（约几分钟，每模型每卡一次）。当前处理是 Agent
在缓存缺失时打一行日志（`devices.go` 的 `tuneCacheNote`），前端选卡控件
（`device-selector.tsx`）下方一条提示。几分钟被认为可以接受；若后来觉得不该白等，可以加一次旧默认
路径的回退读取。

## 已确认存在、还没修

### 1. `process.sse` 的日志重放 —— `main.go`

去重靠在缓冲区里找上一条发出去的行来定位。一旦日志滚过 2000 行上限把它挤掉，
`start` 回落到 0，**整个缓冲区每 300ms 重发一次**。训练任务输出量大时会持续刷。

**这是上游原样代码**（对过 `ddfed89`，一字未改），所以不算本次引入。改它有破坏
旧客户端的风险，建议单独开一条：换成单调递增的序号游标，而不是按内容找锚点。

### 2. 轮询没有 in-flight 保护 —— `src/app/App.tsx`

`refresh` 1.6s 一次，超时 6s（`stores/nodes.ts`），Agent 侧 `status()` 还会做一次
1.5s 的本地探测。节点慢的时候每个节点会叠到 3–4 个在飞的请求。标签页隐藏时也
不暂停。加个 in-flight 标志 + `document.visibilityState` 门控很便宜，但属于性能
调优，不适合和 bugfix 混在一起。

### 3. ROCm 字段名未经实机验证 —— `metrics.go`

`rocmSampleUncached` 按后缀匹配 `rocm-smi --json` 的键，文件里自己标了
T1（W7900 实机验证）仍未完成。`rocmNumber` 对匹配不上的键返回 0——
`pickFreestDevice` 的下溢钳位已经补上，但**指标本身仍可能静默为 0**。
需要在真机上确认键集。

## 低优先级 hardening

| 位置 | 问题 |
| --- | --- |
| `api.go` `handleForward` | `/api/v1/backends/local/api/v1/backends/local/...` 每跳消耗一个前缀，收敛不会无限递归，但足够长的 URL 能撬动 N 层嵌套连接。需要 token，加个深度头即可 |
| `main.go` `validateStartup` | loopback 只认 `127.0.0.1` / `::1` / `localhost`，`127.0.0.2` 会被当成非 loopback 而强制要求 token。偏严，不危险；`net.IP.IsLoopback()` 更准 |
| `api.go` `/api/v1/runtime/logs` | 没有 `agentGate`，而 `GET /api/v1/jobs/{id}/logs` 有。client-only 形态下会开一个空的 SSE 流 |
| `src/lib/format.ts` `formatCount` | `toLocaleString("en-US")` 写死，不跟随界面语言 |
| `src/stores/forms.ts` | `FormEntry<C>` 的 `devices` 字段对量化表单已无用（量化是 CPU 任务），但它是三个表单共用的泛型字段，单独摘除要破泛型，留着无害 |

## 验收边界

`docs/integration-guide.md` 说过、这里重申：Bun 测试里的静态渲染断言不等于浏览器
视觉验收，Go 单测不等于真实 GPU 行为。**以下都还没有实机验收**：

- 多卡自动放置与 `--card` 钉定的实际效果；
- AMD / ROCm 路径的全部字段（见第 3 条）；
- 跨机器的 Client ↔ Agent 部署（转发、token、SSE 穿透）——现在还多一条：
  两端必须同版本，混版本会表现为节点探测不到，而不是功能降级；
- 冷加载期间的 Stop/Restart —— 本次修了门控逻辑并有单测，但没在真机上按过按钮。

另外注意 zustand 在 `renderToStaticMarkup` 下走 `getServerSnapshot`，返回的是
**初始 state**，所以组件级 SSR 测试看不到 `setState` 的值。需要断言的逻辑要抽成
纯函数（`runtime-controls.tsx` 的 `runtimeButtonState()` 就是这么做的）。
