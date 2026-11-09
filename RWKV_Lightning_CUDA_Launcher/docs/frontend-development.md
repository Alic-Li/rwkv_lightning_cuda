# 前端开发指南

## 技术栈与选型

| 层 | 当前选择 | 职责 |
| --- | --- | --- |
| UI | React 19 + TypeScript | 页面与组件 |
| 状态 | Zustand 5 | 表单、会话、任务与设置 |
| 构建 | Vite 6 | 开发服务器与静态资源构建 |
| 样式 | Tailwind CSS 4（`@tailwindcss/vite`，CSS-first `@theme`） | 设计令牌与页面样式，不引入 HeroUI |
| 组件 | shadcn 风格自建组件 + Radix primitives | Dialog / Popover 的焦点管理、键盘与可访问性 |
| 包管理 | Bun，使用 `bun.lock` | 安装依赖与执行脚本 |
| 开发工具链 | Bun（CI 使用 1.3.9） | 沿用原作者的依赖管理、脚本与测试配置 |
| 测试 | Bun 内置测试（`bun:test`） | API/SSE、store 与服务端静态渲染测试 |
| 生产服务 | Go Launcher | 内嵌 `dist/`、控制面、代理转发 |

Bun 不是前端框架；它承担包管理、脚本运行和测试。项目沿用原作者的 Bun 工具链，开发与 CI 使用同一份锁文件。Tailwind 是 CSS 工具集，不提供完整的可访问组件；对话框、菜单、焦点管理与键盘交互由 `@radix-ui/*` 与 `src/components/ui/` 负责。

主题令牌定义在 `src/styles/globals.css`：`:root` 为浅色、`[data-theme="dark"]` 为深色，`@theme inline` 把它们映射成 Tailwind 颜色（`bg-card`、`text-muted-foreground`、`border-border` …）。默认主题是 `system`：`index.html` 的内联脚本在 React 挂载前读取 `rwkv-settings-v2`（缺省按 `prefers-color-scheme` 解析）写入 `data-theme`，`App` 再监听系统外观变化，避免首屏闪白。`settings` store 的 migrate 会把旧版本里硬编码的 `dark` 视为未选择并改为 `system`，显式的 `light` 保留。

## 安装、运行、验证

先安装 Bun（建议与 CI 的 1.3.9 一致），然后在 `RWKV_Lightning_CUDA_Launcher/` 目录执行：

```bash
bun install --frozen-lockfile
bun run test
bun run lint
bun run typecheck
bun run build
```

联调需要两个终端。先构建本机 Client，再启动前端：

```bash
# 终端 A，位于 RWKV_Lightning_CUDA_Launcher
# dist 已构建；二进制会将它嵌入。
go build -o rwkv_launcher .
./rwkv_launcher --client

# 终端 B，同一目录
bun run dev
```

Vite 默认端口 5173，`/api`、`/v1` 代理到 `127.0.0.1:10721`。Go 未启动时前端页面可以加载，API 会失败。Launcher 会拒绝 Origin 不是自己的请求，所以 `vite.config.ts` 的代理除了 `changeOrigin` 改写 Host，还在 `proxyReq` 里手动把 `Origin` 改成 Launcher 地址；去掉这一步，开发页面的每个 POST 都会得到 403。若要验证本机 runtime 启停，把 Launcher 放在原生二进制同目录并以全套形态启动；`go run .` 的临时目录不适合验证这个场景。

其他命令：`bun test --watch tests` 持续测试，`bun run test:go` 跑 Go race 测试，`bun run preview` 仅预览静态构建（不提供 Go 控制面）。生产只需 Go/原生二进制，无需 Node 或 Bun。

## 目录职责

```text
src/
  app/                  应用壳：Header、侧栏 Rail、侧栏底部节点卡片（GPU 热力格 + 节点切换 popover）、日志抽屉、全局轮询、当前节点上下文
  pages/                Nodes / Chat / Translate / Runtime / Training / Quantization / Settings
  components/
    ui/                 shadcn 风格基础组件（button/input/select/card/dialog/popover/field/…）
    chat/               Chat 页专用组件：气泡、代码块、输入区、会话选择、生成参数面板、state / adapter 管理
    training/           训练页组件：任务栏（Start / Stop）、参数卡片、进度与曲线卡片
    *.tsx               通用控件：页头与远端路径字段（common）、日志查看器、节点状态与 GPU 绘制、选卡控件、指标曲线、添加 / 编辑后端与目录对话框
  lib/
    api/                同源 HTTP 传输、控制面与推理接口、SSE parser、schema 类型与默认值
    translate/          分段、语言白名单与调度逻辑
    chat/html.ts        HTML 预览提取与沙箱包装
    i18n.ts             zh / en 文案目录与 useI18n
    router.ts           hash 路由（#/nodes、#/chat、#/translate、#/runtime、#/training、#/quant、#/settings）
  stores/               Zustand 状态与本地持久化
  styles/globals.css    Tailwind 入口、主题令牌与 Markdown 排版
tests/                  Bun 测试
docs/                   对接与开发文档
dist/                   生产构建产物，**不入库**；由 Go `//go:embed dist/*` 嵌入
```

Go 侧：`main.go` 管进程、CLI 参数与启动；`api.go` 管控制路由和鉴权；`request_types.go` 集中定义进程控制请求体；`backends.go` 管注册表和转发；`fsbrowse.go` 管目录白名单；`devices.go` 管选卡；`metrics*.go` 管指标。构建必须使用 `go build .`，不能只编译 `main.go`。

## 多后端前端架构

浏览器只与本机 Client 同源通信，任何节点级请求都带 `/api/v1/backends/${id}` 前缀；`src/lib/api/http.ts` 是唯一的出口，禁止直接 fetch 后端的 `base_url`。

- `stores/backends.ts`：注册表列表、当前 backend ID（持久化 `rwkv-backends-v1`）、`add` / `update` / `remove` / `probe` / `probeAll`。`update` 对应 `PATCH /api/v1/backends/{id}`，由节点总览详情面板的 ⋯ 菜单进入编辑对话框。`hasCapability()` 是唯一的降级判据。
- `stores/nodes.ts`：按 backend ID 保存 `NodeSnapshot`（`info` / `runtime` / `jobs` / `metrics`），并提供 runtime 与 jobs 的控制动作。`useCurrent()` 汇总当前节点上下文。
- `stores/logs.ts`：runtime 与 jobs 的日志 SSE 连接，键为 `backendId:kind`，切换节点会关闭旧连接，重连会重放缓冲日志。任务结束时只删除自己注册的 controller（同一 tick 内 close→open 会换上新的，按键盲删会让新连接失去 abort 能力并在下次 open 时重复建流）。
- `stores/chat.ts`、`stores/translate.ts`：会话与翻译结果按 backend ID 隔离，翻译还记录 `owner`，切换节点不会串结果。
- `stores/forms.ts` 按 backend ID 保存 Runtime / 训练 / 量化表单；`stores/training-series.ts` 记录训练过程中逐次轮询到的 lr 与 tok/s（Agent 只回 loss 历史）；`stores/ui.ts` 管侧栏折叠、日志抽屉、对话框与 toast。
- 轮询在 `app/App.tsx`：注册表 8s、runtime/jobs 1.6s、GPU 指标 6s；`NodesPage` 另外每 5s 刷新全部节点。

能力降级：`kind:"inference_only"` 的节点只用推理 API，不显示起停进程、训练、量化、目录浏览与 GPU 指标；未知 capability 一律忽略。没有旧协议节点这一档——Client 与 Agent 同版本部署。

`visible_devices` 是三态字符串，不要退化成数字数组：`DeviceSelection` 的 `inherit` 会省略字段（由 Agent 按 §选卡 优先级链解析），`none` 发送 `""`（显式不注入），`explicit` 发送实际选卡串。`applyDevice()` 负责落到请求体。量化没有选卡控件——`rwkv_quantize` 是纯 CPU 工具。

Translate 直接调用 `/v1/batch/completions`，每批最多 128 行，结果按 `choices[].index` 回填。Go 侧给旧 WebUI 用的 `contents → batch` 改写已删除，`/v1` 是透明反代——把 `contents` 发到 `/v1/chat/completions` 不会再被改写成 batch。

## 提交与构建产物

仅维护 `bun.lock`，不用第二份包管理锁文件。增减依赖后用 Bun 更新锁文件；CI 使用 frozen install，执行 lint/test/build 后编译 Go。

构建时 `react-markdown` / `remark-gfm` / `rehype-highlight` 单独打成 `markdown` chunk（见 `vite.config.ts` 的 `manualChunks`）：前端改动时这块不变，浏览器可以继续用缓存。它是静态引入的，首屏仍会一起加载，不是懒加载。

修改前端源码、依赖或样式后，必须重新 `bun run build` 再编译 Go，否则 `//go:embed dist/*` 会直接编译失败（`pattern dist/*: no matching files found`）。`dist/` 是构建产物，**不要提交**——CI 会用锁文件重装依赖并重新构建；同样不要提交 node_modules、Go 本机二进制或临时缓存。Bun 测试中的静态渲染断言不等于浏览器视觉或真实 GPU 验收；改动布局或主题后需在浏览器中确认，GPU 实机行为另做验收。
