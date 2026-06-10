# ollama-proxy

将本地 [Ollama](https://ollama.com) 暴露为 **OpenAI 兼容 API** 和 **Anthropic Messages API** 的轻量代理。任何支持这两种协议的客户端（OpenAI SDK、Anthropic SDK、Claude Code、各类聊天客户端等）都可以直接对接本地模型。

无第三方依赖，仅使用 Go 标准库。

## 快速开始

```bash
# 确保本地 Ollama 正在运行（默认 http://127.0.0.1:11434）
go run ./cmd/ollama-proxy
```

```bash
# OpenAI 协议
curl http://127.0.0.1:3000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5:latest","messages":[{"role":"user","content":"你好"}]}'

# Anthropic 协议
curl http://127.0.0.1:3000/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5:latest","max_tokens":1024,"messages":[{"role":"user","content":"你好"}]}'
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PORT` | `3000` | 代理监听端口 |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama 地址 |
| `NUM_CTX` | 不设置 | 上下文窗口下限（见下文「上下文窗口」） |
| `NUM_CTX_MAX` | `32768` | 自适应上下文窗口的上限 |

### 上下文窗口

OpenAI/Anthropic 协议没有传递上下文窗口大小的字段（窗口是云端模型的固有属性，超限会报错）；而 Ollama 的 `num_ctx` 默认很小（4096），**超限时会静默截断 prompt 开头**——长对话里系统提示词会先被截掉，且客户端无从感知。

代理按请求自动计算 `num_ctx`：`估算输入 token + max_tokens（缺省按 2048）+ 余量`，向上取整到 1024 的倍数：

- 计算值 ≤ 4096 且未设 `NUM_CTX` 时不干预，沿用 Ollama / Modelfile 自身配置
- 超出时自动放大到所需大小（受 `NUM_CTX_MAX` 封顶，防止显存耗尽）
- 设置了 `NUM_CTX` 则作为每个请求的窗口下限
- 通过 `/api/show` 查询模型原生 `context_length` 并按模型缓存，计算值不会超过模型自身支持的上限（超过只会浪费显存）

### 能力预校验

代理同时通过 `/api/show` 读取模型的 `capabilities`（按模型缓存），在请求转发前校验，不匹配时返回清晰的 400（而不是 Ollama 的 500/501）：

| 请求内容 | 需要的能力 |
|----------|-----------|
| 图片输入 | `vision` |
| `tools` 工具定义 | `tools` |
| `thinking: enabled` | `thinking` |
| `/v1/embeddings` | `embedding` |
| `/v1/completions` 的 `suffix` | `insert` |

能力信息不可用时（如模型尚未拉取）跳过校验，由 Ollama 自行报错。流式请求的校验失败同样以 HTTP 400 返回（Anthropic 的 `message_start` 事件推迟到后端确认产出后才发送）。

## 支持的端点

### OpenAI 兼容

| 端点 | 说明 |
|------|------|
| `POST /v1/chat/completions` | 聊天补全（也接受 `/chat/completions`） |
| `POST /v1/completions` | Legacy 文本补全（对接 `/api/generate`，支持 `suffix`；不支持多 prompt） |
| `GET /v1/models` | 列出 Ollama 已安装的模型 |
| `GET /v1/models/{id}` | 查询单个模型 |
| `POST /v1/embeddings` | 文本向量（需 embedding 模型，对接 `/api/embed`） |

`chat/completions` 支持的参数：

- **消息**：`messages`（`content` 支持字符串或内容块数组；`image_url` 支持 data-URL base64 和远程 `http(s)` URL——远程图片由代理下载后转 base64 喂给视觉模型，限 20 MB）
- **结构化输出**：`response_format`（`json_object` 映射为 Ollama 的 `format: "json"`；`json_schema` 直接传入 schema 约束输出）
- **工具调用**：`tools` / `tool_choice`（`"none"` 会禁用工具；Ollama 不支持强制指定工具，其余取值按 auto 处理）；响应返回 `tool_calls`，`finish_reason` 为 `tool_calls`；`role: "tool"` + `tool_call_id` 回传工具结果
- **思考模型**：思考内容以 `reasoning_content` 字段返回（DeepSeek 约定），流式同样支持
- **采样**：`temperature`、`top_p`、`seed`、`stop`（字符串或数组）、`presence_penalty`、`frequency_penalty`、`max_tokens` / `max_completion_tokens`
- **流式**：`stream`、`stream_options.include_usage`

### Anthropic Messages

| 端点 | 说明 |
|------|------|
| `POST /v1/messages` | 消息补全 |
| `POST /v1/messages/count_tokens` | Token 估算（约 4 字符/token，本地估算不请求模型） |

`messages` 支持的参数：

- **消息**：`system`（字符串或内容块数组）；`content` 支持 `text`、`image`（`base64` 与 `url` source——远程图片由代理下载后转 base64，限 20 MB）、`tool_use`、`tool_result`、`thinking` 块
- **工具调用**：`tools`（`input_schema`）/ `tool_choice`；响应返回 `tool_use` 内容块，`stop_reason` 为 `tool_use`；流式按规范输出 `content_block_start` + `input_json_delta`
- **思考模型**：`thinking: {"type": "enabled"}` 映射为 Ollama 的 `think`；思考内容以 `thinking` 内容块 / `thinking_delta` 返回
- **采样**：`temperature`、`top_p`、`top_k`、`stop_sequences`、`max_tokens`（必填）
- **流式**：`stream`，事件序列符合规范（`message_start` → `ping` → 内容块事件 → `message_delta` → `message_stop`）

## 架构

六边形（端口与适配器）架构，协议细节与业务逻辑隔离：

```
cmd/ollama-proxy/          入口：装配依赖、HTTP server、优雅关停
internal/
├── domain/                协议无关的领域模型（消息、工具、流式块）
├── application/           用例层：ChatUseCase（输入端口）、OllamaClient（输出端口）
├── adapter/
│   ├── handler/           HTTP 路由、SSE 写入、OpenAI/Anthropic 错误格式
│   └── converter/         两种协议 DTO ↔ 领域模型的相互转换
└── infrastructure/        Ollama HTTP 客户端（/api/chat、/api/tags、/api/embed）
```

新增协议只需添加一组 converter + handler，领域层和 Ollama 客户端无需改动。

## 限制

- 不支持鉴权（面向本机使用；如需对外暴露请自行加反向代理——尤其是开启了远程图片下载，存在 SSRF 面）
- `tool_choice` 无法强制指定某个工具（Ollama 限制）
- `count_tokens` 为本地估算值，并非模型真实分词
- OpenAI `n > 1` 多候选、`logprobs`、`/v1/completions` 的多 prompt 数组未实现
