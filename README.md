# GoForge — 一个可落地的 Go Coding-Agent Harness

> Golang + Agent + LangGraph（状态图编排）+ RAG 四件事，全部用真实的 Go 代码实现，单二进制、零服务依赖、可离线跑通、带测试。
>
> 面向「老板要的 golang + agent + langgraph + rag」：不是 PPT，也不是把 Python 项目翻译一遍——而是把每个概念落到一个可读、可测、可扩展的 Go 包里。

---

## 1. 这个项目解决了什么

一个 Coding Agent 想在真实仓库里干活，需要四块能力：

| 能力 | 在 Python 世界 | 本项目（Go） |
|---|---|---|
| **Agent 循环** | LangChain / LangGraph 的 agent | `internal/agent`：ReAct 循环（call model → 跑工具 → 回填观察 → 直到无 tool call） |
| **图编排** | LangGraph 的 `StateGraph` | `internal/graph`：节点、静态/条件边、环、reducer、检查点、人工中断 |
| **RAG** | LlamaIndex / LangChain retriever | `internal/rag`：切块 → 嵌入 → 向量库 → 检索，向量库可插拔（内置余弦 / [chromem-go](https://github.com/philippgille/chromem-go)） |
| **工具/运行时** | 各种 sandbox | `internal/tools`：read/write/edit/glob/grep/shell/code_search，全部被 workspace 根目录约束 |

产物是一个 CLI：`goforge`。它能 **建索引、跑编码任务、问代码问题、导出图、暂停/恢复（人类审批）**。

```mermaid
flowchart LR
  subgraph CLI[goforge CLI]
    I[index] --- R[run/ask] --- G[graph] --- T[tools] --- RS[resume]
  end
  R --> H[app.Harness]
  H --> AG[agent ReAct]
  H --> GR[graph engine<br/>StateGraph + checkpoint]
  H --> RG[rag retriever]
  H --> TL[tools registry]
  RG --> EM[embedder<br/>hashing | openai]
  RG --> VS[vector store<br/>memory | chromem-go]
  GR --> CP[(checkpoints<br/>JSON)]
  TL --> WS[(workspace files)]
```

---

## 2. 30 秒跑起来（离线、无需 API Key）

```bash
cd goforge
make build                 # 需要 Go 1.24+；产物在 bin/goforge
./bin/goforge index --dir testdata/sample-repo
./bin/goforge run --mock --verbose --dir testdata/sample-repo \
  "Where is the Add function defined and what does it do?"
```

你会看到真实的执行轨迹（不是硬编码）：

```
▶ start at retrieve
✓ retrieve  step 1  → agent
✓ agent     step 2  → guard
✓ guard     step 3  → tools
✓ tools     step 4  → agent
✓ agent     step 5  → guard
✓ guard     step 6  → tools
✓ tools     step 7  → agent
✓ agent     step 8  → finalize
✓ finalize  step 9  → __end__
🏁 done
```

`--mock` 使用一个**确定性的 Demo 客户端**（`internal/llm/demo.go`），它按真实的多步策略走：`code_search → read_file → finalize`。整条链路（图路由、RAG 检索、工具执行、ReAct 环）都会被执行，只是模型换成了脚本代替，所以 CI 和 Demo 完全不依赖网络。

### 用真实模型（OpenAI 兼容）

```bash
export DEEPSEEK_API_KEY=sk-...        # 或 GOFORGE_API_KEY / OPENAI_API_KEY
./bin/goforge index --dir .
./bin/goforge run --provider deepseek --model deepseek-chat \
  "给 pkg/calc 增加 Multiply 函数，并补一个表驱动测试"
```

任何 OpenAI 兼容端点都行（OpenAI / DeepSeek / Moonshot / vLLM / Ollama 的 `/v1`）：

```bash
./bin/goforge run --base-url http://localhost:11434/v1 --model qwen2.5-coder "解释 cmd/main.go"
```

---

## 3. 命令一览

| 命令 | 作用 |
|---|---|
| `goforge index --dir . [--ext .go,.py] [--vector-store chroma]` | 建立代码 RAG 索引 |
| `goforge run "<task>" [--mock] [--index] [--require-approval] [--verbose]` | 跑一个编码任务 |
| `goforge ask "<question>"` | 同 `run`，语义上用于问代码问题 |
| `goforge resume --run-id <id> [--auto-approve=false]` | 恢复被中断的 run（人工审批后继续） |
| `goforge graph` | 输出 Agent 状态图（Mermaid） |
| `goforge tools` | 列出可用工具及其说明 |
| `goforge version` | 版本 |

全局参数（都可任意组合）：`--config --provider --base-url --model --api-key --dir --vector-store --embedder --max-steps --require-approval --verbose`。

### 人类审批（Human-in-the-loop）

```bash
./bin/goforge run --require-approval --dir . "删除所有临时文件并重构 testdata"
# 遇到写文件/执行 shell 时暂停，打印 run id 和待批准动作
./bin/goforge resume --run-id run-20261029T091421-1
```

这就是 LangGraph 的 **interrupt / resume**：图执行到 `guard` 节点时抛 `Interrupt`，检查点落盘；`resume` 时从同一节点恢复，把 `approved=true` 合入状态，继续执行 `tools`。

---

## 4. 与主流开源项目的对位（以及为什么这样搭）

写之前调研了 Go 生态里几类有代表性的实现，并**只取需要的部分**：

| 参考项目 | 它擅长什么 | 本项目怎么用 |
|---|---|---|
| [cloudwego/eino](https://github.com/cloudwego/eino)（字节 CloudWeGo，~12k★） | Go 里最像 LangGraph 的图/工作流编排 + 类型化组件 + ReAct/DeepAgent | 作为**设计参照**：我们的 `internal/graph` 复刻了「节点 + 条件边 + 环 + 检查点 + interrupt」这套心智模型，但保持 400 行以内、零依赖，方便面试时逐行讲清 |
| [tmc/langchaingo](https://github.com/tmc/langchaingo)（~9k★） | Go 版 LangChain：最广的 provider / vectorstore / splitter | 借鉴其 `vectorstores`、`embeddings`、`textsplitter` 的接口分层；我们保留同样的可插拔边界，但不引入其 pre-1.0 的整包重依赖 |
| [smallnest/langgraphgo](https://github.com/smallnest/langgraphgo) | 功能对齐 Python LangGraph 的 Go 实现（并行、checkpoint、HITL、可视化） | **API 对位验证**：我们的 `AddNode / AddEdge / AddConditionalEdge / AddReducer / Invoke / Resume` 与其命名一致，读过其 `graph/` 包后确认接口方向正确 |
| [philippgille/chromem-go](https://github.com/philippgille/chromem-go)（~7k★） | 纯 Go 嵌入式向量库，无服务、可持久化 | **直接作为真实依赖**接入 `internal/rag/chroma_store.go`，`--vector-store chroma` 即启用；默认仍用零依赖的余弦存储，保证 clone 即跑 |
| [LangGraph](https://github.com/langchain-ai/langgraph)（Python 原版） | 状态图、checkpoint、HITL 的源头 | 概念映射见下表 |

> 参考文章：[Go AI Agent Frameworks: Eino vs LangChainGo vs Genkit](https://dreaming.press/posts/golang-ai-agent-framework-eino-vs-langchaingo.html) —— 核心观点很中肯：**Go 自带 goroutine / context 这套 runtime，框架真正值钱的只有「图编排」和「可观测」**。所以本项目自己实现「图」，把「RAG 存储」交给成熟的 chromem-go，把「模型调用」做成 OpenAI 兼容薄客户端。

### LangGraph → GoForge 概念映射

| LangGraph (Python) | GoForge (Go) | 位置 |
|---|---|---|
| `StateGraph` / `State` | `graph.Graph` / `graph.State` | `internal/graph/graph.go`, `state.go` |
| `add_node` | `(*Graph).AddNode` | `internal/graph/graph.go` |
| `add_edge` / `add_conditional_edges` | `AddEdge` / `AddConditionalEdge` | 同上 |
| `State` reducer（如 `add_messages`） | `SetReducer(key, AppendReducer)` | `internal/graph/state.go` |
| `compile()` → `Runnable` | `Compile()` → `*CompiledGraph` | `internal/graph/executor.go` |
| `Checkpointer` / `thread_id` | `Checkpointer` / `RunID` | `internal/graph/checkpoint.go` |
| `interrupt()` / `Command(resume=...)` | `*graph.Interrupt` + `RunConfig{Resume:true}` | `internal/graph/checkpoint.go`, `executor.go` |
| `graph.get_graph().draw_mermaid()` | `(*CompiledGraph).Mermaid()` / `goforge graph` | `internal/graph/graph.go` |
| `ToolNode` / `tools_condition` | `tools` 节点 + `agent` 节点上的条件边 | `internal/app/graph.go` |

---

## 5. Agent 状态图（本项目的真实拓扑）

```mermaid
flowchart TD
  start([start]) --> retrieve
  retrieve["retrieve<br/>RAG 取回代码上下文"] --> agent
  agent{"agent<br/>一次 LLM 回合"}
  agent -->|有 tool_calls| guard
  agent -->|无 tool_calls| finalize
  guard["guard<br/>需要则 interrupt 等审批"] --> tools
  tools["tools<br/>执行工具并回填观察"] --> agent
  finalize["finalize"] --> end([end])
```

- `retrieve`：用 RAG 先给任务「垫」一层相关代码（top-6 片段，带 `path:start-end`）。
- `agent`：一次模型调用；种子消息只在首轮注入（system prompt + task + 检索上下文）。
- 条件边：模型要工具 → `guard`；不要工具 → `finalize`。
- `guard`：`--require-approval` 时，遇到 `write_file / edit_file / run_shell` 暂停。
- `tools` → `agent`：这就是 ReAct 环，`SetMaxSteps` 兜底防止死循环。

---

## 6. RAG 管线

```
文件遍历（跳过 .git/node_modules/vendor…）
   │  rag/indexer.go
   ▼
按行切块（MaxLines=60, Overlap=12, MaxChars=2400）  rag/chunk.go
   │
   ▼
嵌入（hashing 特征哈希，离线确定性 / openai 语义向量）  rag/embedder.go, embedder_openai.go
   │
   ▼
向量库（内置余弦 MemoryStore / chromem-go 持久化）      rag/store.go, chroma_store.go
   │
   ▼
检索 + 轻量重排（代码优先、README 降权、代码意图识别）   rag/retriever.go
   │
   ▼
作为工具 `code_search` 暴露给 Agent                   tools/code_search.go
```

- **为什么默认 hashing 而不是向量模型**：为了「clone 即跑、CI 可复现」。hashing（feature hashing）对代码标识符的字典匹配足够好；`--embedder openai` 可一键切换成语义向量。
- **重排策略**：代码 chunk 权重 1.08，markdown 0.80；当问题里出现 `function/defined/return…` 等代码意图词时，文档类 chunk 再乘 0.3。避免「README 提了一嘴 Add，就压过真正定义 Add 的 calc.go」。

---

## 7. 目录结构

```
goforge/
├── cmd/goforge/main.go        # CLI：index / run / ask / resume / graph / tools
├── internal/
│   ├── llm/                   # 消息模型 + OpenAI 兼容客户端 + Mock/Demo 客户端
│   ├── graph/                 # ★ LangGraph 式状态图引擎（节点/边/环/检查点/中断）
│   ├── agent/                 # ReAct 循环（可直接复用）
│   ├── rag/                   # ★ 切块/嵌入/向量库/检索
│   ├── tools/                 # ★ 文件、grep/glob、shell、code_search
│   ├── app/                   # 把上面拼成 coding-agent 图 + Harness
│   └── config/                # JSON 配置 + 环境变量
├── testdata/sample-repo/      # 演示用的小型 Go 仓库
├── examples/config.json       # 配置样例
├── docs/architecture.md       # 设计细节、失败模式、扩展点
├── Makefile
└── README.md
```

★ = 面试可深聊的三个核心模块。

---

## 8. 测试

```bash
make test            # go test ./...
make race            # go test -race ./...
make cover           # 输出覆盖率
```

覆盖点包括：图的线性/条件分支/环/步数上限、reducer 合并、**中断后恢复**、RAG 切块与检索命中、chromem 适配、工具路径逃逸防护、shell、OpenAI 客户端解析、Demo 端到端多步轨迹、**人工审批后才落盘**。

---

## 9. 配置

优先级：**命令行参数 > 环境变量 > JSON 配置文件 > 默认值**。

```bash
GOFORGE_API_KEY=...        # 或用 OPENAI_API_KEY / DEEPSEEK_API_KEY
GOFORGE_BASE_URL=...
GOFORGE_MODEL=...
GOFORGE_WORKDIR=...
GOFORGE_VECTOR_STORE=memory|chroma
GOFORGE_MAX_STEPS=24
GOFORGE_REQUIRE_APPROVAL=true
```

`--config examples/config.json` 可加载完整配置（JSON，零外部依赖解析）。

---

## 10. 生产化还差什么（诚实清单）

这是「可落地 MVP」，不是「生产完备」：

- **沙箱隔离**：`run_shell` 目前在 workspace 内直接执行。生产应换成容器/gVisor/seatbelt，限制网络与系统调用。
- **真实嵌入**：默认 hashing 是词法级；上语义请配 `--embedder openai` 或接本地 embedding 服务。
- **可观测性**：`graph.Event` 已可接入，下一步接 OpenTelemetry span/指标，并把 trajectory 存成可回放的 run 记录。
- **评估**：接入 SWE-bench-lite 风格任务集，对比 baseline 与 harness 的成功率/成本/步数。
- **并发**：图执行目前单线程；`internal/graph` 预留了按 superstep 扇出的空间（goroutine + channel）。
- **上下文压缩**：已有 `trimMessages` 的简单裁剪；可升级为 LangGraph/LangChain 的 condenser。

---

## 11. 参考链接

- eino — <https://github.com/cloudwego/eino>
- LangChainGo — <https://github.com/tmc/langchaingo>
- LangGraphGo — <https://github.com/smallnest/langgraphgo>
- chromem-go — <https://github.com/philippgille/chromem-go>
- LangGraph (Python) — <https://github.com/langchain-ai/langgraph>
- Go Agent 框架横评 — <https://dreaming.press/posts/golang-ai-agent-framework-eino-vs-langchaingo.html>

---

MIT License. 见 [LICENSE](LICENSE)。
