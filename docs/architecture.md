# 架构与设计说明

本文解释 GoForge 的核心设计、关键取舍与失败模式，方便评审/面试时深入讨论。代码入口见 `internal/`。

---

## 1. 目标与非目标

**目标**

- 用一个真实可跑的 Go 程序，把 Agent / LangGraph 式编排 / RAG 三件事讲清楚。
- 每个核心模块都能单独测试、单独替换。
- `go test ./...` 绿、`--mock` 离线可复现。

**非目标（不在 MVP 内）**

- 生产级沙箱隔离（容器/gVisor）。
- 分布式调度、多租户、RBAC。
- 训练/微调、超大规模向量检索。

---

## 2. 状态图引擎（`internal/graph`）

### 2.1 执行模型

`CompiledGraph.InvokeWith` 是一个「单线程 superstep」解释器：

```
current = entry
loop:
  node = nodes[current]
  update, err = node.fn(state)          // 节点返回“增量”
  state = merge(state, update, reducers) // 按 key 合并，messages 用 AppendReducer
  next = router(current, state)          // 条件边优先，否则静态边，否则 END
  checkpoint(next, state)                // 落盘，Node=下一节点
  current = next
```

- **增量更新**：节点只返回要改的 key，未提及的 key 保持不变。这让节点可以乱序/并行扩展。
- **reducer**：默认覆盖；`AppendReducer` 用于对话历史（`[]llm.Message` 追加）。
- **环**：`tools → agent` 就是环；`SetMaxSteps` 兜底，超限返回 `MaxStepsError`。
- **校验**：`Compile` 会检查 entry、边指向的节点是否存在、节点是否有且仅有一种出边。

### 2.2 检查点与恢复

`Checkpoint{RunID, Step, Node, State, Status}`。`Node` 语义是「恢复时第一个要执行的节点」。

- `FileCheckpointer`：每次 `Save` 先写临时文件再 `rename`，保证原子性。
- 恢复：`InvokeWith(RunConfig{RunID, Resume:true, Initial})` 从 `Load` 的 `Node/State` 起步，再把 `Initial` 里的控制位（如 `approved`）合并进状态。

> **坑 & 处理**：`State` 是 `map[string]any`，JSON 往返会把 `[]llm.Message` 变成 `[]any`，导致恢复后 `messagesOf` 断言失败、工具调用丢失。解决：`internal/app/checkpoint.go` 的 `stateCheckpointer` 在 `Load` 后按 key 重新水化（marshal→unmarshal 到 `[]llm.Message`）。这是一个真实踩过的 bug，测试 `TestHumanInTheLoopApproval` 覆盖了它。

### 2.3 中断 / 人工审批（HITL）

- 节点返回 `*graph.Interrupt{Prompt, Data}` 时，执行器**不合并**该节点的更新，而是在**当前节点**落一个 `interrupted` 检查点，并把 `*Interrupt` 作为 error 抛出。
- 上层（CLI/Harness）识别到 `Interrupt` 后展示待批准动作；`resume` 时带 `approved=true` 重新进入该节点，节点读状态放行，继续 `guard → tools`。

这种「节点自决是否中断」的写法，天然支持「只在写操作前暂停」，比在引擎里硬编码审批更灵活。

---

## 3. Agent 两种形态

1. **`internal/agent`（独立 ReAct 循环）**：`for` 循环 + 工具注册表，约 80 行。适合作为库调用，也方便单测。
2. **`internal/app` 的图形态**：把一次 LLM 回合做成 `agent` 节点，把工具执行做成 `tools` 节点，用条件边连成环。这是 LangGraph 风格，便于加 `guard`、加检索节点、加验证节点。

两者共享同一个系统提示（`agent.DefaultSystemPrompt`）和同一套工具。

### 上下文控制

`trimMessages(msgs, 40)`：保留 system + 最近窗口。真实项目可替换为摘要式 condenser。

---

## 4. RAG 设计

| 阶段 | 实现 | 取舍 |
|---|---|---|
| 遍历 | `filepath.WalkDir`，按扩展名过滤，跳过 `.git/node_modules/vendor/.tools…` | 简单可控；大文件 >1MB 跳过 |
| 切块 | 行窗口 + overlap + 字符上限兜底 | 行窗口能保住函数完整；字符上限防止单行超长 |
| 嵌入 | 默认 `HashingEmbedder`（feature hashing，512 维，L2 归一化）；可选 `OpenAIEmbedder` | 离线确定性 vs 语义能力，一键切换 |
| 存储 | `MemoryStore`（JSON 持久化）/ `ChromaStore`（chromem-go 持久化） | 零依赖默认 vs 真正的嵌入式向量库 |
| 检索 | top-k 过采样 + 轻量重排 | 代码优先、文档降权、代码意图识别 |

### 重排为什么需要

纯余弦会偏好**短文档**：问题「Where is the Add function defined」里，README 因为短且提到 `Add`，余弦高于真正定义 `Add` 的长文件。重排通过语言先验 + 意图识别修正，是有据可依的工程手段，而不是调参玄学。

### 向量库适配

`VectorStore` 接口只有 `Add / Query / Count / Close`。chromem 的 `QueryEmbedding` 要求 `nResults <= Count()`，适配层做了 clamp（见 `TestChromaStoreQueryClampsTopK`）。

---

## 5. 工具与安全

所有工具通过 `Options{Root}` 解析路径，`resolve()` 保证结果落在 workspace 根内，越界直接报错（`TestPathEscapeRejected`）。

| 工具 | 风险 | 约束 |
|---|---|---|
| `read_file` / `list_dir` / `glob` / `grep` / `code_search` | 只读 | 根目录约束、输出截断 |
| `write_file` / `edit_file` | 写 | 根目录约束；`edit_file` 要求唯一锚点防误替换 |
| `run_shell` | 高 | 超时（默认 60s）、工作目录在根内、输出截断；**生产必须再套容器** |

`--require-approval` 会在 `guard` 节点拦住 `write_file / edit_file / run_shell`。

---

## 6. 扩展点

- **加一个节点**：`graph.AddNode("verify", fn)` + 改边。例：在 `finalize` 前加「跑测试 → 失败回 `agent`」的自愈环。
- **加一个工具**：实现 `tools.Tool` 四方法，`Registry.Register`。
- **换嵌入/向量库**：实现 `rag.Embedder` / `rag.VectorStore`，在 `app.BuildEmbedder/BuildStore` 分支注册。
- **换模型**：任何 OpenAI 兼容端点；或实现 `llm.Client`。
- **接可观测**：`RunConfig.OnEvent` 已把 `start/node/interrupt/end/error` 推送出来，直接接 OTel/log。

---

## 7. 失败模式与应对

| 失败模式 | 现象 | 应对 |
|---|---|---|
| 死循环 | 工具反复调用 | `SetMaxSteps` + `MaxStepsError` |
| 崩溃丢任务 | 进程被杀 | 每步落检查点；`resume --run-id` |
| 恢复丢工具调用 | `[]any` 反序列化 | `stateCheckpointer` 重新水化 |
| 模型瞎编文件 | 未读就改 | system prompt 规约 + `edit_file` 唯一锚点 |
| 路径逃逸 | 读写系统文件 | `resolve()` 根约束 |
| 检索不准 | 命中 README | 语言先验 + 代码意图重排 |
| 上下文爆炸 | 消息过长 | `trimMessages`；可升级 condenser |
| 重复副作用 | 重试重写 | 审批关 + 幂等 `edit_file`；生产可加 tool-call ledger |

---

## 8. 与「生产级」的差距（面试可讲）

1. **执行面**：容器化沙箱、资源限额、网络策略。
2. **持久化**：Postgres/Redis 检查点、任务队列、并发 run。
3. **可观测**：OTel trace + run 回放 + token/成本账本。
4. **评估**：SWE-bench-lite 任务集，baseline vs harness 的成功率/步数/成本。
5. **上下文工程**：condenser、repo-map、按需检索。

这五点正好对应「把 demo 做成生产 runtime」的真实工作，也是最能体现工程判断的部分。
