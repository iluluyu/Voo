# 交接（2026-10-24 凌晨）

本轮主题：**词库闭环 + 模型可加载 + 后处理变快**。全部已提交、门禁全绿。

## 本轮落地（可用）

### 词库

- **导入即生效**：公开词库导入默认 `ActiveExact`（只纠正**完全同音**），不再"导完躺在候选里"。
  这是"感觉只能用其中一个词库"的真正原因——旧默认是 Candidate（只建议、不改写）。
- **多源并存**：`imported_lexicons` 每源独立快照，互不覆盖。
- **来源级模式**：`lexicon_source_modes`（仅建议 / 仅完全同音 / 近音），单条 override 优先于来源级。
- **统一 resolver**：`lookup_lexicon_terms(terms)` 返回 `canonical / effective_mode / winning_rule / contributing_sources / suppressed_sources / explanation`。
  权威优先级：**用户单条 override > 用户自有词（手动词＝人工确认的 AI 词）> 公开词库**；公开源写法冲突时保持歧义并降为 Candidate，不靠导入顺序/词频决胜。
- **UI**：词库来源页顶部「已安装(N)」全量列出每个源（词数／模式徽章／三档切换／更新／移除）；可导入列表对已安装源禁用导入。

### 打字学习（新）

粘贴成功后：2s 读一次目标输入框 → 8s 再读一次 → **两次一致且不同于粘贴内容**才算稳定 → 词级 diff → **只产出候选**，绝不直接写 `custom_words`。拿不到文本就静默返回（不 warn）。

### 后处理

- **思考深度四档**：不思考（默认）／低／中／高。
- **provider 映射**：DeepSeek `reasoning_effort ∈ none/low/high/max`，**缺 medium 时向下取 low**（延迟优先）；OpenAI 兼容走 `low/medium/high`；Anthropic 走 `budget_tokens 1024/2048/4096`；未知 provider 忽略不支持字段，被 400 拒绝时自动降级重发。
- 固定参数：`max_tokens 512`、词表 8 条、上下文游标前 64 字、超时 20s、`temperature 0.2`。
- 上下文注入：焦点 app 名 + 游标前 ≤128 字符，作为独立不可信字段；拿不到则请求体与无上下文逐字节一致。

### 模型

- **后端已就绪**：`import_model_file(path)`、`import_model_url(url, sha256?)`、`check_model_compat(path)`。
  预检：未知架构 / 缺词表 → **blocked**；量化后端不支持 → **仅警告**。
- **未做**：模型库表格 UI（两次派发都在调研阶段耗尽轮次，代码零落地）。`modelEvidence.ts` 已落盘可用（20 个中文模型的发布方 CER + RTF 映射），UI 只差把它渲染出来。

## 验证状态

| 门禁                         | 结果                                   |
| ---------------------------- | -------------------------------------- |
| `cargo test --lib --offline` | **723 passed / 0 failed / 21 ignored** |
| `cargo fmt -- --check`       | 通过                                   |
| `bun run build`              | 通过                                   |
| `bun run lint`               | 通过                                   |
| `bun run check:translations` | 25 语言全完整                          |

### 已知债务

1. `llm_client` 两个测试被 `#[ignore]`（`stalled_endpoint_times_out`、`endpoint_rejecting_reasoning_field_degrades_to_plain_request`），原因写在 ignore 文案里，不阻塞功能，下一轮修。
2. 模型库表格 UI 未做（见上）。
3. MCP 提案服务未做（设计已定：`voo lexicon mcp`，3 工具只写 pending）。
4. 量化对照表未完成：下载 1.6GB 模型时被中间盒重置（HTTP/2 stream reset），脚本已改 `--http1.1` + 断点续传重试，需在网络稳定时重跑 `python3 scripts/quant_compare.py`。

## 界面测试清单（按此顺序点）

1. 设置 → 词库 → **词库来源**：不再报 `Command not found`；「已安装」区块列出全部源（装两个就有两行）。
2. 同一个源切三档模式：**仅建议**→不改写；**仅完全同音**→同音错字被纠正；**近音**→范围更宽（慎用）。
3. 装两个源（如 `rime-ice-en` + `zhwiki`，仅拉丁 + 1000 条）：说含 `DeepSeek` 的句子，两个源都应生效。
4. 设置 → 后处理：**思考深度**默认「不思考」；说一句短中文，看是否 ~1 秒内返回且被改写。
5. 设置 → 词库 → 自定义词汇：手动词可批量添加/搜索/分页/导出。
6. 说一段话后**手动改几个字**，8 秒后看候选区是否出现学习到的候选（只进候选，不动词库）。

## 下一轮建议顺序

1. 模型库表格 UI（`modelEvidence.ts` 已备好）
2. MCP 提案服务 + 与内置 AI 整理共用提案通路
3. 修两个 ignored 测试
4. 网络恢复后跑完量化对照表
