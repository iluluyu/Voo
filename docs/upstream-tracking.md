# 上游跟进（Upstream Tracking）

Voo 是 [cjpais/Handy](https://github.com/cjpais/Handy) 的 fork，版本号固定在 **1.0.0**。
我们不追上游的版本号，只**持续吸收上游修复**，冲突在本地解决。

## 远程约定

| 名称       | 地址                                  | 用途               |
| ---------- | ------------------------------------- | ------------------ |
| `origin`   | `https://github.com/iluluyu/Voo.git`  | 我们自己的仓库     |
| `upstream` | `https://github.com/cjpais/Handy.git` | 只读，用于拉取上游 |

首次克隆后确认：

```bash
git remote -v
git fetch upstream --tags
```

## 建议的跟进节奏

每周一次（或上游发版后）：

```bash
# 1. 看上游新提交与新 tag
git fetch upstream --tags
git log --oneline HEAD..upstream/main | head -50
git tag --sort=-creatordate | head -10

# 2. 先只读评估：改到哪些文件、是否碰到我们的核心改动
git diff --stat HEAD..upstream/main

# 3. 在独立分支上合并，绝不在 main 上直接 merge
git switch -c upstream/sync-$(date +%Y%m%d)
git merge --no-ff upstream/main
```

## 冲突高发区（我们改动最多的地方）

以下文件与上游冲突概率最高，合并时**优先保留本地语义**，再把上游修复逐段搬进来：

| 文件                                              | 我们的改动                                            |
| ------------------------------------------------- | ----------------------------------------------------- |
| `src-tauri/src/managers/transcription.rs`         | 词库整合、`/sil` 清理、连续语句、分段提交、上下文注入 |
| `src-tauri/src/audio_toolkit/chinese_hotwords.rs` | 拼音热词、四态模式、倒排索引                          |
| `src-tauri/src/audio_toolkit/audio/sink.rs`       | 电平调理、VAD verdict tap、连续语句跨度               |
| `src-tauri/src/settings.rs`                       | 新配置项（后处理模式、词库四态、上下文等）            |
| `src-tauri/src/llm_client.rs`                     | Responses API                                         |
| `src-tauri/src/managers/lexicon_*.rs`             | 新增模块（上游没有）                                  |
| `src/components/settings/**`                      | 词库页、后处理页、候选 UI                             |

新增文件（上游不存在）一般不会冲突，但**上游改动同一文件的低层结构时会需要手接**。

## 合并后的必跑门禁

```bash
cd src-tauri && cargo test --lib && cargo fmt -- --check && cd ..
bun run build && bun run lint && bun run check:translations
```

四项全绿才算跟进完成；任一项红，先修再合并，不要带着红灯往下叠。

## 版本号策略

- `package.json` / `src-tauri/tauri.conf.json` / `src-tauri/Cargo.toml` 三处**始终一致**，固定 `1.0.0`。
- 上游的版本号不跟；我们的 release tag 用日期或功能名区分（如 `v1.0.0-2026-10`）。
- 每次跟进在提交信息里写清吸收了哪些上游提交范围：`chore(upstream): sync Handy <from>..<to>`。

## 自动更新

安装包的自动更新默认禁用（`tauri.conf.json` 的 updater endpoint 指向占位地址）。跟进上游后需要发版时，再决定是否开启。
