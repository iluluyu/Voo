中文 | [English](README_EN.md)

# Voo

**[cjpais/Handy](https://github.com/cjpais/Handy) 的中文优化 fork —— 按住快捷键说话、本地离线转写、文字直接上屏，面向中文用户的增强版。**

Voo 是一个跨平台桌面语音输入工具：按下快捷键，说话，松开，文字就出现在任何输入框里。识别完全在你的电脑本地完成，不上传任何音频或文字。它在继承 Handy 全部核心能力的基础上，为中文场景补充了热词纠错等增强功能。

## 工作方式

1. **按下** 可自定义的快捷键：按住说话、松开结束，或点按切换录音开关
2. **说话** —— VAD（Silero）自动过滤首尾静音
3. **松开** —— 本地模型完成转写
4. **上屏** —— 文字直接粘贴进当前光标所在的输入框

## 继承自 Handy 的核心特性

Voo 直接继承上游 [Handy](https://github.com/cjpais/Handy) 的核心能力（详情见[上游 README](https://github.com/cjpais/Handy#readme)）：

- **完全离线**：语音不出本机，隐私无忧
- **VAD 静音过滤**：Silero VAD 自动裁掉无声段，转写更快更准
- **多模型**：Whisper 全系、Qwen3-ASR、Fun-ASR、SenseVoice、Parakeet 等，可按需下载切换，支持 GPU 加速
- **跨平台**：Windows / macOS / Linux

## Voo 的中文增强（v0.1）

### 中文热词系统

**拼音热词**：在 设置 → 高级 → **自定义词汇（Custom Words）** 中加入中文词即可生效。人名、术语、品牌名等常被 ASR 写错的词，加入后会被自动纠正（默认拼音后处理仅收录至少 3 个汉字的词，以降低短词误替换风险）。

**拼音音素模糊匹配**：转写结果与热词先转为拼音音素序列再做模糊比对——

- **同音纠错**：如热词为「模型量化」，转写出同音的「模型亮化」会被自动替换回「模型量化」
- **近音纠错**：声母/韵母相近的误识别（平翘舌、前后鼻音等）同样能命中
- **多音字支持**：多音字展开全部候选读音参与匹配，不因一字多音而漏配

**规则替换**：在 设置 → 高级 → **中文热词规则** 文本框中配置（每行一条，在拼音热词之后应用）：

| 写法         | 含义       | 示例                  |
| ------------ | ---------- | --------------------- |
| `词=替换`    | 字面量替换 | `人工智能=AI` |
| `正则=>替换` | 正则替换   | `四([个只])=>四条$1`  |

`#` 开头的行为注释。拼音匹配阈值与方言混淆（n/l、f/h）开关在同一区域。

### 推荐模型

Voo 的模型库中以下列中文模型为推荐（均可在 设置 → 模型 内直接下载）：

| 模型                 | 特点           | 适用                 |
| -------------------- | -------------- | -------------------- |
| **Fun-ASR Nano**     | 速度与精度平衡 | 日常默认之选         |
| **Qwen3-ASR 1.7B**   | 最高精度       | 追求准确率、硬件充裕 |
| **SenseVoice Small** | 低配极速       | 老机器、快速输入     |

## 安装

### 下载安装

从 [Releases](https://github.com/iluluyu/Voo/releases) 页面下载对应平台的安装包。Linux 的 `.deb` 包建议用 APT 安装，以便自动装好依赖：

```bash
sudo apt install ./Voo_*.deb
```

安装后启动 Voo，授予麦克风等系统权限，在设置里配好快捷键即可使用。

### Linux 依赖提示

文字上屏依赖显示服务对应的工具：

| 显示服务 | 推荐工具  | 安装                                           |
| -------- | --------- | ---------------------------------------------- |
| X11      | `xdotool` | `sudo apt install xdotool`                     |
| Wayland  | `wtype`   | `sudo apt install wtype`                       |
| 两者皆可 | `dotool`  | `sudo apt install dotool`（需加入 `input` 组） |

若启动报 `libgtk-layer-shell.so.0` 缺失，安装 `libgtk-layer-shell0`（Debian/Ubuntu）或 `gtk-layer-shell`（Fedora/Arch）。

### 从源码构建

开发与构建指引见 [BUILD.md](BUILD.md)（继承自上游，适用于 Voo）。

## 中文输入评估

修改词库、纠错策略或降噪后，请用同一组固定录音做 A/B，而不是只看是否命中一条例句。
本地评分工具支持字符级 CER、术语召回、标注负例误替换和延迟 P50/P95：

```bash
bun run eval:chinese /path/to/paired-results.jsonl
```

数据格式和指标边界见 [中文输入 A/B 评估](docs/chinese-evaluation.md)。工具自测不等于真实音频识别效果验证。

## Roadmap

Voo 的长期目标——把「听写工具」演进为「懂你词汇的输入法」：

1. **分类词库**：IT、医学、法律等分类词库一键导入（纯汉字 2–50 字），支持订阅 URL 定期更新
2. **热词自动纠错与云端热词更新**
3. **学习修改记录**：根据用户转写后的修改，自动沉淀个人热词
4. **MCP server**：开放热词与词库管理接口，让用户自己的 AI 接入 Voo 做自适应纠错

## 致谢

- **[cjpais/Handy](https://github.com/cjpais/Handy)** —— Voo 的上游与基座，几乎全部核心能力来自它
- **[HaujetZhao/CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline)** —— 中文热词算法思路的启发来源
- 以及上游所依赖与致谢的项目：Whisper、Silero VAD、Tauri 等

## License

[MIT](LICENSE)。Voo 是 Handy 的独立品牌 fork，遵循 MIT 许可发布，不使用上游的名称与品牌资产。
