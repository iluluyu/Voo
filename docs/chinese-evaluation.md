# 中文输入 A/B 评估

本工具只在本机给**已有转写结果**评分，不运行 ASR、不上传文本，也不自动收集用户录音。
工具单测通过不代表模型准确率、降噪效果或真实输入延迟已经改善。

## 准备数据

同一组固定录音分别在 baseline（原配置）与 candidate（新配置）下转写。
手工核对参考文本，再存为 UTF-8 JSONL（每行一个对象）：

```json
{
  "id": "quiet-001",
  "reference": "请测试模型量化",
  "baseline": "请测试模型亮化",
  "candidate": "请测试模型量化",
  "terms": ["模型量化"],
  "forbidden_terms": ["模型良化"],
  "baseline_latency_ms": 240,
  "candidate_latency_ms": 250
}
```

上例只是格式示意，**不是实测数据**。建议至少包括：

- 人名、项目名、领域术语、英文缩写和中英混合词。
- 同音但不该纠正的负例，数字、否定词与专名。
- 安静/噪声、普通音量/轻声、短句/长句。
- 分别比较无词库、解码词库、解码词库+拼音纠错；降噪开/关只改变一个变量。

`id` 必须唯一。`reference`、`baseline`、`candidate` 是必填字符串，可以为空。
`terms` 是参考文本中应出现的术语，`forbidden_terms` 是参考文本中不应出现、需要监测的误替换词；均可省略。
延迟为结束说话到文本可编辑的毫秒数，可省略，但不能为负数或无穷大。

## 运行

```bash
bun run eval:chinese /path/to/paired-results.jsonl
bun run eval:chinese /path/to/paired-results.jsonl --ignore-punctuation
bun run test:chinese-eval
# 仅验证工具接线的合成示例，不是识别效果数据：
bun run eval:chinese tests/fixtures/chinese-eval.synthetic.jsonl
```

默认仅做 Unicode NFC 归一化，空格、标点、大小写均参与 CER。
`--ignore-punctuation` 额外去除 Unicode 标点和空白，不改写数字、不转换繁简、不折叠大小写。
该选项只影响 CER；术语指标始终按原始字符串精确匹配。

## 指标解释

- **CER**：所有样本的字符级 Levenshtein 编辑数之和 / 参考字符总数；按 Unicode 码点计数，不按 UTF-8 字节或 UTF-16 长度。插入过多时可以大于 1。
- **term_recall**：候选中包含的标注术语数 / 标注术语总数。一个术语在同一样本中只计一次，不代表逐次出现的召回率。
- **forbidden_term_rate**：输出含有的负例词数 / 标注负例词总数。
- **introduced_false_terms/rate**：baseline 不含但 candidate 新增的负例词数 / 全部标注负例词数。它是明确标注范围内的误替换代理指标，不能代表所有潜在误改。
- **improved/regressed_samples**：CER 编辑数相对 baseline 下降/上升的样本数；改善样本和退化样本都应人工复核。
- **latency_p50/p95_ms**：每种配置各自已提供延迟的 nearest-rank 分位数，同时报告 latency_samples；未记录的延迟不当成 0。

没有标注/延迟或参考总字符为 0 时，对应比率/分位数返回 `null`，而不是虚假的满分。
评分输出不包含转写原文；数据文件应由用户自行保存在受保护且不纳入版本管理的位置。
