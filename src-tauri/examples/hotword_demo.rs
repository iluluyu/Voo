//! Voo 中文热词演示：用真实句子验证热词纠错效果
//!
//! 用法：
//!   cargo run --example hotword_demo -- "我在用中州运输入法打字"
//!   cargo run --example hotword_demo                # 交互模式，逐行输入
//!
//! 热词与规则可用环境变量覆盖（空格分隔多个热词）：
//!   HOTWORDS="中州韵输入法 机器学习" RULES="中州韵=中州韵输入法" cargo run --example hotword_demo -- "..."
use voo_app_lib::audio_toolkit::chinese_hotwords::{
    apply_chinese_hotwords, build_hotword_index, parse_hotword_rules, ConfusionOptions,
    DEFAULT_MIN_HAN, DEFAULT_THRESHOLD,
};

fn main() {
    let hotwords: Vec<String> = std::env::var("HOTWORDS")
        .unwrap_or_else(|_| {
            "中州韵输入法 人工智能 机器学习 数据库设计 模型量化 Vue3组件".to_string()
        })
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();
    let rules_text = std::env::var("RULES")
        .unwrap_or_else(|_| "四([个只])=>四条$1\n中州韵=中州韵输入法".to_string());

    let index = build_hotword_index(&hotwords, DEFAULT_MIN_HAN);
    let (rules, rule_errors) = parse_hotword_rules(&rules_text);
    let threshold = std::env::var("THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_THRESHOLD);
    let dialect = std::env::var("DIALECT").map(|v| v == "1").unwrap_or(false);
    let opts = ConfusionOptions { dialect };

    println!("热词索引: {} 条（≥{} 汉字）", index.len(), DEFAULT_MIN_HAN);
    for e in &index {
        println!("  · {}", e.display);
    }
    println!("规则: {} 条（解析失败行: {:?}）", rules.len(), rule_errors);
    println!("阈值: {threshold}  方言混淆: {dialect}");
    println!("{}", "-".repeat(60));

    let process = |text: &str| {
        let (final_text, hits) = apply_chinese_hotwords(text, &index, &rules, threshold, &opts);
        for h in &hits {
            println!(
                "  [拼音] {} → {}  (相似度 {:.2})",
                h.matched, h.replacement, h.score
            );
        }
        if final_text != text {
            println!("  [结果] {text}\n       → {final_text}");
        } else {
            println!("  [结果] （无变化）{text}");
        }
    };

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        println!("输入句子测试（Ctrl+D 退出）：");
        let mut line = String::new();
        while let Ok(n) = std::io::stdin().read_line(&mut line) {
            if n == 0 {
                break;
            }
            let t = line.trim();
            if !t.is_empty() {
                process(t);
            }
            line.clear();
        }
    } else {
        for text in &args {
            process(text);
        }
    }
}
