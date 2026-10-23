//! Voo 词库导入基准：模拟 THUOCL 一键导入后的拼音热词索引规模
//!
//! 用可重复词表把索引撑到 5 万词（11 个分类 × top 5000 的上限规模），
//! 测 `build_hotword_index` 构建耗时，以及 100 条 50 字句子过
//! `apply_pinyin_hotwords` 的总耗时。
//!
//! 用法：
//!   cargo run --example lexicon_bench
use std::time::Instant;
use voo_app_lib::audio_toolkit::chinese_hotwords::{
    apply_pinyin_hotwords, build_hotword_index, ConfusionOptions, DEFAULT_MIN_HAN,
    DEFAULT_THRESHOLD,
};

/// 基准词表规模（模拟 11 分类 × 5000 词全额导入）。
const TARGET_WORDS: usize = 50_000;
/// 待匹配句子数量与长度。
const SENTENCES: usize = 100;
const SENTENCE_CHARS: usize = 50;

/// 无声调同音错字映射（plain 拼音下同音），用于制造待纠错的句子。
/// 正确原文的热词会被保护不替换，必须有错字变体才能测到替换路径。
const HOMOPHONE: &[(char, char)] = &[
    ('学', '雪'),
    ('习', '席'),
    ('工', '公'),
    ('语', '雨'),
    ('图', '途'),
    ('医', '一'),
    ('药', '耀'),
    ('网', '往'),
    ('车', '彻'),
    ('云', '匀'),
];

/// 把词中第 `variant % 可替换数` 个可映射字换成同音错字；无可映射字则原样。
fn corrupt(word: &str, variant: usize) -> String {
    let chars: Vec<char> = word.chars().collect();
    let positions: Vec<usize> = chars
        .iter()
        .enumerate()
        .filter(|(_, c)| HOMOPHONE.iter().any(|(a, _)| a == *c))
        .map(|(i, _)| i)
        .collect();
    if positions.is_empty() {
        return word.to_string();
    }
    let idx = positions[variant % positions.len()];
    let orig = chars[idx];
    let wrong = HOMOPHONE
        .iter()
        .find(|(a, _)| *a == orig)
        .map(|(_, b)| *b)
        .expect("orig came from HOMOPHONE");
    chars
        .iter()
        .enumerate()
        .map(|(i, c)| if i == idx { wrong } else { *c })
        .collect()
}

fn main() {
    // 可重复词表模拟：常见三字以上热词循环填充到 5 万条
    let base: Vec<String> = [
        "人工智能",
        "机器学习",
        "深度学习",
        "自然语言",
        "语音识别",
        "图像识别",
        "数据挖掘",
        "数据库设计",
        "搜索引擎",
        "云计算平台",
        "操作系统",
        "网络安全",
        "自动驾驶",
        "区块链技术",
        "电子商务",
        "智能家居",
        "物联网卡",
        "虚拟现实",
        "增强现实",
        "分布式存储",
        "微服务架构",
        "开源社区",
        "软件开发",
        "程序员日常",
        "产品经理",
        "用户体验",
        "移动支付",
        "社交网络",
        "在线教育",
        "远程办公",
        "数字化转型",
        "智能制造",
        "工业互联网",
        "金融科技",
        "量化交易",
        "风险控制",
        "供应链管理",
        "市场营销",
        "品牌推广",
        "客户服务",
        "项目管理",
        "团队协作",
        "知识图谱",
        "推荐算法",
        "内容审核",
        "数据标注",
        "模型训练",
        "神经网络",
        "视频监控",
        "无人机电",
        "智能音箱",
        "可穿戴设备",
        "医疗影像",
        "基因测序",
        "新药研发",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    let mut words: Vec<String> = Vec::with_capacity(TARGET_WORDS);
    while words.len() < TARGET_WORDS {
        words.extend(base.iter().cloned());
    }
    words.truncate(TARGET_WORDS);

    // 1) 索引构建
    let t0 = Instant::now();
    let index = build_hotword_index(&words, DEFAULT_MIN_HAN);
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
    assert!(!index.is_empty(), "索引为空：基准词表未通过收录条件");

    // 2) 匹配：句子由热词（约半数带同音错字）+ 干扰汉字拼成，长度固定
    let filler: Vec<char> = "的一是了我有不人在这上们中来".chars().collect();
    let sentences: Vec<String> = (0..SENTENCES)
        .map(|i| {
            let mut s = String::new();
            let mut w = i;
            while s.chars().count() + base[w % base.len()].chars().count() <= SENTENCE_CHARS {
                s.push_str(&corrupt(&base[w % base.len()], w));
                w += 3;
                if s.chars().count() < SENTENCE_CHARS {
                    for c in &filler[..2] {
                        if s.chars().count() >= SENTENCE_CHARS {
                            break;
                        }
                        s.push(*c);
                    }
                }
            }
            let deficit = SENTENCE_CHARS - s.chars().count();
            for k in 0..deficit {
                s.push(filler[(i + k) % filler.len()]);
            }
            s
        })
        .collect();
    assert!(
        sentences
            .iter()
            .all(|s| s.chars().count() == SENTENCE_CHARS),
        "句子长度不为 {SENTENCE_CHARS}"
    );

    let opts = ConfusionOptions::default();
    let t1 = Instant::now();
    let mut total_hits = 0usize;
    let mut changed = 0usize;
    for s in &sentences {
        let (out, hits) = apply_pinyin_hotwords(s, &index, DEFAULT_THRESHOLD, &opts);
        total_hits += hits.len();
        if out != *s {
            changed += 1;
        }
    }
    let match_ms = t1.elapsed().as_secs_f64() * 1000.0;

    println!("=== 词库导入基准（THUOCL 规模模拟）===");
    println!("输入词条:      {TARGET_WORDS} 条（可重复词表）");
    println!(
        "索引条目:      {} 条（min_han={DEFAULT_MIN_HAN}）",
        index.len()
    );
    println!("构建耗时:      {build_ms:.1} ms");
    println!("句子:          {SENTENCES} 条 × {SENTENCE_CHARS} 字");
    println!(
        "匹配总耗时:    {match_ms:.1} ms（均 {:.3} ms/句）",
        match_ms / SENTENCES as f64
    );
    println!("命中总数:      {total_hits}（句子有变化: {changed}/{SENTENCES}）");
}
