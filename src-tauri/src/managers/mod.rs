pub mod audio;
pub mod gguf_meta;
pub mod history;
// pub mod learn_capture; // 半成品（agent 触顶，语法未闭合），暂不编译
pub mod learn_corrections;
// 临时摘出：#28 自适应候选仍在开发（其单测未通过），不影响既有链路。
pub mod lexicon_curation;
pub mod lexicon_sources;
pub mod lexicon_suggest;
pub mod llm_context;
pub mod llm_prompt;
pub mod model;
pub mod model_capabilities;
pub mod paste_target;
pub mod pause_commit_runner;
pub mod transcription;
