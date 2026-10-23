#!/usr/bin/env bash
# Voo 模型一次性下载脚本
#
# 模型下载到 HuggingFace 标准缓存（~/.cache/huggingface/hub），
# Voo 会自动发现缓存中的模型（设置 → 模型 → 重新扫描），
# 原版 Handy 及其他 gguf 工具也能共享同一份缓存，无需重复下载。
#
# 用法:
#   ./scripts/download-models.sh <模型> [量化]
#   模型: funasr-nano | funasr-mlt | sensevoice | qwen3-asr-1.7b | qwen3-asr-0.6b
#   量化: Q8_0 (默认) | Q5_K_M | Q6_K | F16
#
# 国内网络加速（镜像）:
#   HF_ENDPOINT=https://hf-mirror.com ./scripts/download-models.sh funasr-nano
#
# 直连下载不稳定时可用 hf transfer:
#   HF_HUB_ENABLE_HF_TRANSFER=1 ... （需 pip install hf_transfer）

set -euo pipefail

MODEL="${1:-}"
QUANT="${2:-Q8_0}"

if [[ -z "$MODEL" ]]; then
  echo "用法: $0 <funasr-nano|funasr-mlt|sensevoice|qwen3-asr-1.7b|qwen3-asr-0.6b> [Q8_0|Q5_K_M|Q6_K|F16]"
  exit 1
fi

case "$MODEL" in
  funasr-nano)    REPO="handy-computer/Fun-ASR-Nano-2512-gguf";    FILE_PREFIX="Fun-ASR-Nano-2512" ;;
  funasr-mlt)     REPO="handy-computer/Fun-ASR-MLT-Nano-2512-gguf"; FILE_PREFIX="Fun-ASR-MLT-Nano-2512" ;;
  sensevoice)     REPO="handy-computer/SenseVoiceSmall-gguf";      FILE_PREFIX="SenseVoiceSmall" ;;
  qwen3-asr-1.7b) REPO="handy-computer/Qwen3-ASR-1.7B-gguf";       FILE_PREFIX="Qwen3-ASR-1.7B" ;;
  qwen3-asr-0.6b) REPO="handy-computer/Qwen3-ASR-0.6B-gguf";       FILE_PREFIX="Qwen3-ASR-0.6B" ;;
  *)
    echo "未知模型: $MODEL"
    echo "可选: funasr-nano | funasr-mlt | sensevoice | qwen3-asr-1.7b | qwen3-asr-0.6b"
    exit 1
    ;;
esac

FILE="${FILE_PREFIX}-${QUANT}.gguf"

if ! command -v huggingface-cli >/dev/null 2>&1 && ! command -v hf >/dev/null 2>&1; then
  echo "错误: 未找到 huggingface-cli。安装方式："
  echo "  pip install -U 'huggingface_hub[cli]'"
  echo "  # 或 uv tool install 'huggingface_hub[cli]'"
  exit 1
fi

# huggingface-cli (旧) 与 hf (新) 命令名兼容
HF_CMD="huggingface-cli"
command -v hf >/dev/null 2>&1 && HF_CMD="hf"

echo "==> 下载 ${REPO} / ${FILE}"
echo "==> 缓存位置: ~/.cache/huggingface/hub （Voo 自动识别，一次下载永久使用）"
[[ -n "${HF_ENDPOINT:-}" ]] && echo "==> 使用镜像: ${HF_ENDPOINT}"

# shellcheck disable=SC2086
exec $HF_CMD download "$REPO" "$FILE" ${HF_HUB_ENABLE_HF_TRANSFER:+--max-workers 8}
