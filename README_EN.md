[中文](README.md) | English

# Voo

**A Chinese-optimized fork of [cjpais/Handy](https://github.com/cjpais/Handy) — press a shortcut, speak, and offline speech-to-text lands right where your cursor is. Built for Chinese users.**

Voo is a cross-platform desktop dictation tool: press a shortcut, speak, release, and the transcribed text is typed into whatever field is focused. Everything runs locally on your machine — no audio or text ever leaves it.

## Inherited from Handy

Voo keeps the full core of upstream [Handy](https://github.com/cjpais/Handy#readme):

- **Fully offline** — your voice stays on your computer
- **VAD** — Silero voice activity detection trims silence automatically
- **Multiple models** — Whisper, Qwen3-ASR, Fun-ASR, SenseVoice, Parakeet and more, with GPU acceleration when available
- **Cross-platform** — Windows / macOS / Linux

## Chinese Enhancements (v0.1)

### Chinese Hot Words

Add Chinese words under **Settings → Advanced → Custom Words** (one per line) and corrections apply immediately — great for names, jargon, and brand terms that ASR tends to get wrong. Words of 3+ hanzi are indexed by default; 2-char words only correct exact homophones.

- **Pinyin-phoneme fuzzy matching**: transcripts and hot words are compared as pinyin phoneme sequences, so homophone errors (同音, e.g. 「亮化」→「量化」) and near-homophone errors (flat/retroflex initials, nasal codas) are both corrected. Polyphonic characters (多音字) expand all candidate readings for matching.
- **Rule replacements**, written alongside hot words:

  | Syntax | Meaning | Example |
  | --- | --- | --- |
  | `word=replacement` | literal replace | `人工智能=AI` |
  | `regex=>replacement` | regex replace | `(?i)chat ?gpt=>ChatGPT` |

  A bare word with no `=` acts as a plain hot word for fuzzy correction.

### Recommended Models

| Model | Strength | Best for |
| --- | --- | --- |
| **Fun-ASR Nano** | speed/accuracy balance | daily default |
| **Qwen3-ASR 1.7B** | highest accuracy | best results, capable hardware |
| **SenseVoice Small** | fastest, lightweight | older machines |

## Install

Download from the [releases page](https://github.com/iluluyu/Voo/releases). On Debian/Ubuntu, install the `.deb` with APT so dependencies are resolved:

```bash
sudo apt install ./Voo_*.deb
```

**Linux notes**: text input needs `xdotool` (X11) or `wtype` / `dotool` (Wayland); if startup complains about `libgtk-layer-shell.so.0`, install `libgtk-layer-shell0` (Debian/Ubuntu) or `gtk-layer-shell` (Fedora/Arch).

For building from source, see [BUILD.md](BUILD.md).

## Roadmap

Long-term goal: evolve dictation into an input method that knows your vocabulary.

1. **Categorized dictionaries** — IT, medicine, law and other lexicons with one-click import and subscription updates
2. **Automatic hot-word correction and cloud hot-word updates**
3. **Learn from your edits** — post-transcription corrections sink into personal hot words
4. **MCP server** — expose hot-word/dictionary management so your own AI can do adaptive correction

## Acknowledgments

- **[cjpais/Handy](https://github.com/cjpais/Handy)** — the upstream project Voo is built on
- **[HaujetZhao/CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline)** — inspiration for the Chinese hot-word algorithm

## License

[MIT](LICENSE). Voo is an independently branded fork of Handy and does not use upstream's name or brand assets.
