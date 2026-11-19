# AI Local Translator — SRT & Video Subtitle Translator (Open Source)

**100% local, offline AI subtitle translator.** Translate `.srt` subtitle files between dozens of
languages, and auto-generate subtitles from video/audio — all on your own PC using local
[Ollama](https://ollama.com) models and Whisper. No API keys, no cloud, no character limits.

> This is the **open-source edition** and focuses on **SRT translation** and **video-to-SRT**
> (transcribe + translate). Live web-page translation and manga/webtoon OCR are part of the
> separate PRO edition and are **not** included here.

## ✨ Features

- **Batch SRT translation** — translate many `.srt` files at once, with context-locking and
  zero line-skipping, using any local Ollama text model (Qwen2.5, Gemma, Llama, Phi-4, …).
- **Video / audio → subtitles** — transcribe local files or URLs with
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper), then translate → dual SRT
  (original + translation).
- **Built-in model downloader** — pull and manage Ollama models from inside the app.
- **Runs offline & private** — after the model is downloaded, everything stays on your machine.
- **GPU (NVIDIA CUDA) or CPU** for Whisper speech-to-text.
- **Multi-language UI** — English, Türkçe, 日本語.

## 📦 Requirements

- **Python 3.9+** (tested on 3.11)
- **[Ollama](https://ollama.com)** — runs the local translation models
- **[FFmpeg](https://ffmpeg.org)** — on your PATH (decodes audio/video)
- Python packages: `pip install -r requirements.txt`
- *(Optional)* **[Deno](https://deno.com)** — only for some YouTube downloads via yt-dlp

## 🚀 Install & Run

```bash
# 1) Install Ollama from https://ollama.com and pull a model
ollama pull qwen2.5:7b

# 2) Install Python dependencies
pip install -r requirements.txt

# 3) Run
python srt_video_translator.py
```

Then pick a model in the app, add your `.srt` files (or a video/URL), choose source/target
languages, and start. Translated subtitles are written next to your input files.

## 🧠 Recommended models

Any Ollama text model works. Good starting points:
- `qwen2.5:7b` — balanced & fast
- `gemma2:9b` — natural phrasing for Western languages
- `qwen2.5:14b` — higher quality for long context / anime

## 📜 License

Licensed under the **Apache License 2.0** — see [LICENSE](LICENSE).

Third-party components (Ollama models, FFmpeg, Whisper, yt-dlp, etc.) are downloaded separately
and remain subject to their own licenses.

---

## 🇹🇷 Türkçe

**%100 yerel, çevrimdışı yapay zekâ altyazı çevirmeni.** `.srt` altyazı dosyalarını onlarca dil
arasında çevirin, video/ses dosyalarından otomatik altyazı üretin — hepsi kendi bilgisayarınızda,
yerel [Ollama](https://ollama.com) modelleri ve Whisper ile. API anahtarı yok, bulut yok, karakter
sınırı yok.

> Bu **açık kaynak sürüm**dür; **SRT çevirisi** ve **video→SRT** (transkripsiyon + çeviri)
> özelliklerine odaklanır. Canlı web sayfası çevirisi ve manga/webtoon OCR ayrı PRO sürümün
> parçasıdır ve burada **yer almaz**.

**Özellikler:** Toplu SRT çevirisi (bağlam korumalı, satır atlamasız); video/ses → çift SRT
(faster-whisper ile); uygulama içi model indirici; çevrimdışı ve gizli çalışma; GPU (CUDA) veya
CPU; çok dilli arayüz (TR/EN/JP).

**Gereksinimler:** Python 3.9+, Ollama, FFmpeg (PATH'te), `pip install -r requirements.txt`.

**Çalıştırma:** `python srt_video_translator.py`

**Lisans:** Apache-2.0.
