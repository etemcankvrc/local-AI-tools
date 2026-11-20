# -*- coding: utf-8 -*-
"""
AI Local Image Studio v1.1.0 — %100 yerel yapay zeka görsel oluşturucu.

Görsel tarz AI Local Translator (SRT projesi) ile aynıdır.
Çalıştırma:  python ai_local_image_studio.py

Açılış sırası:
  1) İlk açılışta kullanıcıya "Dosyalar nereye kaydedilsin?" diye sorulur (disklerin boş alanı gösterilir).
  2) Program o klasörde kendi Python ortamını (runtime\\venv) kurar — genel Python'a hiçbir şey yazılmaz.
  3) Arayüz bu ortamda açılır. Görsel motoru, modeller ve LLM'ler yalnızca kullanıcı onayıyla indirilir.
"""
import os
import sys
import json
import time
import random
import shutil
import threading
import queue
import traceback
import subprocess
import locale
import platform
import importlib

APP_NAME = "AI Local Image Studio"
APP_VERSION = "1.3.0"


def _app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


APP_DIR = _app_dir()
SCRIPT_PATH = sys.executable if getattr(sys, "frozen", False) else os.path.abspath(__file__)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import tkinter as tk                              # noqa: E402
from tkinter import ttk, filedialog, messagebox, simpledialog  # noqa: E402
import tkinter.font as tkfont                     # noqa: E402

import ails_core as core                          # noqa: E402

IS_WIN = core.IS_WIN
LANG_DIR = os.path.join(APP_DIR, "lang")

# --- SRT projesiyle birebir aynı renk paleti ---
BG, FRM, TXT, LBL, BTN = "#121212", "#1e1e1e", "#ffffff", "#cccccc", "#333333"
BLUE, ORANGE, RED, WARN, GREEN = "#00aaff", "#D83B01", "#b71c1c", "#ffb74d", "#00e676"
DEV_BG, DEV_BORDER = "#161925", "#2b3248"
CARD_BG, CANVAS_BG = "#181818", "#0d0d0d"
UI_FONT = "Segoe UI" if IS_WIN else "DejaVu Sans"

LANGS = {"English": "en", "Türkçe": "tr", "한국어": "ko", "Deutsch": "de"}

# =============================================================================
# Çoklu dil — İngilizce gömülü (yedek), diğerleri lang/*.json
# Yeni dil eklemek için: lang/xx.json oluştur + LANGS sözlüğüne ekle.
# =============================================================================
EN = {
    # --- ilk açılış: konum seçimi ---
    "loc_title": "AI Local Image Studio — Storage Location",
    "loc_intro": "Choose where the program stores its files. The image engine (~6 GB), every model (2–21 GB each), caches and your images all go into this folder — large files are never written anywhere else. Pick the drive with the most free space.",
    "loc_lang": "Language:",
    "loc_drives": "Drives:",
    "loc_drive": "{drive}   {free:.1f} GB free of {total:.0f} GB",
    "loc_recommended": "✅ recommended",
    "loc_folder": "Folder:",
    "loc_browse": "Browse…",
    "loc_free": "Free space at this location: {free:.1f} GB",
    "loc_warn_space": "⚠️ Only {free:.1f} GB free here. At least {need} GB is recommended.",
    "loc_not_writable": "This folder cannot be written to. Please choose another one.",
    "loc_continue": "✅ Continue",
    "loc_exit": "Exit",
    "loc_continue_anyway": "Continue anyway?",
    "prep_title": "Preparing AI Local Image Studio…",
    "prep_venv": "Creating the program's own Python environment (one time — nothing is installed into your system Python)…",
    "prep_basic": "Installing small basic components (~40 MB)…",
    "prep_fail": "Preparation failed:\n\n{err}",
    "prep_no_venv": "This Python cannot create a virtual environment (the 'venv' / 'ensurepip' modules are missing). Open the program again and answer Yes to the program's own Python, or install 64-bit Python 3.12 from python.org.",
    "prep_old_python": "Python {ver} is too old. Python 3.10 or newer is required.",
    "move_title": "Moving data to the new folder…",
    "move_fail": "Moving failed — the old folder is still used.\n\n{err}",
    "cleanup_title": "Free up space",
    "cleanup_msg": "The previous version left a broken PyTorch installation and a download cache on drive {drive} (~{size:.1f} GB). They are not used anymore.\n\nDelete them to free up space?",
    "cleanup_done": "{size:.1f} GB freed.",
    # --- ana pencere ---
    "btn_style": "🧠 TRAINING / IMAGE ANALYSIS",
    "dev_lbl": "🖥️ Hardware:",
    "hw_gpu": "GPU: {name} ({vram:.1f} GB VRAM)",
    "hw_nogpu": "No NVIDIA GPU found → CPU mode",
    "hw_ram": "RAM: {ram:.0f} GB",
    "hw_page": "Page file: {pf:.0f} GB",
    "hw_disk": "💾 {drive} {free:.1f} GB free",
    "hw_engine_ok": "⚙️ Engine ✔ ({mode})",
    "hw_engine_missing": "⚙️ Engine not installed",
    "profile_lbl": "Profile:",
    "prof_auto": "🤖 Automatic (recommended)",
    "prof_full": "🟢 Full GPU",
    "prof_normal": "🔵 Normal",
    "prof_balanced": "🟡 Balanced",
    "prof_low": "🟠 Low VRAM",
    "prof_cpu": "⚪ CPU only",
    "prof_desc_full": "fast — the whole model stays in VRAM",
    "prof_desc_normal": "model parts are moved to the GPU one by one",
    "prof_desc_balanced": "parts move to the GPU one by one + memory-saving decoding",
    "prof_desc_low": "layer-by-layer loading — slow but stable",
    "prof_desc_cpu": "no GPU acceleration — very slow but works",
    "model_profile_info": "{model} → {profile}: {desc}",
    "sec1": "1. What should be drawn?  (English, Türkçe, 한국어, Deutsch)",
    "use_llm": "🤖 Prepare the prompt with AI (LLM)",
    "show_prompt": "👁️ Show the prompt before generating",
    "sec2": "2. Output Folder",
    "btn_folder": "Select Folder",
    "btn_open_folder": "📂 Open",
    "sec3": "3. Size & Count",
    "lbl_aspect": "Ratio:",
    "lbl_orient": "Orientation:",
    "lbl_res": "Resolution:",
    "lbl_count": "Count:",
    "btn_advanced": "🎛️ Advanced Settings",
    "auto": "🤖 Auto",
    "orient_landscape": "Landscape",
    "orient_portrait": "Portrait",
    "res_low": "Low",
    "res_normal": "Normal",
    "res_high": "High",
    "q_fast": "Fast",
    "q_balanced": "Balanced",
    "q_detailed": "Detailed",
    "sec4": "4. AI Settings",
    "lbl_image_model": "Image model:",
    "lbl_llm": "LLM:",
    "not_installed": "(not downloaded)",
    "lbl_llm_server": "LLM server:",
    "llm_local": "🖥️ This computer",
    "llm_manual": "✏️ Enter IP…",
    "btn_scan": "🔍 Search network",
    "llm_loc_local": "📂 LLM models: {path}  ({free:.1f} GB free)",
    "llm_loc_remote": "🌐 The LLM runs on {host} — this PC's VRAM stays free for images",
    "llm_loc_none": "📂 Ollama is not installed — it is installed on first use (after your approval)",
    "ask_ip_title": "Ollama on the network",
    "ask_ip": "IP address of the computer running Ollama (e.g. 192.168.1.50):",
    "bad_ip": "Please enter a local network address (192.168.x.x, 10.x.x.x or 172.16–31.x.x).",
    "remote_help_title": "Using Ollama on another computer",
    "remote_help": "On the other computer:\n1) Open Ollama → Settings → turn on 'Expose Ollama to the network'.\n2) Allow port 11434 in the Windows Firewall (click 'Allow' if Windows asks).\n3) Both computers must be on the same network.\n\nThen press '🔍 Search network' here or enter the IP manually. If it is not found automatically, entering the IP always works.\n\nBenefit: the LLM runs on the other computer, so all of this PC's VRAM stays free for the image model.",
    "scan_start": "🔍 Searching the network for Ollama (only known neighbour devices, at most {n}, one by one)…",
    "scan_found": "🌐 Ollama found on the network: {hosts} — select it under 'LLM server'.",
    "scan_none": "No Ollama found on the network. You can enter the IP manually.",
    "remote_ok": "🌐 Connected to {host} (Ollama {ver}, {n} models).",
    "remote_fail": "⚠️ {host} could not be reached. New attempts are made at growing intervals (max. 60 s).",
    "remote_fallback": "⚠️ Network Ollama unreachable → using the Ollama on this computer.",
    "sec5": "5. Special Instructions / Things to Avoid",
    "sec5_hint": "LLM on: write freely (e.g. 'no text, no people'). LLM off: used directly as the negative prompt.",
    "sec6": "6. Download / Delete (Engine, Models — Size, PC Fit, License)",
    "btn_download": "Download",
    "btn_delete": "Delete",
    "btn_data_dir": "📁 Data Folder",
    "sec7": "7. Interface Language",
    "btn_start": "🚀 GENERATE IMAGE",
    "btn_stop": "⛔ STOP",
    "chk_shutdown": "🔌 Shut down PC when done",
    "log_title": "System Log",
    "gallery_title": "🖼️ Gallery",
    "gallery_empty": "Generated images will appear here.",
    "btn_show_file": "📂 Show in Folder",
    "btn_upscale": "⬆️ Upscale 2x",
    "btn_copy_prompt": "📋 Copy Prompt",
    "btn_reuse": "♻️ Reuse Settings",
    "progress": "Progress: {p}%",
    "counter": "Image: {i}/{n} | Step: {s}/{S}",
    "status_ready": "Ready.",
    "status_setup": "Checking…",
    "status_llm": "AI is preparing the prompt…",
    "status_loading": "Loading the image model…",
    "status_gen": "Generating…",
    "status_download": "Downloading…",
    "status_install": "Installing…",
    "status_upscale": "Upscaling…",
    "status_scan": "Searching the network…",
    "status_done": "Done.",
    "status_stopped": "Stopped.",
    "status_error": "Error — see the log.",
    "fit_ok": "✅ Runs comfortably",
    "fit_slow": "⚠️ Slow (uses RAM)",
    "fit_no": "❌ Not recommended on this PC",
    "kind_engine": "Engine",
    "kind_image": "Image",
    "kind_llm": "LLM",
    "kind_upscale": "Upscale",
    "engine_name": "Image engine (PyTorch + diffusers)",
    "engine_size": "{dl} download, {disk} on disk",
    "desc_engine": "⚙️ Needed for generating and upscaling",
    "desc_sd15": "🪶 Light & fast — runs even on 4 GB, 512 px",
    "desc_sdxl": "🖼️ Detailed 1024 px — huge ecosystem",
    "desc_zimage": "🔥 Top quality, photorealistic — 8 steps",
    "desc_klein": "⚡ Very fast & modern — 4 steps",
    "desc_qwen3_4b": "🧠 Recommended prompt translator — 4 languages",
    "desc_qwen3_17b": "🪶 Smallest translator — very low VRAM",
    "desc_qwen3_8b": "🎯 Strongest translator — finer Korean/German nuance",
    "desc_qwen25_7b": "🌍 Alternative multilingual translator",
    "desc_esrgan": "⬆️ 2x upscaling — light, runs on 4 GB",
    "adv_title": "🎛️ Advanced Settings",
    "adv_size": "Resulting size:",
    "adv_creativity": "Creativity:",
    "adv_creat_low": "Faithful to prompt",
    "adv_creat_high": "Free interpretation",
    "adv_creat_na": "This model ignores this setting (turbo/distilled model).",
    "adv_quality": "Quality:",
    "adv_steps": "{n} steps",
    "adv_seed": "Seed:",
    "adv_seed_random": "🎲 Random",
    "adv_reset": "Reset to defaults",
    "adv_close": "Close",
    "adv_warn_high": "⚠️ High resolution on this profile can be very slow.",
    "preview_title": "Prompt Preview",
    "preview_pos": "Positive prompt (English):",
    "preview_neg": "Negative prompt:",
    "preview_neg_na": "Negative prompt: not used by this model.",
    "preview_ok": "✅ Generate",
    "preview_cancel": "✖ Cancel",
    "confirm_title": "Download confirmation",
    "confirm_head": "The following will be downloaded:",
    "confirm_item": "• {name} — {size}\n     → {path}",
    "confirm_free": "Free space: {drives}",
    "confirm_q": "Continue?",
    "ollama_name": "Ollama (LLM server)",
    "ollama_size": "~1–2 GB",
    "confirm_remote_pull": "{name} is not installed on {host}.\nDownload it to that computer ({size})?",
    "confirm_remote_delete": "{name} will be deleted from {host}. Continue?",
    "ollama_c_offer": "LLM models are currently saved to:\n{path}  ({free:.1f} GB free)\n\nSave LLM models to the program's data folder instead?\n{target}\n\nOllama is restarted once. Models already downloaded stay where they are.",
    "already_installed": "{name} is already installed.",
    "engine_reinstall_q": "The image engine is already installed. Reinstall / repair it?",
    "confirm_delete": "Delete {name} from disk?",
    "confirm_delete_engine": "The image engine is removed when the program restarts (it can be reinstalled with one click). Continue?",
    "log_hw": "Hardware: {txt}",
    "log_data": "📁 Data folder: {path}",
    "log_pagefile_warn": "⚠️ The Windows page file is small or disabled. If RAM fills up the system may freeze — set it to 'System managed size'.",
    "log_engine_missing": "⚙️ The image engine is not installed yet. It is installed after your approval when you first generate an image (or from section 6).",
    "log_engine_ready": "✅ Image engine ready: PyTorch {torch} ({mode}).",
    "log_engine_install": "⚙️ Installing the image engine into {path}…",
    "log_engine_libs": "Installing diffusers and helper libraries…",
    "log_engine_done": "✅ Image engine installed: PyTorch {torch} ({mode}).",
    "log_torch_index": "Installing PyTorch ({idx})…",
    "log_cuda_fail": "⚠️ The GPU could not be used with the engine — running on the CPU. Updating the NVIDIA driver usually fixes this.",
    "log_ollama_start": "Starting Ollama…",
    "log_ollama_restart": "Restarting Ollama so the new model folder takes effect…",
    "log_ollama_install": "Installing Ollama into {path}…",
    "log_ollama_fail": "⚠️ Ollama unavailable — the prompt is used as written (non-English prompts give weaker results).",
    "log_ollama_manual": "Please install Ollama from https://ollama.com and restart the program.",
    "log_ollama_env_set": "LLM models are now saved to {path}.",
    "log_llm_pull": "Downloading LLM {name}…",
    "log_llm_prompt": "AI prompt → {p}",
    "log_llm_neg": "AI negative → {n}",
    "log_llm_settings": "AI settings → {s}",
    "log_llm_fail": "⚠️ AI prompt preparation failed — using the prompt as written. ({err})",
    "log_llm_unloaded": "Local Ollama models unloaded from VRAM: {names}",
    "log_profile": "Memory profile: {profile} ({desc})",
    "log_loading": "Loading {name}…",
    "log_loaded": "Model ready ({sec:.0f} s).",
    "log_gen_start": "Generating {i}/{n} — {w}×{h}, {steps} steps, seed {seed}",
    "log_saved": "Saved: {path} ({sec:.0f} s)",
    "log_oom": "⚠️ VRAM was not enough → switching to {profile} and retrying.",
    "log_oom_final": "❌ Not enough memory even in the lowest profile. Try a smaller model or lower resolution.",
    "log_ram_guard": "⛔ Memory Guard: free RAM dropped to {free:.1f} GB — the job was stopped safely to protect the system.",
    "err_ram": "❌ Not enough free RAM for this model: needs ~{need:.1f} GB, free {avail:.1f} GB (a reserve is kept for Windows). Close other programs or choose a smaller model.",
    "log_download_start": "Downloading {name} ({size}) → {path}",
    "log_download_done": "✅ {name} downloaded.",
    "log_deleted": "🗑️ {name} deleted.",
    "log_stopped": "⛔ Stopped by user.",
    "log_cancelled": "Cancelled.",
    "log_upscale_done": "✅ Upscaled: {path} ({sec:.0f} s)",
    "log_start": "{app} v{ver} started.",
    "err_disk_full": "❌ Not enough space on drive {drive} — ~{need:.1f} GB needed, {free:.1f} GB free. Free up space or choose another data folder (section 6 → 📁 Data Folder).",
    "err_network": "❌ Internet / network connection problem — the download could not be completed. Check the connection and try again.",
    "err_engine_broken": "❌ The image engine is damaged. Restart the program — it is cleaned up at startup and can then be reinstalled with one click.",
    "err_hf_access": "❌ The model page could not be reached ({err}).",
    "err_generic": "❌ Error: {err}  (details: logs\\ails.log in the data folder)",
    "restart_title": "Restart required",
    "restart_q": "Restart the program now?",
    "data_dir_title": "Data Folder",
    "data_dir_info": "Current data folder:\n{path}\n({free:.1f} GB free)\n\nChoose a new folder? All data is moved there.",
    "data_dir_move_q": "Move ~{size:.1f} GB to:\n{path}\n\nThe program restarts. Ollama and its LLM models are not moved.",
    "data_dir_same": "That is already the current data folder.",
    "data_dir_inside": "The new folder cannot be inside the current data folder.",
    "shutdown_title": "⚠️ Shutdown Scheduled",
    "shutdown_msg": "🔌 Job finished! The PC will shut down in 60 seconds.\nPress the button below or untick the box to cancel.",
    "shutdown_cancel": "❌ CANCEL SHUTDOWN",
    "busy": "Another job is running. Please wait or press STOP.",
    "no_prompt": "Please describe the image first (section 1).",
    "no_image": "Select an image in the gallery first.",
    "licenses_title": "📜 Licenses & Third-Party Notices",
    "lic_intro": "AI Local Image Studio uses only components whose licenses allow commercial use. Models are downloaded by the user from their official sources; this program does not redistribute model weights.",
    "lic_rail": "CreativeML OpenRAIL-M / OpenRAIL++-M (SD 1.5, SDXL): commercial use is allowed, but the license's use-based restrictions (Attachment A) apply to every user and must be respected. Full text: see the model page on Hugging Face.",
    "lic_addons": "Add-ons (LoRA) that you place into the add-ons folder yourself are under your own license responsibility (v2).",
    "prompt_copied": "Prompt copied to clipboard.",
    # --- v1.0.2: kalite, yüz/el düzeltici, v2 hazırlık penceresi ---
    "use_detailer": "🩹 Fix faces & hands",
    "kind_detailer": "Fixer",
    "detailer_name": "Face & Hand Fixer",
    "desc_detailer": "🩹 Finds faces & hands and redraws them sharper — detection on CPU, no extra VRAM",
    "q_ultra": "Ultra",
    "sampler_auto": "🤖 Auto (DPM++ 2M Karras)",
    "sampler_model": "model's own scheduler",
    "hires_on": "✅ On",
    "hires_off": "⛔ Off",
    "on": "on",
    "off": "off",
    "log_det_install": "Installing the face & hand fixer → {path}",
    "log_det_pip": "Installing OpenCV (only this package, ~41 MB download)…",
    "log_det_cv2": "OpenCV {ver} is ready.",
    "log_det_file": "Downloading {name} ({lic})…",
    "log_det_unavailable": "⚠️ The face & hand fixer could not be installed — images are generated without it. ({err})",
    "log_det_skip": "ℹ️ The face & hand fixer is not installed — skipped (you can download it in section 6).",
    "log_quality": "Quality: {q} — {steps} steps • sampler: {sampler} • second pass: {hires} • face/hand fix: {fix}",
    "log_turbo_steps": "ℹ️ This is a turbo (distilled) model: about {n} steps is ideal — many more steps rarely look better.",
    "log_hires": "Second pass (hires fix): {w}×{h}, strength {s}, {steps} steps…",
    "log_fix_found": "Face & hand fixer: {faces} face(s), {hands} hand(s) found.",
    "log_pass_unsupported": "⚠️ Second pass / face & hand fix is not available for {name} with this engine version — skipped.",
    "log_pass_oom": "⚠️ Not enough memory for the extra pass — skipped, the image is kept.",
    "log_pass_fail": "⚠️ Extra pass skipped: {err}",
    "log_saved_partial": "Stopped — the last finished version was saved: {path}",
    "status_hires": "Second pass (adding detail)…",
    "status_fix_detect": "Looking for faces and hands…",
    "status_fix": "Fixing {part} {k}/{K}…",
    "part_face": "face",
    "part_hand": "hand",
    "meta_fix": "🩹 {f} face(s), {h} hand(s)",
    "meta_base": "before fixing",
    "adv_grp_image": "📐 Image",
    "adv_grp_quality": "✨ Quality & fixing",
    "adv_steps_manual": "Steps (manual):",
    "adv_steps_zero": "0 = automatic",
    "adv_steps_manual_info": "✋ {n} steps (manual) — overrides the quality preset.",
    "adv_steps_preset_info": "Preset: {n} steps.",
    "adv_steps_turbo": "Turbo model: about {n} steps is ideal.",
    "adv_sampler": "Sampler:",
    "adv_sampler_info": "DPM++ 2M Karras gives the cleanest faces and hands at 25–40 steps.",
    "adv_sampler_na": "This model uses its own scheduler (turbo / flow model).",
    "adv_hires": "Second pass (hires fix):",
    "adv_hires_info": "→ redrawn at {w}×{h}, strength {s} (sharper faces & hands, ~1.5–2× time)",
    "adv_hires_off_info": "Off for this model / quality. Auto = on for SD 1.5 (Balanced and up) and SDXL (Ultra).",
    "adv_fix": "Face/hand fixer:",
    "adv_fix_on": "🩹 On",
    "adv_fix_faces": "Faces",
    "adv_fix_hands": "Hands",
    "adv_fix_ready": "✔ Installed — runs on the CPU, no extra VRAM.",
    "adv_fix_missing": "Not installed yet — it will be offered with your consent at the next generation ({size}).",
    "adv_fix_off_info": "Off.",
    "adv_auto_neg": "🛡️ Add anatomy negatives automatically (SD 1.5 / SDXL)",
    "adv_save_base": "💾 Also save the version before fixing (for comparison)",
    "adv_warn_hires_slow": "⚠️ The second pass can be very slow on this profile.",
    "v2_title": "🧠 Training / Image Analysis",
    "v2_intro": "Collect images and videos in a dataset (from your computer or, if you allow it, from openly licensed internet sources) and choose what it teaches: a style, a place, a subject or human anatomy. Images without a description are read by the program itself. Then choose: a light word card, a deep LoRA training or self-learning from your 👍/👎 ratings. Everything it creates appears under 🧩 LoRA on the main screen.",
    "v2_sec_dataset": "1. Dataset, type & tag",
    "v2_dataset": "Dataset:",
    "v2_new": "➕ New",
    "v2_delete": "🗑️ Delete",
    "v2_tag": "Tag:",
    "v2_tag_save": "💾 Save",
    "v2_tag_hint": "The name you will pick later when generating, e.g. 'manga style'.",
    "v2_stats": "{n} images • {mb:.1f} MB • from computer: {local}, from internet: {web}, from videos: {video} • created {date}",
    "v2_stats_none": "No dataset selected yet — click ➕ New.",
    "v2_no_dataset": "(no dataset yet)",
    "v2_new_title": "New dataset",
    "v2_new_prompt": "Name of the dataset (e.g. Manga covers):",
    "v2_delete_q": "Delete the dataset '{name}' and its {n} copied images?\n(Your original files are not touched.)",
    "v2_sec_local": "2. Add images from this computer",
    "v2_from_folder": "📁 From a folder…",
    "v2_from_files": "🖼️ Choose files…",
    "v2_recursive": "Include subfolders",
    "v2_copy": "Copy into the data folder (recommended)",
    "v2_min_side": "Skip images smaller than:",
    "v2_local_hint": "PNG, JPG, WEBP, BMP and videos (MP4, MOV, MKV, AVI, WEBM…). Identical images are recognised by their content. If you don't copy, only the file locations are stored (video frames are always saved).",
    "v2_images": "Images",
    "v2_sec_web": "3. Internet",
    "v2_offline": "🔌 Offline mode — this section never connects to the internet",
    "v2_web": "🌐 Allow adding images from the internet (open-licence search + direct links)",
    "v2_download": "⬇️ Download from links",
    "v2_web_limits": "One by one, at least 1 s apart, max {n} links per batch and {mb} MB per image.",
    "v2_web_off_offline": "Offline mode is on — no internet connection is made.",
    "v2_web_off": "Adding from the internet is off.",
    "v2_web_rights": "⚖️ Only add images you have the right to use. Copyright and licence responsibility for added images is yours.",
    "v2_web_rights_title": "Image rights",
    "v2_web_rights_q": "Images downloaded from the internet may be protected by copyright.\n\nI confirm that I have the right to use the images I add and that this responsibility is mine.\n\nContinue?",
    "v2_no_urls": "Paste direct image links (http/https), one per line.",
    "v2_download_q": "{n} link(s) will be downloaded one by one (about {sec:.1f} min or more, max {mb} MB per image).\n\nStart?",
    "v2_sec_analysis": "4. Analysis & learning",
    "v2_mode": "Mode:",
    "v2_mode_light": "Light — word card (fast)",
    "v2_mode_deep": "Deep — LoRA training (SD 1.5)",
    "v2_mode_light_info": "Reads the images and saves the {what} features they share as a word card; works with every image model.",
    "v2_mode_deep_info": "Trains a {what} LoRA on your images (SD 1.5; runs on 4 GB VRAM). {extra} It can also learn a person's face — use real people only with their consent (or yourself).",
    "v2_duration": "Duration:",
    "v2_until_stop": "Until I stop it",
    "v2_minutes": "Time limit",
    "v2_min_unit": "min",
    "v2_checkpoint": "Save progress every:",
    "v2_sample_each": "Generate a sample image at every save",
    "v2_analysis_hint": "The longer it runs, the better the result. Stopping saves the latest state; starting again continues from there.",
    "v2_sec_addons": "📦 Add-ons (LoRA) folder",
    "v2_addons": "{path}\n{n} add-on file(s) (.safetensors) found.",
    "v2_addons_open": "📁 Open folder",
    "v2_addons_note": "The program never downloads add-ons. Files you place here are under your own licence responsibility; they appear under 🧩 LoRA on the main screen.",
    "v2_start": "▶ START",
    "v2_stop": "⏹ Stop",
    "v2_start_empty": "The selected dataset is empty. Add images first (section 2 or 3).",
    "v2_status_scan": "Scanning the folder…",
    "v2_status_import": "Adding images…",
    "v2_status_web": "Downloading images…",
    "v2_prog_import": "Adding images: {a}/{b}",
    "v2_prog_web": "Downloading: {a}/{b}",
    "v2_log_scan": "Scanning: {path}",
    "v2_log_found": "{n} image file(s) found.",
    "v2_log_import_done": "Added: {added} • duplicates: {dup} • too small: {small} • unreadable: {bad} • {mb:.1f} MB copied",
    "v2_log_web_start": "Downloading {n} link(s) one by one…",
    "v2_log_web_done": "Added: {added} • duplicates: {dup} • too small: {small} • not an image: {bad} • failed: {failed} • {mb:.1f} MB",
    "v2_log_web_stopped": "⚠️ Stopped after 5 failed connections in a row (the network is not being forced).",
    "v2_log_created": "Dataset created: {name} → {path}",
    "v2_log_tag": "Tag '{tag}' saved for {name}.",
    # --- v1.1.0: LoRA, görsel okuma, öğrenme ---
    "v2_mode_self": "🧠 Self-learning — reward/penalty (👍/👎)",
    "v2_mode_self_info": "Learns from your 👍 (reward) and 👎 (penalty) in the gallery; with the auto-judge it also generates images itself and rewards the better one. Result: the 'AILS tercih' LoRA (SD 1.5).",
    "v2_captions": "Descriptions:",
    "v2_autocap": "🔍 Read images without a description automatically (Florence-2)",
    "v2_caption_now": "🔍 Read now",
    "v2_caps_count": "{k} / {n} images have a description.",
    "v2_rewards": "Rewards:",
    "v2_rewards_count": "👍 {up}  ·  👎 {down}  ·  pairs: {pairs}",
    "v2_auto_judge": "🤖 Auto-judge (generates and scores images itself)",
    "v2_self_empty": "There is nothing to learn yet: rate images in the gallery with 👍 / 👎 (ideally several for the same prompt), or turn on the auto-judge.",
    "v2_status_caption": "Reading images…",
    "v2_prog_caption": "Reading images: {a}/{b}",
    "v2_log_caption_load": "Loading the image reader (Florence-2) on the {device}…",
    "v2_log_caption_done": "Descriptions written: {done} • failed: {failed}",
    "v2_status_card": "Creating the word card…",
    "v2_log_card_nocap": "⚠️ No image has a description yet — turn on automatic reading (Florence-2).",
    "v2_log_card_heuristic": "ℹ️ No running LLM — the word card is built from the words most descriptions share.",
    "v2_log_card_empty": "⚠️ No clear common features were found in the descriptions.",
    "v2_log_card_done": "{icon} Word card '{name}' ({kind}) saved: {style}",
    "v2_log_card_avoid": "   avoid: {avoid}",
    "v2_status_train": "Training…",
    "tr_counter": "{steps} steps · {time}",
    "tr_log_load": "Loading SD 1.5 for training ({device})…",
    "tr_log_resume": "Continuing the previous training ({steps} steps so far).",
    "tr_log_encode": "Preparing {n} image(s)…",
    "tr_log_train": "Training started — trigger word: '{trigger}'. Press ⏹ Stop whenever you like; the latest state is saved.",
    "tr_log_step": "Step {steps} — loss {loss:.4f}",
    "tr_log_saved": "💾 Saved (step {steps}, loss {loss:.4f}).",
    "tr_log_saved_final": "💾 Final state saved (step {steps}, loss {loss:.4f}).",
    "tr_log_sample_oom": "⚠️ Not enough memory for the sample image — skipped.",
    "tr_log_pref": "Self-learning: {pairs} 👍/👎 pair(s), {singles} single 👍, auto-judge: {judge}.",
    "tr_log_pref_saved": "💾 Preference LoRA saved — step {steps}, auto-judged pairs {judged}, reward accuracy {acc:.0f}%.",
    "tr_log_sample": "🖼️ Sample image: {path}",
    "tr_log_nothing": "⚠️ Nothing to learn: the dataset is empty or there are no ratings.",
    "tr_log_oom": "⚠️ Not enough graphics memory — trying again at a lower resolution (384 px)…",
    "tr_log_done": "✅ '{name}' is ready ({steps} steps): {path} — choose it with 🧩 LoRA on the main screen.",
    "tr_log_stopped_saved": "⏹ Stopped — the latest state was saved.",
    "btn_lora": "🧩 LoRA / Card",
    "lora_none": "No LoRA or card selected.",
    "lora_incompat_note": "(⚠ doesn't fit this model)",
    "lp_title": "🧩 LoRA & Card Selection",
    "lp_model": "Image model: {model}",
    "lp_f_compatible": "Fits this model",
    "lp_f_all": "All",
    "lp_f_own": "Created by the program",
    "lp_f_external": "Added by me",
    "lp_f_cards": "Word cards",
    "lp_col_name": "Name",
    "lp_col_kind": "Source",
    "lp_col_base": "Model",
    "lp_col_size": "Size",
    "lp_col_info": "Trigger / tags",
    "lp_card": "📝 Word card",
    "lp_own": "🧠 Program",
    "lp_own_pref": "🧠 Preference (👍/👎)",
    "lp_external": "📦 Add-on",
    "lp_toggle": "✔ Select / remove",
    "lp_delete": "🗑️ Delete",
    "lp_open_addons": "📁 Add-ons folder",
    "lp_open_own": "📁 Program's LoRAs",
    "lp_selected": "Selected (strength)",
    "lp_card_note": "adds its words to the prompt",
    "lp_note": "{own} created by the program, {ext} added by you — {shown} shown. Double-click to select.",
    "lp_note_peft": "ℹ️ Using LoRAs needs the small PEFT package ({size}); it is installed with your consent at the next generation.",
    "lp_clear": "Clear selection",
    "lp_ok": "✅ OK",
    "lp_max": "You can select at most {n} LoRAs/cards.",
    "lp_delete_q": "Delete {kind} '{name}'?\nFile: {file}\n\nIt is not deleted permanently: it is moved to the loras\\.trash folder, where you can get it back.",
    "peft_name": "LoRA support (PEFT, Apache-2.0)",
    "log_peft_install": "Installing LoRA support (PEFT)…",
    "log_peft_done": "✅ LoRA support installed.",
    "log_peft_restart": "ℹ️ LoRA support is installed but becomes active after restarting the program.",
    "peft_restart_q": "LoRA support was installed. The program needs to restart once to use it.\n\nRestart now?",
    "log_lora_skip_missing": "⚠️ LoRA '{name}' was not found (deleted or moved) — skipped.",
    "log_lora_skip_incompatible": "⚠️ LoRA '{name}' does not fit the selected image model — skipped.",
    "log_lora_off": "ℹ️ This image is generated without LoRAs.",
    "log_lora_applied": "🧩 LoRA: {names}",
    "log_lora_fail": "⚠️ LoRA '{name}' could not be loaded: {err}",
    "log_rated_up": "👍 Reward saved — used in self-learning.",
    "log_rated_down": "👎 Penalty saved — used in self-learning.",
    "log_rated_clear": "Rating removed.",
    "meta_liked": "👍 liked",
    "meta_disliked": "👎 disliked",
    "gal_sample": "sample {steps}",
    "kind_caption": "Reader",
    "desc_captioner": "🔍 Reads images and writes descriptions (for training) — small, runs on 4 GB",
    "cap_unsupported": "The installed image engine is too old for the image reader (Florence-2). Reinstall the engine in section 6 (⚙️ Image engine → Download) and try again.",
    "adv_pref": "Preference LoRA:",
    "adv_pref_auto": "🧠 Apply automatically (SD 1.5)",
    "adv_pref_info": "✔ Trained: {steps} steps, {judged} auto-judged pairs ({date}).",
    "adv_pref_none": "Not trained yet — Training window → mode 🧠 Self-learning.",
    "adv_pref_sd15_only": "Only used with SD 1.5.",
    # --- v1.0.3: Openverse araması ---
    "v2_search_lbl": "🔎 Search:",
    "v2_search_btn": "🔎 Search & add",
    "v2_search_max": "Max:",
    "v2_cat": "Type:",
    "v2_cat_all": "All",
    "v2_cat_illustration": "Illustration",
    "v2_cat_digitized_artwork": "Artwork",
    "v2_cat_photograph": "Photo",
    "v2_lic": "Licence:",
    "v2_lic_commercial": "Commercial OK (CC0, PD, CC BY)",
    "v2_lic_commercial_sa": "+ share-alike (CC BY-SA)",
    "v2_search_note": "Source: Openverse (Wikimedia, Flickr, museums…) — openly licensed images only. Creator, licence and source of every image are saved with the dataset. Without an account Openverse allows about 5 search pages (100 results) per hour.",
    "v2_links_lbl": "Or direct image links (one per line):",
    "v2_no_query": "Type what to search for, e.g. 'watercolor landscape' or 'ukiyo-e'.",
    "v2_search_q": "Search Openverse for '{q}' and add up to {n} openly licensed images to the dataset?\n\nRequests are made one by one, at least 1 s apart.",
    "v2_log_search": "Openverse search: '{q}' — {lic}, {cat}",
    "v2_log_search_tr": "AI search words: {q}",
    "v2_log_search_found": "{n} suitable image(s) selected (Openverse found {total}) — downloading one by one…",
    "v2_log_search_none": "Openverse found no suitable images. Try other words (English works best) or another type.",
    "v2_log_rate": "⚠️ Openverse's hourly limit for searches without an account was reached — please try again in about an hour.",
    "v2_log_rate_left": "Openverse: {n} search request(s) left in the current window.",
    "v2_log_no_llm": "ℹ️ No running LLM — the search text is used as typed (English words find more).",
    "v2_log_search_fail": "⚠️ Openverse search failed: {err}",
    "v2_status_search": "Searching Openverse…",
    # --- v1.2.0: kurulum, video kareleri, öğretilen tür ---
    "rt_title": "Program environment",
    "rt_ask": "AI Local Image Studio sets up its own Python environment once.\n\nRecommended: download the program's own Python {pyv} (~{size} MB: the 'uv' tool + Python; licences MIT/Apache-2.0 and PSF). It goes into the data folder ({path}) — nothing is written to Windows, PATH or the registry — and it does not depend on the Python installed on this computer ({ver}) or its updates.\n\nYes: the program's own Python (recommended)\nNo: use this computer's Python {ver}",
    "rt_need_new": "The Python on this computer ({ver}) is too new: the image engine's packages (PyTorch etc.) are not available for it yet.\n\nThe program can download its own Python {pyv} (~{size} MB) into the data folder ({path}). Download it?",
    "rt_need_old": "The Python on this computer ({ver}) is too old for the image engine.\n\nThe program can download its own Python {pyv} (~{size} MB) into the data folder ({path}). Download it?",
    "rt_need_32bit": "The Python on this computer ({ver}) is 32-bit; the image engine needs 64-bit Python.\n\nThe program can download its own 64-bit Python {pyv} (~{size} MB) into the data folder ({path}). Download it?",
    "rt_dl": "Downloading the program's own Python {pyv} (~{size} MB)…",
    "rt_ok": "✅ The program's own Python is ready: {path}",
    "rt_fallback": "⚠️ The program's own Python could not be downloaded ({err}) — continuing with this computer's Python {ver}.",
    "rt_cancelled": "Setup was cancelled. You can open the program again at any time.",
    "prep_fail_try_portable": "Preparation with this computer's Python {ver} failed:\n\n{err}\n\nShould the program download its own Python {pyv} (~{size} MB) and try again?",
    "prep_details": "Details:",
    "prep_log_at": "Setup log: {path}",
    "err_install": "❌ Installation failed.",
    "err_install_nomatch": "❌ No compatible package was found for this computer.",
    "hint_longpath": "Windows long-path support is off. Fix: move the data folder to a short path (e.g. D:\\AILS) or turn on long paths in Windows (registry HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem → LongPathsEnabled = 1) and restart the computer.",
    "hint_access": "A file was locked or access was denied — usually by an antivirus scan. Open the program again in a few minutes; if it keeps happening, add the data folder to your antivirus exclusions.",
    "hint_build": "No ready-made package exists for this Python version (pip tried to compile one). Use the program's own Python — the program offers it.",
    "hint_nomatch": "No package exists for this Python version or processor type. Use 64-bit Python 3.11–3.13 or the program's own Python.",
    "hint_ssl": "The secure connection could not be verified — an antivirus 'HTTPS scanning' feature or a company proxy may be in the way. Turn HTTPS scanning off for a moment and try again.",
    "learn_style": "🎨 Style (watercolor, manga…)",
    "learn_place": "🏙️ Place / city (Istanbul streets…)",
    "learn_subject": "🧸 Subject / object / character",
    "learn_anatomy": "🧍 Human anatomy / pose",
    "learn_short_style": "Style",
    "learn_short_place": "Place",
    "learn_short_subject": "Subject",
    "learn_short_anatomy": "Anatomy",
    "learn_what_style": "style",
    "learn_what_place": "place",
    "learn_what_subject": "subject",
    "learn_what_anatomy": "anatomy and pose",
    "v2_learn": "What to teach:",
    "v2_tag_hint_style": "The name you will pick later when generating, e.g. 'manga style'.",
    "v2_tag_hint_place": "Name of the place, e.g. 'istanbul street' — it becomes the trigger word.",
    "v2_tag_hint_subject": "Name of the subject, e.g. 'red robot'.",
    "v2_tag_hint_anatomy": "A short name, e.g. 'correct hands'.",
    "v2_log_learn": "'{name}' now teaches: {kind}.",
    "v2_deep_note_style": "Images are also learned mirrored.",
    "v2_deep_note_place": "No mirroring (signs and traffic stay the right way round); wide video frames are also learned from both ends. With 'Hide faces' on, faces are blurred and not learned and people are learned less, so the surroundings come first.",
    "v2_deep_note_subject": "No mirroring, so asymmetric details stay correct.",
    "v2_deep_note_anatomy": "Learns more gently (lower learning rate) and also mirrored.",
    "lp_card_kind": "Card · {kind}",
    "lp_own_kind": "Program · {kind}",
    "tr_log_kind": "What is taught: {kind}",
    "tr_log_views": "({v} crops incl. mirrored/end crops)",
    "v2_vframes": "🎬 From each video at most:",
    "v2_vframes_unit": "frames",
    "v2_media": "Images and videos",
    "v2_videos": "Videos",
    "v2_log_found_media": "{n} image file(s) and {v} video(s) found.",
    "v2_videos_q": "The selection contains {n} video(s) ({gb:.2f} GB).\n\nSplit them into frames and add them to the dataset? From each video at most {k} sharp, varied frames are taken (blurry, dark and repeated scenes are skipped). The frames are named after the dataset and described automatically.\n\nNo: the videos are skipped.",
    "v2_log_videos_skipped": "{n} video(s) skipped.",
    "v2_log_videos_nocv2": "{n} video(s) skipped — OpenCV is needed to read videos.",
    "v2_status_video": "Extracting frames from videos…",
    "v2_log_video_start": "🎬 {i}/{n} {name} — choosing up to {k} frames…",
    "v2_prog_video": "Video {i}/{n}: {p}%",
    "v2_log_video_done": "🎬 {name}: {added} frame(s) added ({mb:.1f} MB, {min:.1f} min of video) — {similar} similar and {unusable} blurry/dark section(s) skipped.",
    "v2_log_video_small": "🎬 {name}: the video is smaller than {px} px — skipped (lower 'Skip images smaller than').",
    "v2_log_video_bad": "🎬 {name}: the video could not be read (unsupported or damaged).",
    "v2_log_video_fail": "⚠️ {name}: {err}",
    "v2_log_import_total": "✅ {added} new image(s) in the dataset.",
    "v2_log_named": "Files are named after the dataset: {base}_0001.jpg …",
    "v2_log_caption_auto": "📝 New images are being described automatically…",
    "v2_log_caption_later": "ℹ️ The descriptions will be written when you press ▶ START (or 🔍 Read now) — the image engine is installed then.",
    "v2_hide_faces": "🙈 Hide faces, focus on the surroundings (Place)",
    "tr_log_faces": "🙈 Hiding faces: {faces} face(s) found in {images} image(s). Faces are blurred and not learned, people are learned less — the surroundings come first.",
    "tr_log_hide_off": "⚠️ This needs the face detector (installed with the Face & Hand Fixer); it isn't installed, so training was not started.",
    "tr_log_keep_old": "ℹ️ '{old}' belongs to another training (a different dataset or type) — it is kept. The new training is saved as '{new}'.",
    "lp_kind_card": "Word card",
    "lp_kind_lora": "LoRA",
    "log_trashed": "🗑️ {kind} '{name}' moved to the trash folder: {path}",
    "lp_hidden": "({n} LoRA(s) hidden because they don't fit this model — set the filter to 'All' to see them.)",
    "learn_face": "👤 Face (a person or character)",
    "learn_short_face": "Face",
    "learn_what_face": "face",
    "v2_tag_hint_face": "Name of the person or character, e.g. 'my face' — it becomes the trigger word.",
    "v2_deep_note_face": "Only the face (with hair) is cut out of each image and learned; background and clothes are barely learned. Images without a detected face are skipped.",
    "v2_face_consent_q": "Face mode learns what a person looks like.\n\nAre these faces yours, people who agreed to it, or fictional characters? Do not use it to imitate real people without their consent.\n\nYes: continue (not asked again for this dataset)",
    "tr_log_face_mode": "👤 Face found and cut out in {n} image(s); {skipped} image(s) without a face were skipped.",
}


class I18N:
    def __init__(self):
        self.code = "en"
        self.data = {}

    def set(self, code):
        self.code = code
        self.data = {}
        if code != "en":
            try:
                with open(os.path.join(LANG_DIR, f"{code}.json"), encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}

    def t(self, key, **kw):
        s = self.data.get(key) or EN.get(key) or key
        if kw:
            try:
                s = s.format(**kw)
            except Exception:
                pass
        return s


T = I18N()


def t(key, **kw):
    return T.t(key, **kw)


def ensure_en_json():
    """Çevirmenler için lang/en.json her zaman güncel kalsın."""
    try:
        os.makedirs(LANG_DIR, exist_ok=True)
        p = os.path.join(LANG_DIR, "en.json")
        cur = None
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                cur = json.load(f)
        if cur != EN:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(EN, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def detect_lang():
    try:
        loc = (locale.getlocale()[0] or "").lower()
    except Exception:
        loc = ""
    return ("Türkçe" if loc.startswith(("tr", "turkish")) else
            "한국어" if loc.startswith(("ko", "korean")) else
            "Deutsch" if loc.startswith(("de", "german")) else "English")


# =============================================================================
# Konum işaretçisi (veri klasörü nerede?) ve ayarlar
# =============================================================================
def _pointer_path():
    if core.is_writable_dir(APP_DIR):
        return os.path.join(APP_DIR, "location.json")
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".local", "share")
    d = os.path.join(base, APP_NAME)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "location.json")


POINTER_PATH = _pointer_path()


def load_pointer():
    try:
        with open(POINTER_PATH, encoding="utf-8") as f:
            d = json.load(f)
            return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save_pointer(cfg):
    try:
        with open(POINTER_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


DEFAULT_SETTINGS = {
    "lang": None, "model": "sdxl", "llm": core.DEFAULT_LLM, "llm_host": "local", "known_hosts": [],
    "profile": "auto", "use_llm": True, "show_prompt": False,
    "aspect": "auto", "orient": "auto", "res": "auto", "quality": "auto",
    "creativity": 5, "creativity_auto": True, "seed_random": True, "seed": "12345", "count": "1",
    "output_dir": None, "ollama_offer_done": False,
    # v1.0.2 — kalite: elle adım, örnekleyici, ikinci geçiş, yüz/el düzeltici, anatomi negatifleri
    "steps_manual": 0, "sampler": "auto", "hires": "auto", "detailer": True, "fix_faces": True, "fix_hands": True,
    "auto_neg": True, "save_base": False,
    # LoRA / stil kartı seçimleri ve kendi kendine öğrenilen tercih LoRA'sı
    "loras": [], "pref_auto": True, "pref_weight": 0.6,
    # Stil Öğrenme / Görsel Analiz penceresi ayarları
    "v2": {},
}
V2_DEFAULTS = {
    "offline": True, "web_import": False, "web_ack": False, "dataset": "", "recursive": True, "copy": True,
    "min_side": 256, "mode": "light", "base_model": "sd15", "duration": "until_stop", "minutes": 60,
    "checkpoint": 15, "sample_each": True, "last_folder": "",
    "last_query": "", "search_count": 40, "search_cat": "all", "search_lic": "commercial",
    "auto_caption": True, "auto_judge": True, "learn": "style", "video_frames": 120, "hide_faces": True,
}


def load_settings(paths):
    s = dict(DEFAULT_SETTINGS)
    data, from_new = None, False
    for p, is_new in ((paths.settings, True), (os.path.join(APP_DIR, "settings.json"), False)):
        try:
            with open(p, encoding="utf-8") as f:
                data, from_new = json.load(f), is_new
            break
        except Exception:
            continue
    if isinstance(data, dict):
        for k in DEFAULT_SETTINGS:
            if k in data and k != "output_dir":
                s[k] = data[k]
        if from_new and data.get("output_dir"):
            s["output_dir"] = data["output_dir"]
    if not s.get("output_dir"):
        s["output_dir"] = paths.outputs
    if s.get("lang") not in LANGS:
        s["lang"] = load_pointer().get("lang") or detect_lang()
    if s["model"] not in core.IMAGE_MODEL_BY_KEY:
        s["model"] = "sdxl"
    if s["llm"] not in core.LLM_MODEL_BY_KEY:
        s["llm"] = core.DEFAULT_LLM
    if s.get("quality") not in ("auto",) + core.QUALITIES:
        s["quality"] = "auto"
    if s.get("sampler") not in core.SAMPLERS:
        s["sampler"] = "auto"
    if s.get("hires") not in ("auto", "on", "off"):
        s["hires"] = "auto"
    try:
        s["steps_manual"] = max(0, min(core.MAX_STEPS, int(s.get("steps_manual") or 0)))
    except (TypeError, ValueError):
        s["steps_manual"] = 0
    v2 = dict(V2_DEFAULTS)
    if isinstance(s.get("v2"), dict):
        v2.update({k: v for k, v in s["v2"].items() if k in V2_DEFAULTS})
    if v2.get("mode") not in ("light", "deep", "self"):
        v2["mode"] = "light"
    if v2.get("base_model") not in core.IMAGE_MODEL_BY_KEY:
        v2["base_model"] = "sd15"
    if v2.get("duration") not in ("until_stop", "minutes"):
        v2["duration"] = "until_stop"
    if v2.get("search_count") not in core.SEARCH_COUNTS:
        v2["search_count"] = 40
    if v2.get("search_cat") not in core.OPENVERSE_CATEGORIES:
        v2["search_cat"] = "all"
    if v2.get("search_lic") not in core.OPENVERSE_LICENSES:
        v2["search_lic"] = "commercial"
    v2["learn"] = core.norm_learn(v2.get("learn"))
    if v2.get("video_frames") not in core.VIDEO_FRAME_CHOICES:
        v2["video_frames"] = 120
    s["v2"] = v2
    clean = []
    for it in (s.get("loras") or [])[:8]:
        try:
            clean.append({"id": str(it["id"]), "w": max(0.0, min(2.0, float(it.get("w", 0.8))))})
        except Exception:
            continue
    s["loras"] = clean
    try:
        s["pref_weight"] = max(0.0, min(1.5, float(s.get("pref_weight", 0.6))))
    except (TypeError, ValueError):
        s["pref_weight"] = 0.6
    s["pref_auto"] = bool(s.get("pref_auto", True))
    hosts = [core.normalize_host(h) for h in (s.get("known_hosts") or []) if isinstance(h, str)]
    s["known_hosts"] = [h for h in hosts if h and not h.startswith("127.")][:12]
    if s.get("llm_host") != "local":
        h = core.normalize_host(str(s.get("llm_host", "")))
        s["llm_host"] = h if h and not h.startswith("127.") else "local"
        if s["llm_host"] != "local" and s["llm_host"] not in s["known_hosts"]:
            s["known_hosts"].insert(0, s["llm_host"])
    return s


def save_settings(paths, s):
    try:
        with open(paths.settings, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def friendly_error(e):
    """Teknik hatayı kullanıcının anlayacağı tek bir cümleye çevirir (ayrıntı log dosyasına yazılır)."""
    if isinstance(e, core.DiskFullError):
        return t("err_disk_full", drive=e.drive, need=e.need_gb, free=max(0.0, e.free_gb))
    if isinstance(e, core.NetworkError):
        return t("err_network")
    if isinstance(e, core.EngineBrokenError) or core.looks_like_engine_broken(e):
        return t("err_engine_broken")
    if isinstance(e, core.InstallError):
        parts = [t("err_install", err=str(e))]
        if getattr(e, "detail", ""):
            parts.append(t("prep_details") + "\n" + e.detail)
        if getattr(e, "hint", None):
            parts.append("💡 " + t("hint_" + e.hint))
        return "\n".join(parts)
    if isinstance(e, OSError) and getattr(e, "errno", None) == 28:
        return t("err_disk_full", drive="?", need=0.0, free=0.0)
    name = type(e).__name__
    lname, mod = name.lower(), (type(e).__module__ or "")
    if any(k in lname for k in ("connecterror", "connectionerror", "timeout", "networkerror", "remoteprotocol",
                                "localentrynotfound", "offline")) or \
            (mod.startswith(("httpx", "httpcore", "urllib3", "requests")) and "error" in lname):
        return t("err_network")
    if name in ("GatedRepoError", "RepositoryNotFoundError", "HfHubHTTPError", "HTTPError", "EntryNotFoundError"):
        return t("err_hf_access", err=str(e).strip().splitlines()[0][:160] if str(e).strip() else name)
    text = str(e).strip()
    msg = text.splitlines()[0][:200] if text else name
    return t("err_generic", err=f"{name}: {msg}")


# =============================================================================
# BAŞLATICI — yalnızca standart kütüphane (henüz hiçbir paket kurulu olmayabilir)
# =============================================================================
def _dark_tk(title, geometry=None):
    root = tk.Tk()
    root.title(title)
    root.configure(bg=BG)
    if geometry:
        root.geometry(geometry)
    try:
        st = ttk.Style(root)
        st.theme_use("clam")
        st.configure("TCombobox", font=(UI_FONT, 9))
    except Exception:
        pass
    for p in (os.path.join(APP_DIR, "simge.ico"),):
        if os.path.exists(p):
            try:
                root.iconbitmap(p)
            except Exception:
                pass
    return root


def _box(kind, title, msg):
    root = tk.Tk()
    root.withdraw()
    try:
        if kind == "error":
            messagebox.showerror(title, msg, parent=root)
            return None
        if kind == "info":
            messagebox.showinfo(title, msg, parent=root)
            return None
        return messagebox.askyesno(title, msg, parent=root)
    finally:
        root.destroy()


class TaskWindow:
    """Başlatıcı aşamasındaki işler için küçük ilerleme penceresi (kapatılamaz, iş bitince kapanır)."""

    def __init__(self, title, headline, paths=None):
        self.paths = paths            # verilirse her satır logs\setup.log'a da yazılır
        self.root = _dark_tk(title, "660x320")
        self.root.resizable(False, False)
        tk.Label(self.root, text=headline, bg=BG, fg=BLUE, font=(UI_FONT, 11, "bold"), wraplength=620,
                 justify="left", anchor="w").pack(fill="x", padx=16, pady=(14, 8))
        self.pg = ttk.Progressbar(self.root, mode="indeterminate", maximum=100)
        self.pg.pack(fill="x", padx=16)
        self.lbl = tk.Label(self.root, text="", bg=BG, fg=LBL, font=(UI_FONT, 9), anchor="w")
        self.lbl.pack(fill="x", padx=16, pady=(4, 0))
        self.txt = tk.Text(self.root, height=9, bg="#111111", fg="#00ff00", font=("Consolas" if IS_WIN else "DejaVu Sans Mono", 9),
                           relief="flat", wrap="word")
        self.txt.pack(fill="both", expand=True, padx=16, pady=(6, 14))
        self.q = queue.Queue()
        self.result = ("err", RuntimeError("not run"))
        self.root.protocol("WM_DELETE_WINDOW", lambda: None)

    def log(self, s):
        self.q.put(("log", s))
        if self.paths is not None:
            core.setup_log(self.paths, s)

    def progress(self, done, total):
        self.q.put(("pg", (done, total)))

    def _poll(self):
        try:
            while True:
                kind, v = self.q.get_nowait()
                if kind == "log":
                    self.txt.insert(tk.END, str(v) + "\n")
                    self.txt.see(tk.END)
                elif kind == "pg":
                    done, total = v
                    if total and total > 0:
                        if str(self.pg["mode"]) != "determinate":
                            self.pg.stop()
                            self.pg.config(mode="determinate")
                        self.pg["value"] = min(100.0, done * 100.0 / total)
                        self.lbl.config(text=f"{done / core.GB:.2f} / {total / core.GB:.2f} GB")
                    else:
                        if str(self.pg["mode"]) != "indeterminate":
                            self.pg.config(mode="indeterminate")
                            self.pg.start(12)
                        self.lbl.config(text="")
                elif kind == "done":
                    # Önce çubuğun animasyonu durdurulur; yoksa Tk kapandıktan sonra bekleyen 'after' komutu
                    # konsola "application has been destroyed" hatası basar.
                    try:
                        self.pg.stop()
                    except Exception:
                        pass
                    self.root.destroy()
                    return
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def run(self, fn):
        def work():
            try:
                self.result = ("ok", fn(self))
            except BaseException as e:  # noqa
                self.result = ("err", e)
            self.q.put(("done", None))

        threading.Thread(target=work, daemon=True).start()
        self.pg.start(12)
        self.root.after(100, self._poll)
        self.root.mainloop()
        return self.result


def _legacy_suggestion():
    """v1.0.0'da model klasörü başka bir diskte seçildiyse o diski öner."""
    try:
        with open(os.path.join(APP_DIR, "settings.json"), encoding="utf-8") as f:
            md = json.load(f).get("model_dir")
        if md and IS_WIN:
            drive = os.path.splitdrive(md)[0]
            if drive and os.path.exists(drive + "\\"):
                return os.path.join(drive + "\\", core.DEFAULT_ROOT_NAME)
    except Exception:
        pass
    return None


def first_run_dialog(cfg):
    """İlk açılışta veri klasörünü (ve dili) sorar. Dönüş: (klasör, dil) ya da None."""
    drives = core.list_drives()
    best_drive = max(drives, key=lambda d: d[1])[0] if drives else None
    root = _dark_tk(t("loc_title"), "700x560")
    root.minsize(640, 520)
    lang_var = tk.StringVar(value=cfg.get("lang") or detect_lang())
    path_var = tk.StringVar(value=_legacy_suggestion() or core.suggest_root(drives))
    drive_var = tk.StringVar(value=core.drive_label(path_var.get()))
    state = {"result": None}
    body = tk.Frame(root, bg=BG)
    body.pack(fill="both", expand=True, padx=18, pady=12)

    def on_drive(d):
        path_var.set(os.path.join(d, core.DEFAULT_ROOT_NAME))

    def update_free(*_a):
        try:
            free = core.disk_free_gb(path_var.get())
            if 0 <= free < 12:
                lbl_free.config(text=t("loc_warn_space", free=free, need=12), fg=WARN)
            else:
                lbl_free.config(text=t("loc_free", free=max(0.0, free)), fg=GREEN)
            drive_var.set(core.drive_label(path_var.get()))
        except Exception:
            pass

    def browse():
        d = filedialog.askdirectory(parent=root, initialdir=os.path.dirname(path_var.get()) or None)
        if d:
            d = os.path.abspath(d)
            if os.path.dirname(d) == d or (IS_WIN and len(d) <= 3):    # disk kökü seçildiyse alt klasör aç
                d = os.path.join(d, core.DEFAULT_ROOT_NAME)
            path_var.set(d)

    def on_continue():
        p = path_var.get().strip()
        if not p:
            return
        p = os.path.abspath(p)
        if not core.is_writable_dir(p):
            messagebox.showerror(t("loc_title"), t("loc_not_writable"), parent=root)
            return
        free = core.disk_free_gb(p)
        if 0 <= free < 12:
            if not messagebox.askyesno(t("loc_title"), t("loc_warn_space", free=free, need=12) + "\n\n" +
                                       t("loc_continue_anyway"), parent=root):
                return
        state["result"] = (p, lang_var.get())
        root.destroy()

    lbl_free = None

    def render():
        nonlocal lbl_free
        T.set(LANGS.get(lang_var.get(), "en"))
        root.title(t("loc_title"))
        for w in body.winfo_children():
            w.destroy()
        tk.Label(body, text="🎨 " + t("loc_title"), bg=BG, fg=BLUE, font=(UI_FONT, 14, "bold"), anchor="w").pack(fill="x")
        tk.Label(body, text=t("loc_intro"), bg=BG, fg=TXT, font=(UI_FONT, 10), wraplength=640, justify="left",
                 anchor="w").pack(fill="x", pady=(8, 10))
        lr = tk.Frame(body, bg=BG)
        lr.pack(fill="x", pady=(0, 8))
        tk.Label(lr, text=t("loc_lang"), bg=BG, fg=TXT, font=(UI_FONT, 10, "bold")).pack(side="left")
        cb = ttk.Combobox(lr, textvariable=lang_var, values=list(LANGS.keys()), state="readonly", width=16)
        cb.pack(side="left", padx=8)
        cb.bind("<<ComboboxSelected>>", lambda e: render())
        tk.Label(body, text=t("loc_drives"), bg=BG, fg=TXT, font=(UI_FONT, 10, "bold"), anchor="w").pack(fill="x")
        for d, free, total in drives:
            txt = t("loc_drive", drive=d, free=free, total=total)
            if d == best_drive:
                txt += "   " + t("loc_recommended")
            tk.Radiobutton(body, text=txt, variable=drive_var, value=core.drive_label(d), command=lambda d=d: on_drive(d),
                           bg=BG, fg=GREEN if d == best_drive else LBL, selectcolor=FRM, activebackground=BG,
                           activeforeground=TXT, font=(UI_FONT, 10), anchor="w").pack(fill="x", padx=8)
        fr = tk.Frame(body, bg=BG)
        fr.pack(fill="x", pady=(12, 2))
        tk.Label(fr, text=t("loc_folder"), bg=BG, fg=TXT, font=(UI_FONT, 10, "bold")).pack(side="left")
        tk.Entry(fr, textvariable=path_var, bg=FRM, fg=TXT, insertbackground=TXT, font=(UI_FONT, 10),
                 relief="flat").pack(side="left", fill="x", expand=True, padx=8, ipady=3)
        tk.Button(fr, text=t("loc_browse"), bg=BTN, fg=TXT, relief="flat", font=(UI_FONT, 9), command=browse,
                  cursor="hand2").pack(side="left")
        lbl_free = tk.Label(body, text="", bg=BG, fg=GREEN, font=(UI_FONT, 9, "bold"), anchor="w")
        lbl_free.pack(fill="x", pady=(2, 10))
        update_free()
        btns = tk.Frame(body, bg=BG)
        btns.pack(fill="x", side="bottom", pady=(8, 0))
        tk.Button(btns, text=t("loc_continue"), bg=ORANGE, fg="white", font=(UI_FONT, 12, "bold"), relief="flat",
                  cursor="hand2", command=on_continue).pack(side="left", fill="x", expand=True, ipady=6)
        tk.Button(btns, text=t("loc_exit"), bg=BTN, fg=TXT, font=(UI_FONT, 10), relief="flat", cursor="hand2",
                  command=root.destroy).pack(side="left", padx=(8, 0), ipady=6, ipadx=14)

    path_var.trace_add("write", update_free)
    render()
    root.mainloop()
    return state["result"]


def _wait_pid(pid, timeout=30):
    """Yeniden başlatmada eski sürecin dosyaları bırakmasını bekle."""
    if not pid:
        return
    t0 = time.time()
    if IS_WIN:
        try:
            import ctypes
            SYNCHRONIZE = 0x00100000
            h = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, int(pid))
            if h:
                ctypes.windll.kernel32.WaitForSingleObject(h, int(timeout * 1000))
                ctypes.windll.kernel32.CloseHandle(h)
        except Exception:
            time.sleep(2)
        return
    while time.time() - t0 < timeout:
        try:
            os.kill(int(pid), 0)
        except OSError:
            return
        time.sleep(0.3)


def _fix_settings_after_move(old_root, new_root):
    p = os.path.join(new_root, "settings.json")
    try:
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        od = s.get("output_dir") or ""
        if od and os.path.normcase(os.path.abspath(od)).startswith(os.path.normcase(os.path.abspath(old_root))):
            s["output_dir"] = os.path.join(new_root, os.path.relpath(od, old_root))
            with open(p, "w", encoding="utf-8") as f:
                json.dump(s, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _file_size(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return 0


def _outside(path, root):
    """path, root klasörünün DIŞINDA mı?"""
    try:
        p, r = os.path.normcase(os.path.abspath(path)), os.path.normcase(os.path.abspath(root))
        return not (p == r or p.startswith(r + os.sep))
    except Exception:
        return True


def _runtime_choice(paths):
    """Programın ortamı hangi Python'la kurulacak: 'portable' | 'system' | None (kullanıcı vazgeçti)."""
    if core.portable_python_exe(paths):
        return "portable"
    if os.path.exists(paths.venv_python) and core.venv_ok(paths) and core.engine_status(paths):
        return "system"                    # motor kurulu bir ortam var: bozulmasın, yalnızca eksik paketler kurulur
    if not IS_WIN:
        return "system"                    # programa özel Python yalnızca Windows'ta önerilir
    fmt = dict(ver=platform.python_version(), pyv=core.PORTABLE_PY, size=core.PORTABLE_SIZE_MB,
               path=core.portable_root(paths))
    problem = core.system_python_problem()
    if problem:
        return "portable" if _box("ask", t("rt_title"), t("rt_need_" + problem, **fmt)) else None
    return "portable" if _box("ask", t("rt_title"), t("rt_ask", **fmt)) else "system"


def _prep_error_text(val):
    if isinstance(val, RuntimeError) and str(val).startswith("NO_VENV"):
        return t("prep_no_venv")
    return friendly_error(val)


def _prepare_runtime(paths, choice=None, force_new=False):
    """
    Programın kendi ortamını (runtime\\venv) hazırlar. Her adım ve pip'in tüm çıktısı logs\\setup.log'a yazılır.
    Bilgisayardaki Python'la olmazsa programa özel Python (3.12) önerilir.
    """
    core.setup_log(paths, "=== %s %s — setup" % (APP_NAME, APP_VERSION))
    core.setup_log(paths, core.python_report())
    choice = choice or _runtime_choice(paths)
    if choice is None:
        core.setup_log(paths, "cancelled by the user")
        _box("info", APP_NAME, t("rt_cancelled"))
        return False
    core.setup_log(paths, "runtime: " + choice)
    state = {"used": choice}

    def prep(w):
        py = None
        new_env = force_new
        if choice == "portable":
            py = core.portable_python_exe(paths)
            if not (py and core.check_python(py, paths)[0]):
                w.log(t("rt_dl", pyv=core.PORTABLE_PY, size=core.PORTABLE_SIZE_MB))
                try:
                    py = core.install_portable_python(paths, w.log, w.progress)
                    w.log(t("rt_ok", path=py))
                    new_env = True
                except core.AbortedError:
                    raise
                except Exception as e:  # noqa
                    core.setup_log(paths, "portable python failed:\n" + traceback.format_exc())
                    if not _can_use_system():
                        raise                      # bilgisayardaki Python da uygun değil: geri dönülecek yer yok
                    w.log(t("rt_fallback", ver=platform.python_version(), err=friendly_error(e)))
                    py = None
                    state["used"] = "system"
        if new_env or not core.venv_ok(paths):
            w.log(t("prep_venv"))
            core.check_space(paths.root, 0.5)
            try:
                core.create_venv(paths, w.log, python_exe=py)
            except Exception:  # noqa
                if not py or not os.path.exists(core.uv_exe_path(paths)):
                    raise
                core.setup_log(paths, "python -m venv failed with the program's own Python — trying 'uv venv':\n" +
                               traceback.format_exc())
                core.create_venv_uv(paths, py, w.log)
        missing = core.missing_basic(paths)
        if missing:
            w.log(t("prep_basic"))
            core.install_basics(paths, missing, w.log, w.progress)
        return True

    kind, val = TaskWindow(t("prep_title"), t("prep_title"), paths).run(prep)
    if kind == "ok":
        core.setup_log(paths, "setup OK")
        return True
    core.setup_log(paths, "setup FAILED:\n" + "".join(traceback.format_exception(type(val), val, val.__traceback__)))
    msg = _prep_error_text(val)
    if state["used"] == "system" and IS_WIN and not core.portable_python_exe(paths) and \
            not isinstance(val, (core.DiskFullError, core.NetworkError, core.AbortedError)):
        if _box("ask", t("rt_title"), t("prep_fail_try_portable", err=msg, ver=platform.python_version(),
                                        pyv=core.PORTABLE_PY, size=core.PORTABLE_SIZE_MB)):
            return _prepare_runtime(paths, "portable", force_new=True)
    _box("error", APP_NAME, t("prep_fail", err=msg) + "\n\n" + t("prep_log_at", path=core.setup_log_path(paths)))
    return False


def _can_use_system():
    return core.system_python_problem() is None


def launcher_main():
    """Programı kendi ortamında başlatır; gerekiyorsa önce konum seçimi / taşıma / hazırlık yapar."""
    cfg = load_pointer()
    if cfg.get("lang") not in LANGS:
        cfg["lang"] = detect_lang()
    T.set(LANGS.get(cfg["lang"], "en"))
    if "--wait-pid" in sys.argv:
        try:
            _wait_pid(sys.argv[sys.argv.index("--wait-pid") + 1])
        except Exception:
            pass
    if sys.version_info < (3, 10):
        _box("error", APP_NAME, t("prep_old_python", ver=platform.python_version()))
        return 1

    if not cfg.get("data_root") or not core.is_writable_dir(cfg["data_root"]):
        res = first_run_dialog(cfg)
        if not res:
            return 0
        cfg["data_root"], cfg["lang"] = res
        T.set(LANGS.get(cfg["lang"], "en"))
        save_pointer(cfg)

    pm = cfg.get("pending_move")
    if isinstance(pm, dict) and pm.get("from") and pm.get("to"):
        win = TaskWindow(t("move_title"), t("move_title") + "\n" + pm["from"] + "  →  " + pm["to"])
        kind, val = win.run(lambda w: core.move_data_root(pm["from"], pm["to"], w.progress, w.log))
        if kind == "ok":
            cfg["data_root"] = pm["to"]
            _fix_settings_after_move(pm["from"], pm["to"])
            core.fix_venv_home(pm["to"], pm["from"])
        else:
            _box("error", APP_NAME, t("move_fail", err=friendly_error(val)))
        cfg.pop("pending_move", None)
        save_pointer(cfg)

    paths = core.DataPaths(cfg["data_root"])
    paths.ensure_dirs()

    if cfg.pop("pending_runtime_reset", False):
        save_pointer(cfg)
        for _i in range(15):
            shutil.rmtree(paths.venv, ignore_errors=True)
            if not os.path.exists(paths.venv):
                break
            time.sleep(1.0)
        core.remove_engine_marker(paths)

    if not cfg.get("cleanup_checked"):
        cfg["cleanup_checked"] = True
        save_pointer(cfg)
        try:
            info = core.legacy_user_site_torch()
            if info and info["size_gb"] > 0.05 and _box("ask", t("cleanup_title"),
                                                         t("cleanup_msg", drive=info["drive"], size=info["size_gb"])):
                freed = core.legacy_cleanup(info)
                _box("info", t("cleanup_title"), t("cleanup_done", size=freed))
        except Exception:
            pass

    if not _outside(sys.executable, paths.root):
        pass                                   # programa özel Python'la açıldı (yeniden başlatma) — kaydı değiştirme
    elif cfg.get("launcher_python") != sys.executable:
        cfg["launcher_python"] = sys.executable
        save_pointer(cfg)

    if not core.venv_ok(paths) or core.missing_basic(paths):
        if not _prepare_runtime(paths):
            return 1

    use_w = IS_WIN and os.path.basename(sys.executable).lower() == "pythonw.exe"
    py = paths.venv_pythonw if use_w and os.path.exists(paths.venv_pythonw) else paths.venv_python
    env = os.environ.copy()
    env.update(paths.env())
    return subprocess.call([py, os.path.abspath(__file__), "--ails-child"], env=env, cwd=APP_DIR)


# =============================================================================
# ARAYÜZ yardımcıları
# =============================================================================
class Fonts:
    def __init__(self, root):
        fam = UI_FONT
        self.n9 = tkfont.Font(root, family=fam, size=9)
        self.b9 = tkfont.Font(root, family=fam, size=9, weight="bold")
        self.n10 = tkfont.Font(root, family=fam, size=10)
        self.b10 = tkfont.Font(root, family=fam, size=10, weight="bold")
        self.b11 = tkfont.Font(root, family=fam, size=11, weight="bold")
        self.b12 = tkfont.Font(root, family=fam, size=12, weight="bold")
        self.b18 = tkfont.Font(root, family=fam, size=18, weight="bold")
        self.mono = tkfont.Font(root, family="Consolas" if IS_WIN else "DejaVu Sans Mono", size=10)

    def apply_lang(self, code):
        # Segoe UI Korece karakterleri tam göstermez → Windows'ta hazır gelen Malgun Gothic.
        if code == "ko":
            fam = "Malgun Gothic" if IS_WIN else "Noto Sans CJK KR"
        else:
            fam = UI_FONT
        for f in (self.n9, self.b9, self.n10, self.b10, self.b11, self.b12, self.b18):
            f.configure(family=fam)


def set_dark(rt, fonts):
    """SRT projesindeki set_dark ile aynı tema."""
    s = ttk.Style(rt)
    s.theme_use("clam")
    s.configure("TFrame", background=BG)
    s.configure("TLabelframe", background=BG, foreground=TXT)
    s.configure("TLabelframe.Label", background=BG, foreground=TXT, font=fonts.b10)
    s.configure("TLabel", background=BG, foreground=LBL, font=fonts.n9)
    s.configure("TRadiobutton", background=BG, foreground=LBL)
    s.configure("TCheckbutton", background=BG, foreground=LBL)
    s.configure("TButton", background=BTN, foreground=TXT, font=fonts.n9)
    s.map("TButton", background=[("active", "#555555")])
    s.configure("TCombobox", font=fonts.n9)
    rt.option_add("*TCombobox*Listbox.font", fonts.n9)
    try:
        rh = max(20, int(fonts.n9.metrics("linespace") * 1.45))
    except Exception:
        rh = 22
    s.configure("Treeview", background=FRM, fieldbackground=FRM, foreground=TXT, font=fonts.n9, rowheight=rh)
    s.configure("Treeview.Heading", background=BTN, foreground=TXT, font=fonts.b9)
    s.map("Treeview", background=[("selected", "#0d47a1")], foreground=[("selected", "white")])
    rt.configure(bg=BG)


class KeyCombo:
    """Ekranda çeviri metni gösteren, değişkende ise sabit anahtar tutan combobox."""

    def __init__(self, parent, var, items_fn, width=14, on_change=None):
        self.var, self.items_fn, self.on_change = var, items_fn, on_change
        self.keys = []
        self.cb = ttk.Combobox(parent, state="readonly", width=width)
        self.cb.bind("<<ComboboxSelected>>", self._sel)
        self._tid = var.trace_add("write", lambda *a: self.refresh())
        self.cb.bind("<Destroy>", self._on_destroy, add="+")
        self.refresh()

    def _on_destroy(self, _e=None):
        if self._tid:
            try:
                self.var.trace_remove("write", self._tid)
            except Exception:
                pass
            self._tid = None

    def refresh(self):
        if not self._tid or not self.cb.winfo_exists():
            return
        items = self.items_fn()
        self.keys = [k for k, _ in items]
        self.cb["values"] = [lbl for _, lbl in items]
        k = self.var.get()
        if k in self.keys:
            i = self.keys.index(k)
            if self.cb.current() != i or self.cb.get() != items[i][1]:
                self.cb.current(i)
        elif self.keys:
            self.cb.set("")

    def _sel(self, _e=None):
        i = self.cb.current()
        if 0 <= i < len(self.keys):
            self.var.set(self.keys[i])
            if self.on_change:
                self.on_change()


def dark_check(parent, var, fonts, fg=LBL, bg=BG, command=None):
    return tk.Checkbutton(parent, variable=var, bg=bg, fg=fg, selectcolor=FRM, activebackground=bg,
                          activeforeground=fg, font=fonts.b9, command=command, anchor="w")


def fmt_duration(sec):
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s_ = divmod(rem, 60)
    return f"{h}:{m:02d}:{s_:02d}" if h else f"{m}:{s_:02d}"


class JobFail(Exception):
    """Mesajı zaten log'a yazılmış, sessizce biten iş hatası."""


# =============================================================================
# v2 hazırlık penceresi — Stil Öğrenme / Görsel Analiz
#   Şimdilik: veri seti oluşturma, bilgisayardan görsel alma, (izinle) internetten görsel ekleme,
#   analiz ayarları ve eklenti (LoRA) klasörü. Analiz motoru v2'de gelecek.
# =============================================================================
class StyleStudioWindow:
    MIN_SIDES = ("0", "256", "384", "512", "768", "1024")
    CHECKPOINTS = ("10", "15", "30", "60")

    def __init__(self, app):
        self.app = app
        self.f = app.f
        self.paths = app.paths
        self.v2 = app.st["v2"]
        w = self.w = tk.Toplevel(app.r)
        w.title(t("v2_title"))
        w.configure(bg=BG)
        w.transient(app.r)
        self.combos = []
        v = self.v2
        self.v_ds = tk.StringVar(value=v.get("dataset", ""))
        self.v_tag = tk.StringVar(value="")
        self.v_recursive = tk.BooleanVar(value=bool(v["recursive"]))
        self.v_copy = tk.BooleanVar(value=bool(v["copy"]))
        self.v_min_side = tk.StringVar(value=str(v["min_side"]))
        self.v_offline = tk.BooleanVar(value=bool(v["offline"]))
        self.v_web = tk.BooleanVar(value=bool(v["web_import"]))
        self.v_mode = tk.StringVar(value=v["mode"])
        self.v_base = tk.StringVar(value=v["base_model"])
        self.v_duration = tk.StringVar(value=v["duration"])
        self.v_minutes = tk.StringVar(value=str(v["minutes"]))
        self.v_ckpt = tk.StringVar(value=str(v["checkpoint"]))
        self.v_sample = tk.BooleanVar(value=bool(v["sample_each"]))
        self.v_query = tk.StringVar(value=v.get("last_query", ""))
        self.v_scount = tk.StringVar(value=str(v["search_count"]))
        self.v_scat = tk.StringVar(value=v["search_cat"])
        self.v_slic = tk.StringVar(value=v["search_lic"])
        self.v_autocap = tk.BooleanVar(value=bool(v.get("auto_caption", True)))
        self.v_judge = tk.BooleanVar(value=bool(v.get("auto_judge", True)))
        self.v_learn = tk.StringVar(value=core.norm_learn(v.get("learn")))
        self.v_hide = tk.BooleanVar(value=bool(v.get("hide_faces", True)))
        self.v_vframes = tk.StringVar(value=str(v.get("video_frames", 120)))
        self._loading_ds = False
        self._build()
        self.refresh_datasets()
        self.update_state()
        for var in (self.v_offline, self.v_web, self.v_mode, self.v_duration, self.v_base, self.v_judge):
            var.trace_add("write", lambda *a: self.update_state())
        self.v_ds.trace_add("write", lambda *a: self.on_dataset_change())
        self.v_learn.trace_add("write", lambda *a: self.on_learn_change())
        w.protocol("WM_DELETE_WINDOW", self.close)
        w.update_idletasks()
        w.minsize(w.winfo_reqwidth(), w.winfo_reqheight())

    # ------------------------------------------------------------------ UI
    def _lf(self, parent, key, row):
        lf = ttk.LabelFrame(parent, text=t(key))
        lf.grid(row=row, column=0, sticky="nsew", pady=(0, 8))
        body = tk.Frame(lf, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=6)
        return body

    def _note(self, parent, fg="#888888", wrap=440):
        return tk.Label(parent, bg=BG, fg=fg, font=self.f.n9, anchor="w", justify="left", wraplength=wrap)

    def _build(self):
        w, f = self.w, self.f
        tk.Label(w, text=t("v2_title"), bg=BG, fg=ORANGE, font=f.b12).pack(pady=(10, 2))
        self._note(w, fg=LBL, wrap=1240).pack(fill="x", padx=16)
        w.winfo_children()[-1].config(text=t("v2_intro"))
        cols = tk.Frame(w, bg=BG)
        cols.pack(fill="both", expand=True, padx=12, pady=(8, 0))
        cols.grid_columnconfigure(0, weight=1)
        cols.grid_columnconfigure(1, weight=1)
        left = tk.Frame(cols, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.grid_columnconfigure(0, weight=1)
        right = tk.Frame(cols, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right.grid_columnconfigure(0, weight=1)

        # 1. veri seti
        b = self._lf(left, "v2_sec_dataset", 0)
        r1 = tk.Frame(b, bg=BG)
        r1.pack(fill="x")
        tk.Label(r1, text=t("v2_dataset"), bg=BG, fg=TXT, font=f.b9).pack(side="left")
        self.kc_ds = KeyCombo(r1, self.v_ds, self._ds_items, width=30)
        self.kc_ds.cb.pack(side="left", padx=(6, 6))
        self.combos.append(self.kc_ds)
        ttk.Button(r1, text=t("v2_new"), command=self.on_new).pack(side="left", padx=2)
        self.b_del = ttk.Button(r1, text=t("v2_delete"), command=self.on_delete)
        self.b_del.pack(side="left", padx=2)
        self.b_open = ttk.Button(r1, text="📁", width=3, command=self.on_open)
        self.b_open.pack(side="left", padx=2)
        rk = tk.Frame(b, bg=BG)
        rk.pack(fill="x", pady=(6, 0))
        tk.Label(rk, text=t("v2_learn"), bg=BG, fg=TXT, font=f.b9).pack(side="left")
        self.kc_learn = KeyCombo(rk, self.v_learn, lambda: [(k, t("learn_" + k)) for k in core.LEARN_KINDS], width=36)
        self.kc_learn.cb.pack(side="left", padx=(6, 0))
        self.combos.append(self.kc_learn)
        r2 = tk.Frame(b, bg=BG)
        r2.pack(fill="x", pady=(6, 0))
        tk.Label(r2, text=t("v2_tag"), bg=BG, fg=TXT, font=f.b9).pack(side="left")
        self.ent_tag = tk.Entry(r2, textvariable=self.v_tag, bg=FRM, fg=TXT, insertbackground=TXT, width=30,
                                font=f.n10, relief="flat")
        self.ent_tag.pack(side="left", padx=(6, 6))
        self.b_tag = ttk.Button(r2, text=t("v2_tag_save"), command=self.on_save_tag)
        self.b_tag.pack(side="left")
        self.lbl_tag_hint = self._note(b)
        self.lbl_tag_hint.config(text=t("v2_tag_hint_" + core.norm_learn(self.v_learn.get())))
        self.lbl_tag_hint.pack(fill="x", pady=(2, 0))
        self.lbl_stats = tk.Label(b, bg=BG, fg=GREEN, font=f.b9, anchor="w", justify="left")
        self.lbl_stats.pack(fill="x", pady=(4, 0))

        # 2. bilgisayardan
        b = self._lf(left, "v2_sec_local", 1)
        r = tk.Frame(b, bg=BG)
        r.pack(fill="x")
        self.b_folder = ttk.Button(r, text=t("v2_from_folder"), command=self.on_import_folder)
        self.b_folder.pack(side="left")
        self.b_files = ttk.Button(r, text=t("v2_from_files"), command=self.on_import_files)
        self.b_files.pack(side="left", padx=6)
        r = tk.Frame(b, bg=BG)
        r.pack(fill="x", pady=(4, 0))
        c = dark_check(r, self.v_recursive, f, fg=TXT)
        c.config(text=t("v2_recursive"))
        c.pack(side="left")
        c = dark_check(r, self.v_copy, f, fg=GREEN)
        c.config(text=t("v2_copy"))
        c.pack(side="left", padx=(10, 0))
        r = tk.Frame(b, bg=BG)
        r.pack(fill="x", pady=(4, 0))
        tk.Label(r, text=t("v2_min_side"), bg=BG, fg=TXT, font=f.b9).pack(side="left")
        ttk.Combobox(r, textvariable=self.v_min_side, values=self.MIN_SIDES, state="readonly", width=6).pack(
            side="left", padx=6)
        tk.Label(r, text="px", bg=BG, fg=TXT, font=f.n9).pack(side="left")
        r = tk.Frame(b, bg=BG)
        r.pack(fill="x", pady=(4, 0))
        tk.Label(r, text=t("v2_vframes"), bg=BG, fg=TXT, font=f.b9).pack(side="left")
        ttk.Combobox(r, textvariable=self.v_vframes, values=[str(x) for x in core.VIDEO_FRAME_CHOICES],
                     state="readonly", width=5).pack(side="left", padx=6)
        tk.Label(r, text=t("v2_vframes_unit"), bg=BG, fg=TXT, font=f.n9).pack(side="left")
        self.lbl_local = self._note(b)
        self.lbl_local.config(text=t("v2_local_hint"))
        self.lbl_local.pack(fill="x", pady=(4, 0))

        # 3. internet (sağ sütun): açık lisanslı arama + doğrudan bağlantılar
        b = self._lf(right, "v2_sec_web", 0)
        c = dark_check(b, self.v_offline, f, fg=GREEN)
        c.config(text=t("v2_offline"))
        c.pack(fill="x")
        self.chk_web = dark_check(b, self.v_web, f, fg=WARN)
        self.chk_web.config(text=t("v2_web"))
        self.chk_web.pack(fill="x")
        r = tk.Frame(b, bg=BG)
        r.pack(fill="x", pady=(6, 0))
        tk.Label(r, text=t("v2_search_lbl"), bg=BG, fg=TXT, font=f.b9).pack(side="left")
        self.ent_query = tk.Entry(r, textvariable=self.v_query, bg=FRM, fg=TXT, insertbackground=TXT, font=f.n10,
                                  relief="flat", disabledbackground="#161616")
        self.ent_query.pack(side="left", fill="x", expand=True, padx=(6, 6))
        self.ent_query.bind("<Return>", lambda e: self.on_search())
        self.b_search = ttk.Button(r, text=t("v2_search_btn"), command=self.on_search)
        self.b_search.pack(side="left")
        r = tk.Frame(b, bg=BG)
        r.pack(fill="x", pady=(4, 0))
        tk.Label(r, text=t("v2_search_max"), bg=BG, fg=TXT, font=f.n9).pack(side="left")
        self.cb_scount = ttk.Combobox(r, textvariable=self.v_scount, values=[str(x) for x in core.SEARCH_COUNTS],
                                      state="readonly", width=4)
        self.cb_scount.pack(side="left", padx=(4, 10))
        tk.Label(r, text=t("v2_cat"), bg=BG, fg=TXT, font=f.n9).pack(side="left")
        self.kc_scat = KeyCombo(r, self.v_scat, lambda: [(k, t("v2_cat_" + k)) for k in core.OPENVERSE_CATEGORIES],
                                width=14)
        self.kc_scat.cb.pack(side="left", padx=(4, 10))
        self.combos.append(self.kc_scat)
        tk.Label(r, text=t("v2_lic"), bg=BG, fg=TXT, font=f.n9).pack(side="left")
        self.kc_slic = KeyCombo(r, self.v_slic, lambda: [(k, t("v2_lic_" + k)) for k in core.OPENVERSE_LICENSES],
                                width=27)
        self.kc_slic.cb.pack(side="left", padx=(4, 0))
        self.combos.append(self.kc_slic)
        n = self._note(b, wrap=640)
        n.config(text=t("v2_search_note"))
        n.pack(fill="x", pady=(2, 0))
        tk.Label(b, text=t("v2_links_lbl"), bg=BG, fg=TXT, font=f.b9, anchor="w").pack(fill="x", pady=(8, 0))
        self.txt_urls = tk.Text(b, height=2, font=f.n9, bg=FRM, fg=TXT, insertbackground=TXT, wrap="none", relief="flat")
        self.txt_urls.pack(fill="x", pady=(2, 2))
        r = tk.Frame(b, bg=BG)
        r.pack(fill="x")
        self.b_dl = ttk.Button(r, text=t("v2_download"), command=self.on_download)
        self.b_dl.pack(side="left")
        self.lbl_web = self._note(r, wrap=380)
        self.lbl_web.pack(side="left", padx=8)
        self.lbl_web_rights = self._note(b, fg=WARN, wrap=640)
        self.lbl_web_rights.config(text=t("v2_web_rights"))
        self.lbl_web_rights.pack(fill="x", pady=(4, 0))

        # 4. analiz ve öğrenme
        b = self._lf(right, "v2_sec_analysis", 1)
        b.grid_columnconfigure(1, weight=1)

        def lab(row, key):
            tk.Label(b, text=t(key), bg=BG, fg=TXT, font=f.b9, anchor="w").grid(row=row, column=0, sticky="w",
                                                                              pady=3, padx=(0, 8))
        lab(0, "v2_mode")
        kc = KeyCombo(b, self.v_mode, lambda: [("light", t("v2_mode_light")), ("deep", t("v2_mode_deep")),
                                               ("self", t("v2_mode_self"))], width=34)
        kc.cb.grid(row=0, column=1, sticky="w")
        self.combos.append(kc)
        self.lbl_mode = self._note(b, wrap=560)
        self.lbl_mode.grid(row=1, column=1, sticky="w")
        lab(2, "v2_captions")
        cf = tk.Frame(b, bg=BG)
        cf.grid(row=2, column=1, sticky="w")
        c = dark_check(cf, self.v_autocap, f, fg=GREEN)
        c.config(text=t("v2_autocap"))
        c.pack(side="left")
        self.b_caption = ttk.Button(cf, text=t("v2_caption_now"), command=self.on_caption)
        self.b_caption.pack(side="left", padx=(8, 0))
        self.lbl_caps = self._note(b, wrap=560)
        self.lbl_caps.grid(row=3, column=1, sticky="w")
        lab(4, "v2_rewards")
        rf = tk.Frame(b, bg=BG)
        rf.grid(row=4, column=1, sticky="w")
        self.lbl_rewards = tk.Label(rf, bg=BG, fg=BLUE, font=f.b9)
        self.lbl_rewards.pack(side="left")
        self.chk_judge = dark_check(rf, self.v_judge, f, fg=WARN)
        self.chk_judge.config(text=t("v2_auto_judge"))
        self.chk_judge.pack(side="left", padx=(12, 0))
        lab(5, "v2_duration")
        df = tk.Frame(b, bg=BG)
        df.grid(row=5, column=1, sticky="w")
        kc = KeyCombo(df, self.v_duration, lambda: [("until_stop", t("v2_until_stop")), ("minutes", t("v2_minutes"))],
                      width=20)
        kc.cb.pack(side="left")
        self.combos.append(kc)
        vcmd = (self.w.register(lambda s: s == "" or (s.isdigit() and int(s) <= 100000)), "%P")
        self.spin_min = tk.Spinbox(df, from_=5, to=100000, increment=5, textvariable=self.v_minutes, width=6,
                                   validate="key", validatecommand=vcmd, bg=FRM, fg=TXT, insertbackground=TXT,
                                   buttonbackground=BTN, font=f.n10, relief="flat")
        self.spin_min.pack(side="left", padx=6)
        tk.Label(df, text=t("v2_min_unit"), bg=BG, fg=TXT, font=f.n9).pack(side="left")
        tk.Label(df, text=t("v2_checkpoint"), bg=BG, fg=TXT, font=f.b9).pack(side="left", padx=(14, 4))
        ttk.Combobox(df, textvariable=self.v_ckpt, values=self.CHECKPOINTS, state="readonly", width=4).pack(side="left")
        tk.Label(df, text=t("v2_min_unit"), bg=BG, fg=TXT, font=f.n9).pack(side="left", padx=4)
        sf = tk.Frame(b, bg=BG)
        sf.grid(row=6, column=0, columnspan=2, sticky="w", pady=(4, 0))
        c = dark_check(sf, self.v_sample, f, fg=TXT)
        c.config(text=t("v2_sample_each"))
        c.pack(side="left")
        self.chk_hide = dark_check(sf, self.v_hide, f, fg=GREEN)
        self.chk_hide.config(text=t("v2_hide_faces"))
        self.chk_hide.pack(side="left", padx=(12, 0))
        n = self._note(b, wrap=640)
        n.config(text=t("v2_analysis_hint"))
        n.grid(row=7, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # eklentiler (LoRA) — sol sütunun altında
        b = self._lf(left, "v2_sec_addons", 2)
        r = tk.Frame(b, bg=BG)
        r.pack(fill="x")
        self.lbl_addons = tk.Label(r, bg=BG, fg=TXT, font=f.n9, anchor="w", justify="left", wraplength=360)
        self.lbl_addons.pack(side="left", fill="x", expand=True)
        ttk.Button(r, text=t("v2_addons_open"), command=self.on_open_addons).pack(side="right")
        n = self._note(b, fg=WARN)
        n.config(text=t("v2_addons_note"))
        n.pack(fill="x", pady=(4, 0))

        # ilerleme + başlat (sol sütunun altında; sağ sütun internet + analiz ayarlarıyla dolu)
        pf = tk.Frame(left, bg=BG)
        pf.grid(row=3, column=0, sticky="ew", pady=(0, 4))
        self.pg = ttk.Progressbar(pf, mode="determinate", maximum=100)
        self.pg.pack(fill="x")
        self.lbl_prog = tk.Label(pf, bg=BG, fg=BLUE, font=f.b9, anchor="w")
        self.lbl_prog.pack(fill="x")
        self.b_start = tk.Button(left, text=t("v2_start"), bg=ORANGE, fg="white", font=f.b11, relief="flat",
                                 cursor="hand2", activebackground="#ff5722", command=self.on_start)
        self.b_start.grid(row=4, column=0, sticky="ew", ipady=6)

        btns = tk.Frame(w, bg=BG)
        btns.pack(fill="x", padx=16, pady=10)
        self.b_stop = ttk.Button(btns, text=t("v2_stop"), command=self.app.abort.set)
        self.b_stop.pack(side="left")
        ttk.Button(btns, text=t("adv_close"), command=self.close).pack(side="right")

    # ------------------------------------------------------------------ yardımcılar
    def alive(self):
        try:
            return bool(self.w.winfo_exists())
        except Exception:
            return False

    def _ds_items(self):
        items = [(d["slug"], f"{d['name']}  ({d['count']})") for d in core.list_datasets(self.paths)]
        return items or [("", t("v2_no_dataset"))]

    def refresh_datasets(self, select=None):
        if select is not None:
            self.v_ds.set(select)
        keys = [k for k, _ in self._ds_items()]
        if self.v_ds.get() not in keys:
            self.v_ds.set(keys[0] if keys else "")
        for kc in self.combos:
            kc.refresh()
        self.on_dataset_change()

    def on_dataset_change(self):
        if not self.alive():
            return
        slug = self.v_ds.get()
        data = core.load_dataset(self.paths, slug) if slug else None
        if data is None:
            self.v_tag.set("")
            self.lbl_stats.config(text=t("v2_stats_none"))
        else:
            self.v_tag.set(data.get("tag", ""))
            if data["learn"] != self.v_learn.get():
                self._loading_ds = True            # veri setinin türünü göster (kaydetme döngüsüne girmeden)
                try:
                    self.v_learn.set(data["learn"])
                    self.kc_learn.refresh()
                finally:
                    self._loading_ds = False
            st = core.dataset_stats(self.paths, slug, data)
            self.lbl_stats.config(text=t("v2_stats", n=st["count"], mb=st["bytes"] / core.MB, local=st["local"],
                                         web=st["web"], video=st["video"], date=data.get("created", "")))
        self.lbl_tag_hint.config(text=t("v2_tag_hint_" + core.norm_learn(self.v_learn.get())))
        self.update_state()

    def on_learn_change(self):
        """'Ne öğretiliyor?' değişti: seçili veri setine kaydedilir; açıklamalar ve etiket ipucu güncellenir."""
        if not self.alive():
            return
        learn = core.norm_learn(self.v_learn.get())
        self.v2["learn"] = learn
        self.lbl_tag_hint.config(text=t("v2_tag_hint_" + learn))
        if not self._loading_ds:
            slug = self.v_ds.get()
            data = core.load_dataset(self.paths, slug) if slug else None
            if data is not None and data.get("learn") != learn:
                data["learn"] = learn
                core.save_dataset(self.paths, slug, data)
                self.app.log(t("v2_log_learn", name=data.get("name", slug), kind=t("learn_" + learn)))
        self.update_state()

    def update_state(self):
        if not self.alive():
            return
        has_ds = bool(self.v_ds.get())
        busy = self.app.running()
        for btn in (self.b_del, self.b_open, self.b_tag):
            btn.state(["!disabled"] if has_ds else ["disabled"])
        for btn in (self.b_folder, self.b_files):
            btn.state(["disabled"] if busy else ["!disabled"])
        offline = self.v_offline.get()
        self.chk_web.config(state="disabled" if offline else "normal")
        web_ok = (not offline) and self.v_web.get()
        self.txt_urls.config(state="normal" if web_ok else "disabled", bg=FRM if web_ok else "#161616")
        self.b_dl.state(["!disabled"] if (web_ok and not busy) else ["disabled"])
        self.b_search.state(["!disabled"] if (web_ok and not busy) else ["disabled"])
        self.ent_query.config(state="normal" if web_ok else "disabled")
        for cb in (self.cb_scount, self.kc_scat.cb, self.kc_slic.cb):
            cb.config(state="readonly" if web_ok else "disabled")
        self.lbl_web.config(text=t("v2_web_limits", n=core.WEB_MAX_PER_BATCH, mb=core.WEB_MAX_MB) if web_ok
                            else (t("v2_web_off_offline") if offline else t("v2_web_off")))
        self.spin_min.config(state="normal" if self.v_duration.get() == "minutes" else "disabled")
        hw = self.app.hw
        mode = self.v_mode.get()
        learn = core.norm_learn(self.v_learn.get())
        what = t("learn_what_" + learn)
        gpu = core.train_device(hw) != "cpu"
        if mode == "deep":
            self.lbl_mode.config(text=t("v2_mode_deep_info", what=what, extra=t("v2_deep_note_" + learn)) + "  " +
                                 (t("fit_ok") if gpu else t("fit_slow")), fg=GREEN if gpu else WARN)
        elif mode == "self":
            self.lbl_mode.config(text=t("v2_mode_self_info") + "  " + (t("fit_ok") if gpu else t("fit_slow")),
                                 fg=GREEN if gpu else WARN)
        else:
            self.lbl_mode.config(text=t("v2_mode_light_info", what=what) + "  " + t("fit_ok"), fg=GREEN)
        self.kc_learn.cb.config(state="disabled" if busy else "readonly")
        self.chk_hide.config(state="normal" if (learn == "place" and mode == "deep" and not busy) else "disabled")
        self.chk_judge.config(state="normal" if mode == "self" else "disabled")
        self.b_caption.state(["!disabled"] if (has_ds and not busy) else ["disabled"])
        self.b_start.config(state="normal" if not busy else "disabled")
        self._update_counts()
        adds = core.list_addons(self.paths)
        self.lbl_addons.config(text=t("v2_addons", path=core.addons_dir(self.paths), n=len(adds)))
        self.b_stop.state(["!disabled"] if busy else ["disabled"])

    def set_progress(self, done, total, text=""):
        if not self.alive():
            return
        self.pg["value"] = (done * 100.0 / total) if total else 0
        self.lbl_prog.config(text=text)

    def collect(self):
        def num(var, default, lo, hi):
            try:
                return max(lo, min(hi, int(str(var.get()).strip())))
            except (TypeError, ValueError):
                return default
        self.v2.update({
            "offline": self.v_offline.get(), "web_import": self.v_web.get(), "dataset": self.v_ds.get(),
            "recursive": self.v_recursive.get(), "copy": self.v_copy.get(),
            "min_side": num(self.v_min_side, 256, 0, 4096), "mode": self.v_mode.get(), "base_model": self.v_base.get(),
            "duration": self.v_duration.get(), "minutes": num(self.v_minutes, 60, 5, 100000),
            "checkpoint": num(self.v_ckpt, 15, 1, 600), "sample_each": self.v_sample.get(),
            "last_query": self.v_query.get().strip()[:200], "search_count": num(self.v_scount, 40, 1, 100),
            "search_cat": self.v_scat.get(), "search_lic": self.v_slic.get(),
            "auto_caption": self.v_autocap.get(), "auto_judge": self.v_judge.get(),
            "learn": core.norm_learn(self.v_learn.get()), "video_frames": num(self.v_vframes, 120, 1, 400),
            "hide_faces": bool(self.v_hide.get()),
        })
        if self.v2["video_frames"] not in core.VIDEO_FRAME_CHOICES:
            self.v2["video_frames"] = 120
        if self.v2["search_count"] not in core.SEARCH_COUNTS:
            self.v2["search_count"] = 40
        self.app.st["v2"] = self.v2
        self.app.save()

    def close(self):
        try:
            self.collect()
        finally:
            self.app.style_win = None
            for kc in self.combos:
                kc._on_destroy()
            self.w.destroy()

    def _need_dataset(self, suggested=""):
        """Seçili veri seti yoksa yeni bir tane oluşturmayı önerir."""
        slug = self.v_ds.get()
        if slug and core.load_dataset(self.paths, slug) is not None:
            return slug
        name = simpledialog.askstring(t("v2_new_title"), t("v2_new_prompt"), initialvalue=suggested, parent=self.w)
        if not name or not name.strip():
            return None
        slug = core.create_dataset(self.paths, name.strip(), learn=core.norm_learn(self.v_learn.get()))
        self.refresh_datasets(select=slug)
        return slug

    # ------------------------------------------------------------------ eylemler
    def on_new(self):
        name = simpledialog.askstring(t("v2_new_title"), t("v2_new_prompt"), parent=self.w)
        if name and name.strip():
            slug = core.create_dataset(self.paths, name.strip(), learn=core.norm_learn(self.v_learn.get()))
            self.refresh_datasets(select=slug)
            self.app.log(t("v2_log_created", name=name.strip(), path=os.path.join(core.datasets_root(self.paths), slug)))

    def on_delete(self):
        slug = self.v_ds.get()
        data = core.load_dataset(self.paths, slug) if slug else None
        if data is None or self.app.running():
            return
        if not messagebox.askyesno(APP_NAME, t("v2_delete_q", name=data.get("name", slug), n=len(data["items"])),
                                   parent=self.w):
            return
        core.delete_dataset(self.paths, slug)
        self.app.log(t("log_deleted", name=data.get("name", slug)))
        self.refresh_datasets(select="")

    def on_open(self):
        slug = self.v_ds.get()
        if slug:
            self.app.open_path(os.path.join(core.datasets_root(self.paths), slug))

    def on_open_addons(self):
        d = core.addons_dir(self.paths)
        os.makedirs(d, exist_ok=True)
        self.app.open_path(d)
        self.update_state()

    def on_save_tag(self):
        slug = self.v_ds.get()
        data = core.load_dataset(self.paths, slug) if slug else None
        if data is None:
            return
        data["tag"] = self.v_tag.get().strip()
        core.save_dataset(self.paths, slug, data)
        self.app.log(t("v2_log_tag", tag=data["tag"] or "—", name=data.get("name", slug)))
        self.refresh_datasets()

    def on_import_folder(self):
        if self.app.running():
            messagebox.showinfo(APP_NAME, t("busy"), parent=self.w)
            return
        d = filedialog.askdirectory(parent=self.w, title=t("v2_from_folder"),
                                    initialdir=self.v2.get("last_folder") or os.path.expanduser("~"))
        if not d:
            return
        self.v2["last_folder"] = d
        slug = self._need_dataset(os.path.basename(os.path.normpath(d)))
        if slug:
            self.collect()
            self.app.start_job("import", self.job_import, slug, d, None)
            self.update_state()

    def on_import_files(self):
        if self.app.running():
            messagebox.showinfo(APP_NAME, t("busy"), parent=self.w)
            return
        img_pat = " ".join("*" + e for e in core.IMAGE_EXTS)
        vid_pat = " ".join("*" + e for e in core.VIDEO_EXTS)
        files = filedialog.askopenfilenames(parent=self.w, title=t("v2_from_files"),
                                            initialdir=self.v2.get("last_folder") or os.path.expanduser("~"),
                                            filetypes=[(t("v2_media"), img_pat + " " + vid_pat),
                                                       (t("v2_images"), img_pat), (t("v2_videos"), vid_pat)])
        if not files:
            return
        self.v2["last_folder"] = os.path.dirname(files[0])
        slug = self._need_dataset()
        if slug:
            self.collect()
            self.app.start_job("import", self.job_import, slug, None, list(files))
            self.update_state()

    def on_download(self):
        if self.v_offline.get() or not self.v_web.get():
            return
        if self.app.running():
            messagebox.showinfo(APP_NAME, t("busy"), parent=self.w)
            return
        urls = core.clean_url_list(self.txt_urls.get("1.0", "end"))
        if not urls:
            messagebox.showinfo(APP_NAME, t("v2_no_urls"), parent=self.w)
            return
        if not self._web_rights_ok():
            return
        slug = self._need_dataset()
        if not slug:
            return
        if not messagebox.askyesno(APP_NAME, t("v2_download_q", n=len(urls), sec=len(urls) * core.WEB_MIN_DELAY / 60.0,
                                               mb=core.WEB_MAX_MB), parent=self.w):
            return
        self.collect()
        self.app.start_job("import", self.job_download, slug, urls)
        self.update_state()

    def _web_rights_ok(self):
        if self.v2.get("web_ack"):
            return True
        if not messagebox.askyesno(t("v2_web_rights_title"), t("v2_web_rights_q"), parent=self.w):
            return False
        self.v2["web_ack"] = True
        return True

    def on_search(self):
        if self.v_offline.get() or not self.v_web.get():
            return
        if self.app.running():
            messagebox.showinfo(APP_NAME, t("busy"), parent=self.w)
            return
        text = self.v_query.get().strip()
        if not text:
            messagebox.showinfo(APP_NAME, t("v2_no_query"), parent=self.w)
            return
        if not self._web_rights_ok():
            return
        slug = self._need_dataset(text)
        if not slug:
            return
        try:
            n = int(self.v_scount.get())
        except ValueError:
            n = 40
        if not messagebox.askyesno(APP_NAME, t("v2_search_q", q=text, n=n), parent=self.w):
            return
        self.collect()
        llm = (self.app.v_llm_host.get(), self.app.v_llm.get())      # Tk değişkenleri arayüz thread'inde okunur
        self.app.start_job("import", self.job_search, slug, text, n, self.v_scat.get(), self.v_slic.get(), llm)
        self.update_state()

    def _translate_query(self, text, llm):
        """Kurulu ve çalışan LLM varsa arama metnini İngilizce anahtar kelimelere çevirir (indirme/kurulum YOK)."""
        host, name = llm
        app = self.app
        try:
            c = app.client(host)
            if not c.alive(timeout=1.0) or not core.model_in_list(name, c.installed_models()):
                app.log(t("v2_log_no_llm"))
                return text
            if c.is_local:
                app.engine.park()
            system, user, schema = core.search_query_request(text)
            q = core.sanitize_search_query(c.chat_json(name, system, user, schema, timeout=(10, 180)), text)
            if c.is_local:
                c.unload_all()
            if q.lower() != text.lower():
                app.log(t("v2_log_search_tr", q=q))
            return q
        except core.AbortedError:
            raise
        except Exception as e:
            app._write_file_log("query translation failed: %r" % (e,))
            app.log(t("v2_log_no_llm"))
            return text

    def job_search(self, slug, text, n, cat, lic, llm):
        if self.v2.get("offline", True):
            self.app.log(t("v2_web_off_offline"))
            raise JobFail()
        self.app.status("v2_status_search")
        query = self._translate_query(text, llm)
        self.app.log(t("v2_log_search", q=query, lic=t("v2_lic_" + lic), cat=t("v2_cat_" + cat)))
        try:
            results, info = core.openverse_search(query, max_results=n,
                                                  licenses=core.OPENVERSE_LICENSES.get(lic, "cc0,pdm,by"),
                                                  category=cat, min_side=self.v2.get("min_side", 256),
                                                  abort_event=self.app.abort)
        except (core.AbortedError, JobFail):
            raise
        except Exception as e:
            self.app._write_file_log(traceback.format_exc())
            self.app.log(t("v2_log_search_fail", err=friendly_error(e)))
            raise JobFail()
        if info.get("rate_limited"):
            self.app.log(t("v2_log_rate"))
        elif info.get("available") is not None:
            self.app.log(t("v2_log_rate_left", n=info["available"]))
        if not results:
            if not info.get("rate_limited"):
                self.app.log(t("v2_log_search_none"))
            self._finish(None)
            self.app.status("status_done")
            return
        self.app.log(t("v2_log_search_found", n=len(results), total=info.get("total", 0)))
        self.app.status("v2_status_web")
        try:
            st = core.download_images(self.paths, slug, results, offline=self.v2.get("offline", True),
                                      min_side=self.v2.get("min_side", 256),
                                      progress_cb=lambda a, b: self._prog(a, b, t("v2_prog_web", a=a, b=b)),
                                      abort_event=self.app.abort, log_cb=self.app.log)
        except core.OfflineError:
            self.app.log(t("v2_web_off_offline"))
            raise JobFail()
        finally:
            self._finish(None)
        msg = t("v2_log_web_done", added=st["added"], dup=st["dup"], small=st["small"], bad=st["bad"],
                failed=st["failed"], mb=st["bytes"] / core.MB)
        self.app.log(msg)
        if st["stopped"]:
            self.app.log(t("v2_log_web_stopped"))
        self.app.post(self.set_progress, 1, 1, msg)
        self.app.status("status_done")

    def _update_counts(self):
        slug = self.v_ds.get()
        data = core.load_dataset(self.paths, slug) if slug else None
        if data:
            n = len(data["items"])
            k = sum(1 for it in data["items"] if it.get("caption"))
            self.lbl_caps.config(text=t("v2_caps_count", k=k, n=n), fg=GREEN if (n and k == n) else "#888888")
        else:
            self.lbl_caps.config(text="", fg="#888888")
        _p, _s, st = core.feedback_training_data(self.paths)
        self.lbl_rewards.config(text=t("v2_rewards_count", up=st["liked"], down=st["disliked"], pairs=st["pairs"]))

    def _need_captioner(self, slug):
        """Eksik açıklama varsa ve otomatik okuma açıksa Florence-2 gerekir."""
        if not self.v_autocap.get():
            return False
        data = core.load_dataset(self.paths, slug)
        return bool(data) and any(not it.get("caption") for it in data["items"])

    def on_caption(self):
        if self.app.running():
            messagebox.showinfo(APP_NAME, t("busy"), parent=self.w)
            return
        slug = self.v_ds.get()
        data = core.load_dataset(self.paths, slug) if slug else None
        if not data or not data["items"]:
            messagebox.showinfo(APP_NAME, t("v2_start_empty"), parent=self.w)
            return
        if not self.app.captioner_ok(parent=self.w):
            return
        plan = self.app._plan(engine=True, captioner=True)
        if plan is None:
            return
        self.collect()
        self.app.start_job("analysis", self.job_analysis, "caption", slug, plan, {})
        self.update_state()

    def on_start(self):
        if self.app.running():
            messagebox.showinfo(APP_NAME, t("busy"), parent=self.w)
            return
        self.collect()
        mode = self.v2["mode"]
        slug = self.v_ds.get()
        data = core.load_dataset(self.paths, slug) if slug else None
        if mode in ("light", "deep") and (not data or not data["items"]):
            messagebox.showinfo(APP_NAME, t("v2_start_empty"), parent=self.w)
            return
        if mode == "self":
            _p, _s, st = core.feedback_training_data(self.paths)
            if not st["pairs"] and not st["liked"] and not self.v_judge.get():
                messagebox.showinfo(APP_NAME, t("v2_self_empty"), parent=self.w)
                return
        learn = (data or {}).get("learn")
        # Yüz modunun derin eğitimi açıklama kullanmaz (sabit metin + yüz kesiti): Florence-2 gerekmez.
        need_cap = (mode == "light" or (mode == "deep" and learn != "face")) and self._need_captioner(slug)
        if need_cap and not self.app.captioner_ok(parent=self.w):
            need_cap = False
        train = mode in ("deep", "self")
        if mode == "deep" and learn == "face" and not data.get("face_ok"):
            # Yüz modu bir kişinin görünüşünü öğrenir: veri seti başına bir kez izin sorulur.
            if not messagebox.askyesno(APP_NAME, t("v2_face_consent_q"), parent=self.w):
                return
            data["face_ok"] = True
            core.save_dataset(self.paths, slug, data)
        # Mekân (yüzleri gizlemek) ve Yüz (yüzü bulmak) için yüz algılayıcı (Yüz ve El Düzeltici ile aynı, ~50 MB)
        hide = mode == "deep" and learn == "place" and self.v2.get("hide_faces", True)
        need_det = hide or (mode == "deep" and learn == "face")
        plan = self.app._plan(engine=True, model=core.IMAGE_MODEL_BY_KEY["sd15"] if train else None,
                              peft=train, captioner=need_cap, detailer=need_det)
        if plan is None:
            return
        opts = {"minutes": self.v2["minutes"] if self.v2["duration"] == "minutes" else None,
                "checkpoint": self.v2["checkpoint"], "sample": self.v2["sample_each"], "judge": self.v2["auto_judge"],
                "caption": need_cap, "tag": (data or {}).get("tag") or (data or {}).get("name") or slug,
                "llm": (self.app.v_llm_host.get(), self.app.v_llm.get()), "outdir": self.app.v_outdir.get(),
                "hide_faces": hide, "need_detector": need_det}
        self.app.start_job("analysis", self.job_analysis, mode, slug, plan, opts)
        self.update_state()

    # ------------------------------------------------------------------ analiz / eğitim işleri (arka plan)
    def job_analysis(self, mode, slug, plan, opts):
        app = self.app
        m15 = core.IMAGE_MODEL_BY_KEY["sd15"]
        if not core.engine_status(app.paths):
            if not plan.get("engine"):
                raise JobFail()
            app._install_engine()
        if mode in ("deep", "self"):
            if not core.is_image_model_installed(app.paths.models, m15):
                if not plan.get("model"):
                    raise JobFail()
                app._download_image_model(m15)
            if not app._ensure_peft(plan):
                raise JobFail()
        want_cap = mode == "caption" or opts.get("caption")
        if want_cap and not core.is_captioner_installed(app.paths.models):
            if not plan.get("captioner"):
                if mode == "caption":
                    raise JobFail()
                want_cap = False
            else:
                app._ensure_captioner()
        app.free_vram_for_training()
        if want_cap:
            self._caption(slug)
        if mode == "caption":
            app.status("status_done")
            return
        if mode == "light":
            self._style_card(slug, opts)
        elif mode == "deep":
            hide = bool(opts.get("hide_faces"))
            if opts.get("need_detector") and not core.is_detailer_installed(app.paths):
                if not plan.get("detailer"):
                    app.log(t("tr_log_hide_off"))
                    raise JobFail()
                app.status("status_install")
                core.install_detector(app.paths, app.log, t, app.bytes_progress, app.abort)
            self._train(core.run_style_training, slug=slug, tag=opts["tag"], minutes=opts["minutes"],
                        checkpoint_min=opts["checkpoint"], sample_each=opts["sample"], hide_faces=hide)
        else:
            pool = core.collect_prompt_pool(app.paths, opts["outdir"])
            self._train(core.run_preference_training, minutes=opts["minutes"], checkpoint_min=opts["checkpoint"],
                        sample_each=opts["sample"], auto_judge=opts["judge"], prompt_pool=pool)
        app.post(app.update_lora_label)
        app.status("status_done")

    def _caption(self, slug):
        app = self.app
        app.status("v2_status_caption")
        device = app.hw.device if (app.hw.cuda_usable and app.hw.vram_gb >= 2) else "cpu"
        app.log(t("v2_log_caption_load", device="GPU" if device != "cpu" else "CPU"))
        try:
            cap = core.Captioner(core.captioner_dir(app.paths.models), device)
        except Exception as e:
            if device == "cpu" or not core.is_oom_error(e):
                raise
            core.free_gpu_memory()
            cap = core.Captioner(core.captioner_dir(app.paths.models), "cpu")
        try:
            st = core.caption_dataset(app.paths, slug, cap.caption, abort_event=app.abort,
                                      progress_cb=lambda a, b: self._prog(a, b, t("v2_prog_caption", a=a, b=b)))
        finally:
            cap.close()
            self._finish(None)
        app.log(t("v2_log_caption_done", done=st["done"], failed=st["failed"]))

    def _style_card(self, slug, opts):
        """Hafif mod: görsellerin ortak özellikleri (öğretilen türe göre: tarz / mekân / konu / anatomi) kelime kartına."""
        app = self.app
        data = core.load_dataset(app.paths, slug) or {"items": [], "learn": "style"}
        learn = core.norm_learn(data.get("learn"))
        caps = [it.get("caption", "") for it in data["items"] if it.get("caption")]
        if not caps:
            app.log(t("v2_log_card_nocap"))
            raise JobFail()
        app.status("v2_status_card")
        card = None
        host, name = opts["llm"]
        try:
            c = app.client(host)
            if c.alive(timeout=1.0) and core.model_in_list(name, c.installed_models()):
                if c.is_local:
                    app.engine.park()
                system, user, schema = core.card_request(caps, opts["tag"], learn)
                d = c.chat_json(name, system, user, schema, timeout=(10, 600))
                card = {"style": str(d.get("style", "")), "avoid": str(d.get("avoid", "")),
                        "summary": str(d.get("summary", ""))}
                if c.is_local:
                    c.unload_all()
        except core.AbortedError:
            raise
        except Exception as e:
            app._write_file_log("card LLM failed: %r" % (e,))
        if not card or not card["style"].strip():
            app.log(t("v2_log_card_heuristic"))
            card = core.heuristic_card(caps, learn)
        if not card["style"]:
            app.log(t("v2_log_card_empty"))
            raise JobFail()
        _path, body = core.save_card(app.paths, opts["tag"], opts["tag"], card, slug, len(caps), learn)
        app.log(t("v2_log_card_done", icon=core.learn_icon(learn), kind=t("learn_short_" + learn), name=body["name"],
                  style=body["style"][:220]))
        if body.get("avoid"):
            app.log(t("v2_log_card_avoid", avoid=body["avoid"][:200]))

    def _train(self, fn, **kw):
        app = self.app
        app.status("v2_status_train")

        def prog(frac, steps, elapsed):
            txt = t("tr_counter", steps=steps, time=fmt_duration(elapsed))
            if frac is None:
                app.post(app.set_indeterminate, txt)
            else:
                app.progress(frac * 100.0, txt)
            app.post(self.set_progress, frac or 0.0, 1, txt)

        def on_sample(path, info):
            meta = {"model": "LoRA: %s" % info.get("name", ""), "stage": "sample", "steps": info.get("steps"),
                    "width": info.get("res"), "height": info.get("res"), "prompt": info.get("trigger", "")}
            app.post(app.add_to_gallery, path, meta, True)
            app.log(t("tr_log_sample", path=path))

        info = None
        wd = core.RamWatchdog(app.abort, app.hw.ram_total_gb)
        for res in (None, 384):
            try:
                with wd:
                    info = fn(app.paths, app.hw, log_cb=app.log, msg=t, progress_cb=prog, abort_event=app.abort,
                              on_sample=on_sample, resolution=res, **kw)
                break
            except core.NothingToTrainError:
                app.log(t("tr_log_nothing"))
                raise JobFail()
            except core.AbortedError:
                if wd.triggered:
                    app.log(t("log_ram_guard", free=wd.free_at_trigger))
                    raise core.AbortedError("ram")
                raise
            except Exception as e:
                if res is None and core.is_oom_error(e):
                    core.free_gpu_memory()
                    app.log(t("tr_log_oom"))
                    continue
                raise
        if wd.triggered:
            app.log(t("log_ram_guard", free=wd.free_at_trigger))
        app.log(t("tr_log_done", name=info.get("name", ""), steps=info.get("steps", 0), path=info.get("path", "")))
        if info.get("stop_reason") == "stop":
            app.log(t("tr_log_stopped_saved"))

    # ------------------------------------------------------------------ işler (arka plan)
    def _prog(self, done, total, text):
        self.app.progress(done * 100.0 / max(1, total), f"{done}/{total}")
        self.app.post(self.set_progress, done, total, text)

    def _finish(self, st):
        self.app.post(self.refresh_datasets)
        self.app.post(self.update_state)

    def job_import(self, slug, folder, files):
        app = self.app
        app.status("v2_status_scan")
        if folder:
            app.log(t("v2_log_scan", path=folder))
            images, videos = core.scan_media_files(folder, self.v2.get("recursive", True))
        else:
            images, videos = core.split_media_files(files)
        app.log(t("v2_log_found_media", n=len(images), v=len(videos)) if videos else t("v2_log_found", n=len(images)))
        use_videos = False
        n_max = int(self.v2.get("video_frames", 120))
        min_side = self.v2.get("min_side", 256)
        if videos:
            size_gb = sum(_file_size(p) for p in videos) / core.GB
            if app.call_ui(self._ask_videos, len(videos), size_gb, n_max):
                use_videos = self._ensure_cv2()
                if not use_videos:
                    app.log(t("v2_log_videos_nocv2", n=len(videos)))
            else:
                app.log(t("v2_log_videos_skipped", n=len(videos)))
        if not images and not use_videos:
            self._finish(None)
            app.status("status_done")
            return
        added = 0
        try:
            if images:
                app.status("v2_status_import")
                st = core.import_images(self.paths, slug, images, copy=self.v2.get("copy", True), min_side=min_side,
                                        progress_cb=lambda a, b: self._prog(a, b, t("v2_prog_import", a=a, b=b)),
                                        abort_event=app.abort)
                added += st["added"]
                app.log(t("v2_log_import_done", added=st["added"], dup=st["dup"], small=st["small"], bad=st["bad"],
                          mb=st["bytes"] / core.MB))
            if use_videos:
                app.status("v2_status_video")
                for i, vp in enumerate(videos):
                    name = os.path.basename(vp)
                    app.log(t("v2_log_video_start", name=name, i=i + 1, n=len(videos), k=n_max))
                    try:
                        vs = core.extract_video_frames(
                            self.paths, slug, vp, max_frames=n_max, min_side=min_side, abort_event=app.abort,
                            progress_cb=lambda a, b, i=i: self._prog(a, b, t("v2_prog_video", i=i + 1, n=len(videos),
                                                                              p=int(a * 100 / max(1, b)))))
                    except core.AbortedError:
                        raise
                    except Exception as e:  # noqa
                        app._write_file_log(traceback.format_exc())
                        app.log(t("v2_log_video_fail", name=name, err=friendly_error(e)))
                        continue
                    if not vs["added"] and vs["small"]:
                        app.log(t("v2_log_video_small", name=name, px=min_side))
                        continue
                    if not vs["added"] and vs["bad"]:
                        app.log(t("v2_log_video_bad", name=name))
                        continue
                    added += vs["added"]
                    app.log(t("v2_log_video_done", name=name, added=vs["added"], similar=vs["similar"] + vs["dup"],
                              unusable=vs["unusable"], mb=vs["bytes"] / core.MB, min=vs["seconds"] / 60.0))
        finally:
            self._finish(None)
        app.post(self.set_progress, 1, 1, t("v2_log_import_total", added=added))
        if added:
            data = core.load_dataset(self.paths, slug)
            app.log(t("v2_log_import_total", added=added) + "  " +
                    t("v2_log_named", base=core.dataset_file_base(data, slug)))
            if self.v2.get("auto_caption", True):
                app.after_job = lambda: self._auto_caption(slug)
        app.status("status_done")

    def _ask_videos(self, n, size_gb, k):
        parent = self.w if self.alive() else self.app.r
        return messagebox.askyesno(APP_NAME, t("v2_videos_q", n=n, gb=size_gb, k=k), parent=parent)

    def _ensure_cv2(self):
        """Videodan kare çıkarmak için OpenCV (programın ortamında). Yoksa onayla kurulur. (İş thread'inde çalışır.)"""
        app = self.app
        if core.cv2_installed(self.paths):
            try:
                importlib.invalidate_caches()
                importlib.import_module("cv2")
                return True
            except Exception:
                pass
        plan = app.call_ui(lambda: app._plan(cv2=True))
        if not plan or not plan.get("cv2"):
            return bool(plan is not None and core.cv2_installed(self.paths))
        app.status("status_install")
        core.install_cv2(self.paths, app.log, t, app.bytes_progress, app.abort)
        importlib.invalidate_caches()
        try:
            importlib.import_module("cv2")
            return True
        except Exception as e:  # noqa
            app._write_file_log(traceback.format_exc())
            app.log(t("v2_log_video_fail", name="OpenCV", err=friendly_error(e)))
            return False

    def _auto_caption(self, slug):
        """İçe aktarmadan sonra: açıklaması olmayan yeni görseller (video kareleri dahil) hemen okunur."""
        if not self.alive() or self.app.running() or self.v_ds.get() != slug:
            return
        data = core.load_dataset(self.paths, slug)
        if not data or all(it.get("caption") for it in data["items"]):
            return
        if not core.engine_status(self.paths):
            self.app.log(t("v2_log_caption_later"))
            return
        self.app.log(t("v2_log_caption_auto"))
        self.on_caption()

    def job_download(self, slug, urls):
        self.app.status("v2_status_web")
        self.app.log(t("v2_log_web_start", n=len(urls)))
        try:
            st = core.download_images(self.paths, slug, urls, offline=self.v2.get("offline", True),
                                      min_side=self.v2.get("min_side", 256),
                                      progress_cb=lambda a, b: self._prog(a, b, t("v2_prog_web", a=a, b=b)),
                                      abort_event=self.app.abort, log_cb=self.app.log)
        except core.OfflineError:
            self.app.log(t("v2_web_off_offline"))
            raise JobFail()
        finally:
            self._finish(None)
        msg = t("v2_log_web_done", added=st["added"], dup=st["dup"], small=st["small"], bad=st["bad"],
                failed=st["failed"], mb=st["bytes"] / core.MB)
        self.app.log(msg)
        if st["stopped"]:
            self.app.log(t("v2_log_web_stopped"))
        self.app.post(self.set_progress, 1, 1, msg)
        self.app.status("status_done")


# =============================================================================
# LoRA / stil seçme penceresi — dışarıdan eklenenler + programın ürettikleri + stil kartları
# =============================================================================
class LoraPickerWindow:
    FILTERS = ("compatible", "all", "own", "external", "cards")
    MAX_SEL = 6

    def __init__(self, app):
        self.app = app
        self.paths = app.paths
        f = self.f = app.f
        w = self.w = tk.Toplevel(app.r)
        w.title(t("lp_title"))
        w.configure(bg=BG)
        w.transient(app.r)
        self.sel = [dict(x) for x in app.st.get("loras", [])]
        self.items = []
        self.v_search = tk.StringVar()
        self.v_filter = tk.StringVar(value="compatible")
        self.weight_vars = {}
        tk.Label(w, text=t("lp_title"), bg=BG, fg=BLUE, font=f.b12).pack(pady=(10, 2))
        self.lbl_model = tk.Label(w, bg=BG, fg=LBL, font=f.n9)
        self.lbl_model.pack()
        top = tk.Frame(w, bg=BG)
        top.pack(fill="x", padx=12, pady=(8, 4))
        tk.Label(top, text=t("v2_search_lbl"), bg=BG, fg=TXT, font=f.b9).pack(side="left")
        ent = tk.Entry(top, textvariable=self.v_search, bg=FRM, fg=TXT, insertbackground=TXT, font=f.n10, relief="flat")
        ent.pack(side="left", fill="x", expand=True, padx=(6, 10))
        self.kc_filter = KeyCombo(top, self.v_filter, lambda: [(k, t("lp_f_" + k)) for k in self.FILTERS], width=22)
        self.kc_filter.cb.pack(side="left")
        ttk.Button(top, text="🔄", width=3, command=self.reload).pack(side="left", padx=(6, 0))
        mid = tk.Frame(w, bg=BG)
        mid.pack(fill="both", expand=True, padx=12)
        cols = ("name", "kind", "base", "size", "info")
        self.tree = ttk.Treeview(mid, columns=cols, show="headings", height=12, selectmode="browse")
        widths = {"name": 200, "kind": 165, "base": 85, "size": 70, "info": 310}
        for c in cols:
            self.tree.heading(c, text=t("lp_col_" + c))
            self.tree.column(c, width=widths[c], anchor="w", stretch=(c in ("name", "info")))
        self.tree.tag_configure("sel", foreground=GREEN)
        self.tree.tag_configure("bad", foreground="#777777")
        sb = ttk.Scrollbar(mid, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.tree.bind("<Double-1>", lambda e: self.toggle())
        self.tree.bind("<Return>", lambda e: self.toggle())
        btns = tk.Frame(w, bg=BG)
        btns.pack(fill="x", padx=12, pady=6)
        ttk.Button(btns, text=t("lp_toggle"), command=self.toggle).pack(side="left")
        self.b_del = ttk.Button(btns, text=t("lp_delete"), command=self.on_delete)
        self.b_del.pack(side="left", padx=6)
        ttk.Button(btns, text=t("lp_open_addons"), command=lambda: self._open(core.addons_dir(self.paths))).pack(side="right")
        ttk.Button(btns, text=t("lp_open_own"), command=lambda: self._open(core.own_lora_dir(self.paths))).pack(
            side="right", padx=6)
        self.sel_fr = ttk.LabelFrame(w, text=t("lp_selected"))
        self.sel_fr.pack(fill="x", padx=12, pady=(2, 4))
        self.lbl_note = tk.Label(w, bg=BG, fg="#888888", font=f.n9, anchor="w", justify="left", wraplength=820)
        self.lbl_note.pack(fill="x", padx=14)
        bot = tk.Frame(w, bg=BG)
        bot.pack(fill="x", padx=12, pady=10)
        ttk.Button(bot, text=t("lp_clear"), command=self.clear).pack(side="left")
        tk.Button(bot, text=t("lp_ok"), bg=ORANGE, fg="white", font=f.b10, relief="flat", cursor="hand2",
                  activebackground="#ff5722", command=self.ok).pack(side="right", ipadx=18, ipady=2)
        for v in (self.v_search, self.v_filter):
            v.trace_add("write", lambda *a: self.populate())
        w.protocol("WM_DELETE_WINDOW", self.ok)
        self.reload()
        w.update_idletasks()
        w.minsize(w.winfo_reqwidth(), w.winfo_reqheight())
        ent.focus_set()

    def alive(self):
        try:
            return bool(self.w.winfo_exists())
        except Exception:
            return False

    def _open(self, d):
        os.makedirs(d, exist_ok=True)
        self.app.open_path(d)

    def reload(self):
        self.items = core.scan_loras(self.paths)
        self.populate()

    def _item(self, lid):
        return next((i for i in self.items if i["id"] == lid), None)

    def populate(self):
        if not self.alive():
            return
        m = self.app.model()
        self.lbl_model.config(text=t("lp_model", model=m["name"]))
        q = self.v_search.get().strip().lower()
        flt = self.v_filter.get()
        sel_ids = {s["id"] for s in self.sel}
        self.tree.delete(*self.tree.get_children())
        shown = 0
        for it in self.items:
            comp = core.lora_compat(it["base"], m["key"])
            if flt == "compatible" and comp == "no":
                continue
            if flt == "own" and not (it["origin"] == "own" and it["kind"] == "lora"):
                continue
            if flt == "external" and it["origin"] != "external":
                continue
            if flt == "cards" and it["kind"] != "card":
                continue
            learn = it.get("learn") or ""
            short = t("learn_short_" + learn) if learn in core.LEARN_KINDS else ""
            hay = " ".join(str(it.get(k, "")) for k in ("name", "tag", "trigger", "style", "id")).lower() + " " + \
                short.lower()
            if q and q not in hay:
                continue
            if it["kind"] == "card":
                kind, size, info = "📝 " + t("lp_card_kind", kind=short), "—", it.get("style", "")
            else:
                if it.get("type") == "preference":
                    kind = t("lp_own_pref")
                elif it["origin"] == "own" and short:
                    kind = core.learn_icon(learn) + " " + t("lp_own_kind", kind=short)
                else:
                    kind = t("lp_own") if it["origin"] == "own" else t("lp_external")
                size = f"{it['size_mb']:.0f} MB" if it["size_mb"] >= 1 else f"{it['size_mb'] * 1024:.0f} KB"
                info = it.get("trigger") or it.get("tag") or ""
            base = core.BASE_LABELS.get(it["base"], "?") + (" ⚠" if comp == "no" else " ?" if comp == "maybe" else "")
            tags = ("sel",) if it["id"] in sel_ids else (("bad",) if comp == "no" else ())
            name = ("✔ " if it["id"] in sel_ids else "") + it["name"]
            self.tree.insert("", "end", iid=it["id"], values=(name, kind, base, size, info[:120]), tags=tags)
            shown += 1
        own = sum(1 for i in self.items if i["origin"] == "own")
        note = t("lp_note", own=own, ext=len(self.items) - own, shown=shown)
        hidden = sum(1 for i in self.items if core.lora_compat(i["base"], m["key"]) == "no")
        if flt == "compatible" and hidden:
            note += "  " + t("lp_hidden", n=hidden)
        if not core.peft_installed(self.paths):
            note += "\n" + t("lp_note_peft", size=core.PEFT_SIZE)
        self.lbl_note.config(text=note)
        sel = self.tree.focus()
        it = self._item(sel) if sel else None
        self.b_del.state(["!disabled"] if (it and it["origin"] == "own") else ["disabled"])
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._on_select())
        self.render_selected()

    def _on_select(self):
        it = self._item(self.tree.focus())
        self.b_del.state(["!disabled"] if (it and it["origin"] == "own") else ["disabled"])

    def toggle(self):
        lid = self.tree.focus()
        if not lid:
            return
        if any(s["id"] == lid for s in self.sel):
            self.sel = [s for s in self.sel if s["id"] != lid]
        else:
            if len(self.sel) >= self.MAX_SEL:
                messagebox.showinfo(APP_NAME, t("lp_max", n=self.MAX_SEL), parent=self.w)
                return
            it = self._item(lid)
            self.sel.append({"id": lid, "w": 1.0 if (it and it["kind"] == "card") else 0.8})
        self.populate()
        if self.tree.exists(lid):
            self.tree.focus(lid)
            self.tree.selection_set(lid)

    def render_selected(self):
        for ch in self.sel_fr.winfo_children():
            ch.destroy()
        self.weight_vars = {}
        if not self.sel:
            tk.Label(self.sel_fr, text=t("lora_none"), bg=BG, fg="#888888", font=self.f.n9).pack(anchor="w", padx=8,
                                                                                               pady=4)
            return
        for s in self.sel:
            it = self._item(s["id"])
            row = tk.Frame(self.sel_fr, bg=BG)
            row.pack(fill="x", padx=8, pady=1)
            name = it["name"] if it else "⚠ " + s["id"].split(":", 1)[-1]
            tk.Label(row, text=name, bg=BG, fg=TXT, font=self.f.b9, width=28, anchor="w").pack(side="left")
            if it and it["kind"] == "lora":
                v = tk.DoubleVar(value=float(s.get("w", 0.8)))
                self.weight_vars[s["id"]] = v
                tk.Scale(row, from_=0.0, to=1.5, resolution=0.05, orient="horizontal", variable=v, length=260,
                         bg=BG, fg=TXT, troughcolor=FRM, highlightthickness=0, font=self.f.n9,
                         activebackground=BLUE, command=lambda _v, k=s["id"], var=v: self._set_w(k, var.get())
                         ).pack(side="left", padx=6)
            elif it:
                tk.Label(row, text=t("lp_card_note"), bg=BG, fg="#888888", font=self.f.n9).pack(side="left", padx=6)
            ttk.Button(row, text="✖", width=3, command=lambda k=s["id"]: self._remove(k)).pack(side="right")

    def _set_w(self, lid, w):
        for s in self.sel:
            if s["id"] == lid:
                s["w"] = round(float(w), 2)

    def _remove(self, lid):
        self.sel = [s for s in self.sel if s["id"] != lid]
        self.populate()

    def clear(self):
        self.sel = []
        self.populate()

    def on_delete(self):
        it = self._item(self.tree.focus())
        if not it or it["origin"] != "own" or self.app.running():
            return
        kind = t("lp_kind_card") if it["kind"] == "card" else t("lp_kind_lora")
        if not messagebox.askyesno(APP_NAME, t("lp_delete_q", name=it["name"], kind=kind,
                                               file=os.path.basename(it["path"])), parent=self.w):
            return
        try:
            if it["kind"] == "lora" and self.app.engine._loras:
                self.app.engine.unload()
            d, _moved = core.trash_own_item(self.paths, it)
            self.app.log(t("log_trashed", name=it["name"], kind=kind, path=d))
        except Exception as e:
            self.app.log(friendly_error(e))
        self.sel = [s for s in self.sel if s["id"] != it["id"]]
        self.reload()

    def ok(self):
        valid = {i["id"] for i in self.items}
        self.app.st["loras"] = [s for s in self.sel if s["id"] in valid][:self.MAX_SEL]
        self.app.save()
        self.app.update_lora_label()
        self.app.lora_win = None
        try:
            self.kc_filter._on_destroy()
        except Exception:
            pass
        self.w.destroy()


# =============================================================================
# Ana uygulama
# =============================================================================
class ImageStudio:
    def __init__(self, root, paths):
        self.r = root
        self.paths = paths
        self.st = load_settings(paths)
        T.set(LANGS[self.st["lang"]])
        ensure_en_json()
        self.log_file = os.path.join(paths.logs, "ails.log")

        self.f = Fonts(root)
        self.f.apply_lang(T.code)
        set_dark(root, self.f)
        self.r.title(f"{APP_NAME} v{APP_VERSION}")
        sh = self.r.winfo_screenheight()
        self.r.geometry(f"1480x{max(820, min(1000, sh - 80))}")
        self.r.minsize(1200, 820)
        self._set_icon()

        self.hw = core.detect_hardware()
        self.engine_info = core.engine_status(paths)
        self.hw.cuda_usable = bool(self.engine_info and self.engine_info.get("cuda_ok"))
        self.clients = {}
        self.engine = core.ImageEngine()
        self.q = queue.Queue()
        self.abort = threading.Event()
        self.job = None
        self.llm_installed = {}
        self.llm_loc = None
        self.gallery = []
        self.sel_index = None
        self.thumb_refs = []
        self.preview_ref = None
        self.adv_win = None
        self.style_win = None
        self.lora_win = None
        self.after_job = None          # iş başarıyla bitince arayüzde çalışacak devam adımı
        self.ratings = core.load_feedback(paths)
        self.combos = []
        self._shutdown_active = False
        self._prev_host = self.st["llm_host"]

        self._init_vars()
        self._build()
        self.apply_lang()
        self.r.protocol("WM_DELETE_WINDOW", self.on_close)
        self.r.after(60, self._poll)
        self._load_existing_gallery()
        self.log(t("log_start", app=APP_NAME, ver=APP_VERSION))
        self.log(t("log_data", path=paths.root))
        self.log(t("log_hw", txt=self._hw_text()))
        if IS_WIN and 0 <= self.hw.pagefile_gb < 4:
            self.log(t("log_pagefile_warn"))
        self.start_job("setup", self.job_setup)

    # ------------------------------------------------------------------ vars
    def _init_vars(self):
        s = self.st
        self.v_lang = tk.StringVar(value=s["lang"])
        self.v_model = tk.StringVar(value=s["model"])
        self.v_llm = tk.StringVar(value=s["llm"])
        self.v_llm_host = tk.StringVar(value=s["llm_host"])
        self.v_profile = tk.StringVar(value=s["profile"])
        self.v_use_llm = tk.BooleanVar(value=s["use_llm"])
        self.v_show_prompt = tk.BooleanVar(value=s["show_prompt"])
        self.v_aspect = tk.StringVar(value=s["aspect"])
        self.v_orient = tk.StringVar(value=s["orient"])
        self.v_res = tk.StringVar(value=s["res"])
        self.v_quality = tk.StringVar(value=s["quality"])
        self.v_creat = tk.IntVar(value=int(s["creativity"]))
        self.v_creat_auto = tk.BooleanVar(value=s["creativity_auto"])
        self.v_seed_random = tk.BooleanVar(value=s["seed_random"])
        self.v_seed = tk.StringVar(value=str(s["seed"]))
        self.v_count = tk.StringVar(value=str(s["count"]))
        self.v_outdir = tk.StringVar(value=s["output_dir"])
        self.v_steps_manual = tk.StringVar(value=str(s["steps_manual"]))
        self.v_sampler = tk.StringVar(value=s["sampler"])
        self.v_hires = tk.StringVar(value=s["hires"])
        self.v_detailer = tk.BooleanVar(value=bool(s["detailer"]))
        self.v_fix_faces = tk.BooleanVar(value=bool(s["fix_faces"]))
        self.v_fix_hands = tk.BooleanVar(value=bool(s["fix_hands"]))
        self.v_auto_neg = tk.BooleanVar(value=bool(s["auto_neg"]))
        self.v_save_base = tk.BooleanVar(value=bool(s["save_base"]))
        self.v_pref_auto = tk.BooleanVar(value=bool(s["pref_auto"]))
        self.v_pref_weight = tk.DoubleVar(value=float(s["pref_weight"]))
        self.v_manage = tk.StringVar(value="engine" if not self.engine_info else "img:" + s["model"])
        self.v_shutdown = tk.BooleanVar(value=False)
        self.v_status = tk.StringVar(value="")
        for v in (self.v_model, self.v_profile, self.v_res):
            v.trace_add("write", lambda *a: self.update_profile_info())
        self.v_model.trace_add("write", lambda *a: self.update_lora_label())
        self.v_shutdown.trace_add("write", lambda *a: (not self.v_shutdown.get()) and self._cancel_shutdown())

    def _collect_settings(self):
        self.st.update({
            "lang": self.v_lang.get(), "model": self.v_model.get(), "llm": self.v_llm.get(),
            "llm_host": self.v_llm_host.get() if self.v_llm_host.get() != "manual" else self._prev_host,
            "profile": self.v_profile.get(), "use_llm": self.v_use_llm.get(), "show_prompt": self.v_show_prompt.get(),
            "aspect": self.v_aspect.get(), "orient": self.v_orient.get(), "res": self.v_res.get(),
            "quality": self.v_quality.get(), "creativity": self.v_creat.get(), "creativity_auto": self.v_creat_auto.get(),
            "seed_random": self.v_seed_random.get(), "seed": self.v_seed.get(), "count": self.v_count.get(),
            "output_dir": self.v_outdir.get(), "steps_manual": self._steps_manual(), "sampler": self.v_sampler.get(),
            "hires": self.v_hires.get(), "detailer": self.v_detailer.get(), "fix_faces": self.v_fix_faces.get(),
            "fix_hands": self.v_fix_hands.get(), "auto_neg": self.v_auto_neg.get(), "save_base": self.v_save_base.get(),
            "pref_auto": self.v_pref_auto.get(), "pref_weight": round(float(self.v_pref_weight.get()), 2),
        })
        return self.st

    def _steps_manual(self):
        try:
            return max(0, min(core.MAX_STEPS, int(str(self.v_steps_manual.get()).strip() or 0)))
        except (TypeError, ValueError):
            return 0

    def save(self):
        save_settings(self.paths, self._collect_settings())

    # ------------------------------------------------------------------ build
    def _set_icon(self):
        for p in (os.path.join(APP_DIR, "simge.ico"), os.path.join(getattr(sys, "_MEIPASS", APP_DIR), "simge.ico")):
            if os.path.exists(p):
                try:
                    self.r.iconbitmap(p)
                    return
                except Exception:
                    pass

    def _labelframe(self, parent, row, sticky="ew", pady=3, weight=None):
        lf = ttk.LabelFrame(parent)
        lf.grid(row=row, column=0, sticky=sticky, padx=12, pady=pady)
        if weight is not None:
            parent.grid_rowconfigure(row, weight=weight)
        return lf

    def _build(self):
        r = self.r
        r.grid_columnconfigure(0, weight=0, minsize=720)
        r.grid_columnconfigure(1, weight=1)
        r.grid_rowconfigure(0, weight=1)

        left = tk.Frame(r, bg=BG)
        left.grid(row=0, column=0, sticky="nsew")
        left.grid_columnconfigure(0, weight=1)
        right = tk.Frame(r, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=8)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=1)

        # --- başlık ---
        hdr = tk.Frame(left, bg=BG)
        hdr.grid(row=0, column=0, pady=6)
        self.hdr_lbl = tk.Label(hdr, text=f" 🎨 {APP_NAME.upper()} v{APP_VERSION}", font=self.f.b18, bg=BG, fg=BLUE)
        self.hdr_lbl.pack(side="left")
        tk.Button(hdr, text="📜", bg=BG, fg="#aaaaaa", bd=0, font=self.f.n10, cursor="hand2",
                  activebackground=BG, command=self.show_licenses).pack(side="left", padx=(10, 0))

        # --- donanım çubuğu (SRT video çevirmendeki GPU/CPU çubuğu tarzında) ---
        dev = tk.Frame(left, bg=DEV_BG, highlightthickness=1, highlightbackground=DEV_BORDER)
        dev.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 6))
        dev.grid_columnconfigure(2, weight=1)
        self.lbl_dev = tk.Label(dev, bg=DEV_BG, fg=BLUE, font=self.f.b9)
        self.lbl_dev.grid(row=0, column=0, sticky="nw", padx=(10, 6), pady=(5, 0))
        self.lbl_hw = tk.Label(dev, bg=DEV_BG, fg=TXT, font=self.f.b9, anchor="w", justify="left", wraplength=600)
        self.lbl_hw.grid(row=0, column=1, columnspan=2, sticky="w", pady=(5, 0))
        self.lbl_prof = tk.Label(dev, bg=DEV_BG, fg=BLUE, font=self.f.b9)
        self.lbl_prof.grid(row=1, column=0, sticky="w", padx=(10, 6), pady=5)
        kc = KeyCombo(dev, self.v_profile, self._profile_items, width=22)
        kc.cb.grid(row=1, column=1, sticky="w", pady=5)
        self.combos.append(kc)
        self.lbl_prof_info = tk.Label(dev, bg=DEV_BG, fg=WARN, font=self.f.b9, anchor="w", justify="left", wraplength=430)
        self.lbl_prof_info.grid(row=1, column=2, sticky="w", padx=(8, 8), pady=5)

        # --- turuncu özellik butonları (v2 / v3 pencereleri) ---
        feat = tk.Frame(left, bg=BG)
        feat.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 6))
        feat.grid_columnconfigure(0, weight=1)
        self.btn_style = tk.Button(feat, bg=ORANGE, fg="white", font=self.f.b11, relief="flat", cursor="hand2",
                                   activebackground="#ff5722", command=self.open_style_window)
        self.btn_style.grid(row=0, column=0, sticky="ew", ipady=5)

        # --- 1. prompt ---
        self.l1 = self._labelframe(left, 3, sticky="nsew", weight=2)
        opt = tk.Frame(self.l1, bg=BG)
        opt.pack(side="bottom", fill="x", padx=6, pady=(0, 3))
        self.chk_llm = dark_check(opt, self.v_use_llm, self.f, fg=GREEN)
        self.chk_llm.pack(side="left")
        self.chk_show = dark_check(opt, self.v_show_prompt, self.f, fg=WARN)
        self.chk_show.pack(side="left", padx=(12, 0))
        self.chk_detail = dark_check(opt, self.v_detailer, self.f, fg=BLUE)
        self.chk_detail.pack(side="left", padx=(12, 0))
        lrow = tk.Frame(self.l1, bg=BG)
        lrow.pack(side="bottom", fill="x", padx=6, pady=(0, 2))
        self.b_lora = ttk.Button(lrow, command=self.open_lora_picker)
        self.b_lora.pack(side="left")
        self.lbl_loras = tk.Label(lrow, bg=BG, fg=BLUE, font=self.f.n9, anchor="w", justify="left", wraplength=560)
        self.lbl_loras.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.txt_prompt = tk.Text(self.l1, height=3, font=self.f.n10, bg=FRM, fg=TXT, insertbackground=TXT,
                                  wrap="word", relief="flat")
        self.txt_prompt.pack(fill="both", expand=True, padx=10, pady=(4, 2))

        # --- 2. çıktı klasörü ---
        self.l2 = self._labelframe(left, 4)
        self.lbl_out = ttk.Label(self.l2, textvariable=self.v_outdir)
        self.lbl_out.pack(side="left", padx=10, pady=4)
        self.b_open = ttk.Button(self.l2, command=lambda: self.open_path(self.v_outdir.get()))
        self.b_open.pack(side="right", padx=(2, 10), pady=3)
        self.b_out = ttk.Button(self.l2, command=self.choose_outdir)
        self.b_out.pack(side="right", padx=4, pady=3)

        # --- 3. boyut ve adet ---
        self.l3 = self._labelframe(left, 5)
        row3 = tk.Frame(self.l3, bg=BG)
        row3.pack(fill="x", padx=6, pady=4)
        self.lbl_aspect = tk.Label(row3, bg=BG, fg=TXT, font=self.f.n9)
        self.lbl_aspect.pack(side="left", padx=(4, 2))
        kc = KeyCombo(row3, self.v_aspect, self._aspect_items, width=9)
        kc.cb.pack(side="left")
        self.combos.append(kc)
        self.lbl_orient = tk.Label(row3, bg=BG, fg=TXT, font=self.f.n9)
        self.lbl_orient.pack(side="left", padx=(10, 2))
        kc = KeyCombo(row3, self.v_orient, self._orient_items, width=10)
        kc.cb.pack(side="left")
        self.combos.append(kc)
        self.lbl_res = tk.Label(row3, bg=BG, fg=TXT, font=self.f.n9)
        self.lbl_res.pack(side="left", padx=(10, 2))
        kc = KeyCombo(row3, self.v_res, self._res_items, width=9)
        kc.cb.pack(side="left")
        self.combos.append(kc)
        self.lbl_count = tk.Label(row3, bg=BG, fg=TXT, font=self.f.n9)
        self.lbl_count.pack(side="left", padx=(10, 2))
        self.cb_count = ttk.Combobox(row3, textvariable=self.v_count, values=[str(i) for i in range(1, 9)],
                                     state="readonly", width=3)
        self.cb_count.pack(side="left")
        self.b_adv = ttk.Button(row3, command=self.open_advanced)
        self.b_adv.pack(side="right", padx=4)

        # --- 4. yapay zeka ayarları ---
        self.l4 = self._labelframe(left, 6)
        row4 = tk.Frame(self.l4, bg=BG)
        row4.pack(fill="x", padx=6, pady=(4, 2))
        self.lbl_imodel = tk.Label(row4, bg=BG, fg=TXT, font=self.f.n9)
        self.lbl_imodel.pack(side="left", padx=(4, 2))
        self.kc_model = KeyCombo(row4, self.v_model, self._image_model_items, width=32)
        self.kc_model.cb.pack(side="left")
        self.combos.append(self.kc_model)
        self.lbl_llm = tk.Label(row4, bg=BG, fg=TXT, font=self.f.n9)
        self.lbl_llm.pack(side="left", padx=(12, 2))
        self.kc_llm = KeyCombo(row4, self.v_llm, self._llm_items, width=22)
        self.kc_llm.cb.pack(side="left", fill="x", expand=True)
        self.combos.append(self.kc_llm)
        row4b = tk.Frame(self.l4, bg=BG)
        row4b.pack(fill="x", padx=6, pady=(2, 0))
        self.lbl_llm_server = tk.Label(row4b, bg=BG, fg=TXT, font=self.f.n9)
        self.lbl_llm_server.pack(side="left", padx=(4, 2))
        self.kc_host = KeyCombo(row4b, self.v_llm_host, self._host_items, width=24, on_change=self.on_host_change)
        self.kc_host.cb.pack(side="left")
        self.combos.append(self.kc_host)
        self.b_scan = ttk.Button(row4b, command=self.on_scan)
        self.b_scan.pack(side="left", padx=(8, 2))
        self.b_help = ttk.Button(row4b, text="❔", width=3, command=self.on_remote_help)
        self.b_help.pack(side="left", padx=2)
        self.lbl_llm_loc = tk.Label(self.l4, bg=BG, fg="#888888", font=self.f.n9, anchor="w", justify="left", wraplength=680)
        self.lbl_llm_loc.pack(fill="x", padx=10, pady=(1, 4))

        # --- 5. özel talimat / negatif ---
        self.l5 = self._labelframe(left, 7, sticky="nsew", weight=1)
        self.lbl_hint5 = tk.Label(self.l5, bg=BG, fg="#888888", font=self.f.n9, anchor="w", justify="left", wraplength=680)
        self.lbl_hint5.pack(side="bottom", fill="x", padx=10, pady=(0, 3))
        self.txt_extra = tk.Text(self.l5, height=2, font=self.f.n10, bg=FRM, fg=TXT, insertbackground=TXT,
                                 wrap="word", relief="flat")
        self.txt_extra.pack(fill="both", expand=True, padx=10, pady=(4, 1))

        # --- 6. indir / sil ---
        self.l6 = self._labelframe(left, 8)
        row6 = tk.Frame(self.l6, bg=BG)
        row6.pack(fill="x", padx=6, pady=4)
        self.b_ddir = ttk.Button(row6, command=self.on_data_dir)
        self.b_ddir.pack(side="right", padx=(4, 4))
        self.b_del = ttk.Button(row6, command=self.on_delete)
        self.b_del.pack(side="right", padx=4)
        self.b_dl = ttk.Button(row6, command=self.on_download)
        self.b_dl.pack(side="right", padx=4)
        self.kc_manage = KeyCombo(row6, self.v_manage, self._manage_items, width=40)
        self.kc_manage.cb.pack(side="left", fill="x", expand=True, padx=(4, 4))
        self.combos.append(self.kc_manage)

        # --- 7. arayüz dili ---
        self.l7 = self._labelframe(left, 9)
        self.cb_lang = ttk.Combobox(self.l7, textvariable=self.v_lang, values=list(LANGS.keys()), state="readonly", width=22)
        self.cb_lang.pack(side="left", padx=10, pady=4)
        self.cb_lang.bind("<<ComboboxSelected>>", lambda e: self.on_lang())

        # --- ilerleme ---
        stf = ttk.Frame(left)
        stf.grid(row=10, column=0, sticky="ew", padx=12, pady=3)
        self.pg = ttk.Progressbar(stf, length=100, mode="determinate", maximum=100)
        self.pg.pack(fill="x", pady=2)
        info = ttk.Frame(stf)
        info.pack(fill="x")
        self.lbl_pct = ttk.Label(info, font=self.f.b10)
        self.lbl_pct.pack(side="left")
        self.lbl_counter = ttk.Label(info, font=self.f.b10)
        self.lbl_counter.pack(side="right")
        ttk.Label(info, textvariable=self.v_status, foreground=BLUE, font=self.f.b10, anchor="center").pack(
            side="left", fill="x", expand=True)

        # --- başlat / durdur ---
        act = tk.Frame(left, bg=BG)
        act.grid(row=11, column=0, sticky="ew", padx=12, pady=4)
        self.bs = tk.Button(act, bg=BTN, fg="white", font=self.f.b12, command=self.on_start, cursor="hand2",
                            activebackground="#555555", activeforeground="white")
        self.bs.pack(side="left", fill="x", expand=True, ipadx=20, ipady=6)
        self.chk_shutdown = dark_check(act, self.v_shutdown, self.f, fg=WARN)
        self.chk_shutdown.configure(font=self.f.b10)
        self.chk_shutdown.pack(side="right", padx=10)

        # --- log ---
        self.log_fr = self._labelframe(left, 12, sticky="nsew", pady=(3, 10), weight=3)
        # Uzun satırlı fontlarda (ör. Korece) bile metin kutuları ve log kaybolmasın.
        left.grid_rowconfigure(3, minsize=140)
        left.grid_rowconfigure(7, minsize=80)
        left.grid_rowconfigure(12, minsize=92)
        sb = ttk.Scrollbar(self.log_fr, orient="vertical")
        self.txt_log = tk.Text(self.log_fr, height=5, bg="#111111", fg="#00ff00", font=self.f.mono,
                               yscrollcommand=sb.set, relief="flat", wrap="word")
        sb.config(command=self.txt_log.yview)
        sb.pack(side="right", fill="y", pady=5)
        self.txt_log.pack(side="left", fill="both", expand=True, padx=(5, 0), pady=5)

        # --- sağ: galeri ---
        self.gal_fr = ttk.LabelFrame(right)
        self.gal_fr.grid(row=0, column=0, sticky="nsew")
        self.gal_fr.grid_columnconfigure(0, weight=1)
        self.gal_fr.grid_rowconfigure(0, weight=3)
        self.gal_fr.grid_rowconfigure(3, weight=2)
        self.lbl_preview = tk.Label(self.gal_fr, bg=CANVAS_BG, fg="#666666", font=self.f.b11)
        self.lbl_preview.grid(row=0, column=0, sticky="nsew", padx=8, pady=(6, 4))
        self.lbl_preview.bind("<Configure>", lambda e: self._schedule_preview())
        # bilgi satırı + 👍/👎 (sağda; alt buton satırı taşmasın diye burada)
        meta_row = tk.Frame(self.gal_fr, bg=BG)
        meta_row.grid(row=1, column=0, sticky="ew", padx=(10, 8))
        meta_row.grid_columnconfigure(0, weight=1)
        self.lbl_meta = tk.Label(meta_row, bg=BG, fg=LBL, font=self.f.n9, anchor="w", justify="left", wraplength=520)
        self.lbl_meta.grid(row=0, column=0, sticky="ew")
        rate = tk.Frame(meta_row, bg=BG)
        rate.grid(row=0, column=1, sticky="ne", padx=(6, 0))
        self.b_like = ttk.Button(rate, width=4, command=lambda: self.on_rate(1))
        self.b_dislike = ttk.Button(rate, width=4, command=lambda: self.on_rate(-1))
        self.b_like.pack(side="left", padx=(0, 3))
        self.b_dislike.pack(side="left")
        meta_row.bind("<Configure>", lambda e: self.lbl_meta.config(
            wraplength=max(140, e.width - rate.winfo_reqwidth() - 12)))
        # buton satırı: sığmazsa (uzun dil / dar pencere) kendiliğinden 2. satıra geçer
        acts = tk.Frame(self.gal_fr, bg=BG)
        acts.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        self._acts = acts
        self._act_rows = [tk.Frame(acts, bg=BG) for _ in range(4)]   # önce satırlar (butonlar üstte kalsın)
        self.b_show = ttk.Button(acts, command=self.on_show_file)
        self.b_up = ttk.Button(acts, command=self.on_upscale)
        self.b_copy = ttk.Button(acts, command=self.on_copy_prompt)
        self.b_reuse = ttk.Button(acts, command=self.on_reuse)
        self._act_btns = (self.b_show, self.b_up, self.b_copy, self.b_reuse)
        self._act_sig = None
        acts.bind("<Configure>", lambda e: self._layout_actions())
        self._layout_actions(force_width=640)

        thumbs = tk.Frame(self.gal_fr, bg="#161616", bd=1, relief="sunken")
        thumbs.grid(row=3, column=0, sticky="nsew", padx=8, pady=(0, 8))
        thumbs.grid_columnconfigure(0, weight=1)
        thumbs.grid_rowconfigure(0, weight=1)
        self.th_canvas = tk.Canvas(thumbs, bg="#161616", highlightthickness=0)
        self.th_canvas.grid(row=0, column=0, sticky="nsew")
        tsb = ttk.Scrollbar(thumbs, orient="vertical", command=self.th_canvas.yview)
        tsb.grid(row=0, column=1, sticky="ns")
        self.th_canvas.configure(yscrollcommand=tsb.set)
        self.th_inner = tk.Frame(self.th_canvas, bg="#161616")
        self.th_canvas.create_window((0, 0), window=self.th_inner, anchor="nw")
        self.th_inner.bind("<Configure>", lambda e: self.th_canvas.configure(scrollregion=self.th_canvas.bbox("all")))
        self.th_canvas.bind("<Configure>", lambda e: self._layout_thumbs())
        for w in (self.th_canvas, self.th_inner):
            w.bind("<MouseWheel>", lambda e: self.th_canvas.yview_scroll(int(-e.delta / 120), "units"))

    # ------------------------------------------------------------- item lists
    def _profile_items(self):
        return [("auto", t("prof_auto"))] + [(p, t("prof_" + p)) for p in core.PROFILES]

    def _aspect_items(self):
        return [("auto", t("auto"))] + [(a, a) for a in core.ASPECTS]

    def _orient_items(self):
        return [("auto", t("auto")), ("landscape", t("orient_landscape")), ("portrait", t("orient_portrait"))]

    def _res_items(self):
        return [("auto", t("auto")), ("low", t("res_low")), ("normal", t("res_normal")), ("high", t("res_high"))]

    def _quality_items(self):
        m = self.model()
        return [("auto", t("auto"))] + [(q, f"{t('q_' + q)} ({t('adv_steps', n=m['steps'][q])})")
                                        for q in core.QUALITIES]

    def _sampler_items(self):
        return [("auto", t("sampler_auto"))] + [(k, core.SAMPLER_NAMES[k]) for k in core.SAMPLERS if k != "auto"]

    def _hires_items(self):
        return [("auto", t("auto")), ("on", t("hires_on")), ("off", t("hires_off"))]

    def _image_model_items(self):
        out = []
        for m in core.IMAGE_MODELS:
            ok = core.is_image_model_installed(self.paths.models, m)
            out.append((m["key"], f"{'✔' if ok else '⬇'} {m['name']}" + ("" if ok else "  " + t("not_installed"))))
        return out

    def _host_names(self):
        return self.llm_installed.get(self.v_llm_host.get(), [])

    def _llm_items(self):
        names = self._host_names()
        out = []
        for m in core.LLM_MODELS:
            ok = core.model_in_list(m["key"], names)
            out.append((m["key"], f"{'✔' if ok else '⬇'} {m['name']}" + ("" if ok else "  " + t("not_installed"))))
        return out

    def _host_items(self):
        items = [("local", t("llm_local"))]
        items += [(h, "🌐 " + h) for h in self.st.get("known_hosts", [])]
        items.append(("manual", t("llm_manual")))
        return items

    def _fit_text(self, level):
        return t({"ok": "fit_ok", "slow": "fit_slow", "no": "fit_no"}[level])

    def _manage_items(self):
        root = self.paths.models
        kind = core.engine_kind(self.hw)
        dl, disk = core.ENGINE_SIZE[kind]
        info = core.engine_status(self.paths)
        mode = ("CUDA" if info.get("cuda_ok") else "CPU") if info else ("CUDA" if kind == "cuda" else "CPU")
        out = [("engine", f"{'✔' if info else '⬇'} [{t('kind_engine')}] {t('engine_name')} — {mode} | "
                          f"{t('engine_size', dl=dl, disk=disk)} | BSD-3 / Apache-2.0 | {t('desc_engine')}")]
        for m in core.IMAGE_MODELS:
            inst = core.is_image_model_installed(root, m)
            size = f"{core.installed_size_gb(root, m):.1f} GB" if inst else f"~{m['size_gb']:.1f} GB"
            out.append(("img:" + m["key"], f"{'✔' if inst else '⬇'} [{t('kind_image')}] {m['name']} | {size} | "
                                           f"{m['license']} | {self._fit_text(core.fit_level(m, self.hw))} | {t(m['desc'])}"))
        names = self._host_names()
        for m in core.LLM_MODELS:
            inst = core.model_in_list(m["key"], names)
            fit = "ok" if (self.v_llm_host.get() != "local" or self.hw.vram_gb >= m["vram"]) else "slow"
            out.append(("llm:" + m["key"], f"{'✔' if inst else '⬇'} [{t('kind_llm')}] {m['name']} | {m['size']} | "
                                           f"{m['license']} | {self._fit_text(fit)} | {t(m['desc'])}"))
        u = core.UPSCALER
        inst = core.is_upscaler_installed(root)
        out.append(("up:" + u["key"], f"{'✔' if inst else '⬇'} [{t('kind_upscale')}] {u['name']} | {u['size']} | "
                                      f"{u['license']} | {self._fit_text('ok')} | {t(u['desc'])}"))
        cp = core.CAPTIONER
        inst = core.is_captioner_installed(root)
        out.append(("cap:" + cp["key"], f"{'✔' if inst else '⬇'} [{t('kind_caption')}] {cp['name']} | {cp['size']} | "
                                       f"{cp['license']} | {self._fit_text('ok')} | {t(cp['desc'])}"))
        dt = core.DETAILER
        inst = core.is_detailer_installed(self.paths)
        out.append(("det:" + dt["key"], f"{'✔' if inst else '⬇'} [{t('kind_detailer')}] {t('detailer_name')} | {dt['size']} | "
                                        f"{dt['license']} | {self._fit_text('ok')} | {t(dt['desc'])}"))
        return out

    def refresh_lists(self):
        for kc in self.combos:
            kc.refresh()
        self.update_profile_info()
        self._update_llm_loc()
        self.lbl_hw.config(text=self._hw_text())

    # ------------------------------------------------------------- language
    def on_lang(self):
        T.set(LANGS.get(self.v_lang.get(), "en"))
        self.f.apply_lang(T.code)
        self.apply_lang()
        self.save()
        cfg = load_pointer()
        cfg["lang"] = self.v_lang.get()
        save_pointer(cfg)

    def apply_lang(self):
        self.btn_style.config(text=t("btn_style"))
        self.lbl_dev.config(text=t("dev_lbl"))
        self.lbl_prof.config(text=t("profile_lbl"))
        self.l1.config(text=t("sec1"))
        self.chk_llm.config(text=t("use_llm"))
        self.chk_show.config(text=t("show_prompt"))
        self.chk_detail.config(text=t("use_detailer"))
        self.b_lora.config(text=t("btn_lora"))
        self.update_lora_label()
        self.l2.config(text=t("sec2"))
        self.b_out.config(text=t("btn_folder"))
        self.b_open.config(text=t("btn_open_folder"))
        self.l3.config(text=t("sec3"))
        self.lbl_aspect.config(text=t("lbl_aspect"))
        self.lbl_orient.config(text=t("lbl_orient"))
        self.lbl_res.config(text=t("lbl_res"))
        self.lbl_count.config(text=t("lbl_count"))
        self.b_adv.config(text=t("btn_advanced"))
        self.l4.config(text=t("sec4"))
        self.lbl_imodel.config(text=t("lbl_image_model"))
        self.lbl_llm.config(text=t("lbl_llm"))
        self.lbl_llm_server.config(text=t("lbl_llm_server"))
        self.b_scan.config(text=t("btn_scan"))
        self.l5.config(text=t("sec5"))
        self.lbl_hint5.config(text=t("sec5_hint"))
        self.l6.config(text=t("sec6"))
        self.b_dl.config(text=t("btn_download"))
        self.b_del.config(text=t("btn_delete"))
        self.b_ddir.config(text=t("btn_data_dir"))
        self.l7.config(text=t("sec7"))
        self.chk_shutdown.config(text=t("chk_shutdown"))
        self.log_fr.config(text=t("log_title"))
        self.gal_fr.config(text=t("gallery_title"))
        self.b_show.config(text=t("btn_show_file"))
        self.b_up.config(text=t("btn_upscale"))
        self._update_rate_buttons()
        self.b_copy.config(text=t("btn_copy_prompt"))
        self.b_reuse.config(text=t("btn_reuse"))
        self._act_sig = None
        self.r.after(60, self._layout_actions)              # yazı uzunluğu değişti → satırları yeniden diz
        self._set_running_ui(self.running())
        if not self.v_status.get() or not self.running():
            self.v_status.set(t("status_ready"))
        self.set_progress(self.pg["value"])
        self.lbl_counter.config(text="")
        self.refresh_lists()
        if not self.gallery:
            self.lbl_preview.config(text=t("gallery_empty"), image="")
        else:
            self._show_meta()
        if self.adv_win is not None and self.adv_win.winfo_exists():
            self._adv_close()
            self.open_advanced()

    # ------------------------------------------------------------- hardware / status texts
    def _hw_text(self):
        parts = []
        if self.hw.vendor == "nvidia":
            parts.append(t("hw_gpu", name=self.hw.gpu_name, vram=self.hw.vram_gb))
        else:
            parts.append(t("hw_nogpu"))
        parts.append(t("hw_ram", ram=self.hw.ram_total_gb))
        if self.hw.pagefile_gb >= 0:
            parts.append(t("hw_page", pf=self.hw.pagefile_gb))
        parts.append(t("hw_disk", drive=core.drive_label(self.paths.root), free=max(0.0, core.disk_free_gb(self.paths.root))))
        info = core.engine_status(self.paths)
        parts.append(t("hw_engine_ok", mode="CUDA" if info.get("cuda_ok") else "CPU") if info else t("hw_engine_missing"))
        return "  |  ".join(parts)

    def model(self):
        return core.IMAGE_MODEL_BY_KEY.get(self.v_model.get(), core.IMAGE_MODELS[1])

    def effective_profile(self, m=None):
        m = m or self.model()
        p = self.v_profile.get()
        if p == "auto":
            p = core.auto_profile(m, self.hw)
        if core.engine_status(self.paths) and not self.hw.cuda_usable:
            p = "cpu"
        return p

    def update_profile_info(self):
        if not hasattr(self, "lbl_prof_info"):
            return
        m = self.model()
        p = self.effective_profile(m)
        txt = t("model_profile_info", model=m["name"], profile=t("prof_" + p), desc=t("prof_desc_" + p))
        txt += "   " + self._fit_text(core.fit_level(m, self.hw))
        if self.v_res.get() == "high" and p in ("low", "cpu"):
            txt += "\n" + t("adv_warn_high")
        self.lbl_prof_info.config(text=txt)

    def _update_llm_loc(self):
        if not hasattr(self, "lbl_llm_loc"):
            return
        h = self.v_llm_host.get()
        if h not in ("local", "manual"):
            txt = t("llm_loc_remote", host=h)
        elif self.llm_loc:
            txt = t("llm_loc_local", path=self.llm_loc[0], free=max(0.0, core.disk_free_gb(self.llm_loc[0])))
        elif not core.ollama_installed(self.paths):
            txt = t("llm_loc_none")
        else:
            txt = ""
        self.lbl_llm_loc.config(text=txt)

    # ------------------------------------------------------------- thread-safe UI
    def post(self, fn, *a, **kw):
        self.q.put((fn, a, kw))

    def _poll(self):
        try:
            while True:
                fn, a, kw = self.q.get_nowait()
                try:
                    fn(*a, **kw)
                except Exception:
                    self._write_file_log(traceback.format_exc())
        except queue.Empty:
            pass
        self.r.after(60, self._poll)

    def call_ui(self, fn, *a):
        """Arka plan thread'inden arayüzde bir fonksiyon çalıştırıp sonucunu bekler."""
        box, ev = {}, threading.Event()

        def run():
            try:
                box["v"] = fn(*a)
            finally:
                ev.set()

        self.post(run)
        while not ev.wait(0.2):
            if not self.r.winfo_exists():
                return None
        return box.get("v")

    def _write_file_log(self, txt):
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(time.strftime("[%Y-%m-%d %H:%M:%S] ") + txt + "\n")
        except Exception:
            pass

    def log(self, txt):
        self._write_file_log(txt)
        if threading.current_thread() is not threading.main_thread():
            self.post(self.log_ui, txt)
        else:
            self.log_ui(txt)

    def log_ui(self, txt):
        if not self.r.winfo_exists():
            return
        self.txt_log.insert(tk.END, time.strftime("[%H:%M:%S] ") + txt + "\n")
        self.txt_log.see(tk.END)

    def status(self, key, **kw):
        self.post(self.v_status.set, t(key, **kw))

    def set_progress(self, pct, counter=None):
        if str(self.pg["mode"]) != "determinate":
            self.pg.stop()
            self.pg.config(mode="determinate")
        pct = max(0.0, min(100.0, float(pct)))
        self.pg["value"] = pct
        self.lbl_pct.config(text=t("progress", p=int(pct)))
        if counter is not None:
            self.lbl_counter.config(text=counter)

    def set_indeterminate(self, counter=""):
        if str(self.pg["mode"]) != "indeterminate":
            self.pg.config(mode="indeterminate")
            self.pg.start(12)
        self.lbl_pct.config(text="")
        self.lbl_counter.config(text=counter)

    def progress(self, pct, counter=None):
        self.post(self.set_progress, pct, counter)

    def bytes_progress(self, done, total):
        if total and total > 0:
            self.progress(done * 100.0 / total, f"{done / core.GB:.2f} / {total / core.GB:.2f} GB")
        else:
            self.post(self.set_indeterminate, t("status_install"))

    def client(self, host):
        key = core.LOCAL_HOST if host in ("local", "manual", None) else host
        c = self.clients.get(key)
        if c is None:
            c = core.OllamaClient(key)
            self.clients[key] = c
        return c

    # ------------------------------------------------------------- jobs
    def running(self):
        # İş, thread'i bittikten sonra da "bitti" sinyali arayüzde işlenene kadar çalışıyor sayılır.
        return self.job is not None

    def _set_running_ui(self, running):
        for b in (self.b_dl, self.b_del, self.b_up, self.b_scan, self.b_ddir):
            b.state(["disabled"] if running else ["!disabled"])
        if running:
            self.bs.config(text=t("btn_stop"), bg=RED, activebackground="#d32f2f")
        else:
            self.bs.config(text=t("btn_start"), bg=BTN, activebackground="#555555")
        if self.style_win is not None and self.style_win.alive():
            self.style_win.update_state()

    def start_job(self, kind, target, *args):
        if self.running():
            messagebox.showinfo(APP_NAME, t("busy"), parent=self.r)
            return False
        self.abort.clear()
        self._set_running_ui(True)

        def wrapper():
            ok = False
            try:
                target(*args)
                ok = True
            except core.AbortedError as e:
                if e.reason == "ram":
                    self.status("status_stopped")
                elif e.reason == "cancel":
                    self.log(t("log_cancelled"))
                    self.status("status_ready")
                else:
                    self.log(t("log_stopped"))
                    self.status("status_stopped")
            except JobFail:
                self.status("status_error")
            except Exception as e:
                self._write_file_log(traceback.format_exc())
                self.log(friendly_error(e))
                self.status("status_error")
                if isinstance(e, core.EngineBrokenError) or core.looks_like_engine_broken(e):
                    self.post(self._engine_broken_prompt)
            finally:
                self.post(self._job_finished, kind, ok, threading.current_thread())

        self.job = threading.Thread(target=wrapper, daemon=True)
        self.job.start()
        return True

    def _job_finished(self, kind, ok, thread=None):
        if thread is not None and self.job is not thread:
            return
        self.job = None
        self._set_running_ui(False)
        if str(self.pg["mode"]) != "determinate":
            self.set_progress(100 if ok else 0, "")
        self.refresh_lists()
        nxt, self.after_job = self.after_job, None
        if nxt is not None and ok:
            self.r.after(200, nxt)
        if kind == "generate" and ok and self.v_shutdown.get():
            self._trigger_shutdown()

    def _engine_broken_prompt(self):
        cfg = load_pointer()
        cfg["pending_runtime_reset"] = True
        save_pointer(cfg)
        if messagebox.askyesno(t("restart_title"), t("err_engine_broken") + "\n\n" + t("restart_q"), parent=self.r):
            self.restart_program()

    # ------------------------------------------------------------- setup (açılış: indirme YOK)
    def job_setup(self):
        self.status("status_setup")
        info = self.engine_info
        if info:
            self.log(t("log_engine_ready", torch=info.get("torch", "?"), mode="CUDA" if info.get("cuda_ok") else "CPU"))
        else:
            self.log(t("log_engine_missing"))
        if core.ollama_installed(self.paths):
            self.llm_loc = core.ollama_models_dir()
        host = self.v_llm_host.get()
        if host == "local":
            self._refresh_local_models()
            if self.v_use_llm.get() and not core.ollama_installed(self.paths) and \
                    not self.client("local").alive(timeout=1.0):
                self._scan(auto=True)     # yerelde Ollama yoksa açılışta TEK bir sınırlı arama
        else:
            self._check_remote(host)
        self.status("status_ready")

    def _refresh_local_models(self):
        c = self.client("local")
        if c.alive(timeout=1.0):
            self.llm_installed["local"] = c.installed_models()
        elif self.llm_loc:
            self.llm_installed["local"] = core.installed_models_fs(self.llm_loc[0])
        else:
            self.llm_installed["local"] = []
        self.llm_installed[core.LOCAL_HOST] = self.llm_installed["local"]

    def _check_remote(self, host):
        c = self.client(host)
        c.reset_backoff()
        if c.alive(timeout=3.0):
            names = c.installed_models()
            self.llm_installed[host] = names
            self.log(t("remote_ok", host=host, ver=c.version(), n=len(names)))
            return True
        self.log(t("remote_fail", host=host))
        return False

    def _scan(self, auto=False):
        self.status("status_scan")
        self.log(t("scan_start", n=10))
        found = core.discover_lan_ollama(known=self.st.get("known_hosts", []), limit=10, delay=0.3,
                                         abort_event=self.abort)
        hosts = [h for h, _v in found]
        if hosts:
            def add():
                for h in hosts:
                    if h not in self.st["known_hosts"]:
                        self.st["known_hosts"].append(h)
                self.save()
                self.refresh_lists()
            self.post(add)
            self.log(t("scan_found", hosts=", ".join(hosts)))
        elif not auto:
            self.log(t("scan_none"))
        self.status("status_ready")

    def on_scan(self):
        self.start_job("scan", self._scan, False)

    def on_remote_help(self):
        messagebox.showinfo(t("remote_help_title"), t("remote_help"), parent=self.r)

    def on_host_change(self):
        v = self.v_llm_host.get()
        if v == "manual":
            ip = simpledialog.askstring(t("ask_ip_title"), t("ask_ip"), parent=self.r)
            host = core.normalize_host(ip or "")
            if not host or host.startswith("127."):
                if ip:
                    messagebox.showwarning(t("ask_ip_title"), t("bad_ip"), parent=self.r)
                self.v_llm_host.set(self._prev_host)
                return
            if host not in self.st["known_hosts"]:
                self.st["known_hosts"].insert(0, host)
            self.v_llm_host.set(host)
            v = host
        self._prev_host = v
        self.save()
        self.refresh_lists()
        if not self.running():
            if v == "local":
                self.start_job("check", self._refresh_local_models)
            else:
                self.start_job("check", self._check_remote, v)

    # ------------------------------------------------------------- download plan (UI thread)
    def _plan(self, engine=False, model=None, upscaler=False, llm=None, detailer=False, peft=False, captioner=False,
              cv2=False):
        """Neyin indirileceğini listeler ve TEK bir onay ister. Dönüş: plan sözlüğü ya da None (iptal)."""
        items, plan = [], {}
        if engine and not core.engine_status(self.paths):
            dl, disk = core.ENGINE_SIZE[core.engine_kind(self.hw)]
            items.append((t("engine_name"), t("engine_size", dl=dl, disk=disk), self.paths.runtime))
            plan["engine"] = True
        if model and not core.is_image_model_installed(self.paths.models, model):
            items.append((model["name"], f"~{model['size_gb']:.1f} GB", os.path.join(self.paths.models, "image")))
            plan["model"] = True
        if upscaler and not core.is_upscaler_installed(self.paths.models):
            items.append((core.UPSCALER["name"], core.UPSCALER["size"], os.path.join(self.paths.models, "upscale")))
            plan["upscaler"] = True
        if peft and not core.peft_installed(self.paths) and not plan.get("engine"):
            items.append((t("peft_name"), core.PEFT_SIZE, self.paths.runtime))
            plan["peft"] = True
        if captioner and not core.is_captioner_installed(self.paths.models):
            items.append((core.CAPTIONER["name"] + " — MIT", core.CAPTIONER["size"], core.captioner_dir(self.paths.models)))
            plan["captioner"] = True
        if cv2 and not core.cv2_installed(self.paths) and not (detailer and not core.is_detailer_installed(self.paths)):
            items.append(("OpenCV (opencv-python-headless) — Apache-2.0", core.CV2_SIZE, self.paths.runtime))
            plan["cv2"] = True
        if detailer and not core.is_detailer_installed(self.paths):
            items.append((t("detailer_name") + " (OpenCV + YuNet + MediaPipe Palm)", core.DETAILER["size"],
                          core.detector_dir(self.paths.models)))
            plan["detailer"] = True
        if llm and self.v_llm_host.get() == "local":
            lm = core.LLM_MODEL_BY_KEY.get(llm, {"name": llm, "size": "?", "size_gb": 3.0})
            c = self.client("local")
            alive = c.alive(timeout=0.8, force=True)
            if not alive and not core.ollama_installed(self.paths):
                if IS_WIN:
                    items.append((t("ollama_name"), t("ollama_size"), self.paths.ollama_dir))
                    items.append((lm["name"], lm["size"], self.paths.ollama_models))
                    plan["ollama_install"] = True
                    plan["llm_pull"] = llm
            else:
                mdir, src = core.ollama_models_dir()
                self.llm_loc = (mdir, src)
                names = c.installed_models() if alive else core.installed_models_fs(mdir)
                self.llm_installed["local"] = names
                if not core.model_in_list(llm, names):
                    if src == "default" and not core.same_drive(mdir, self.paths.root) and \
                            not self.st.get("ollama_offer_done"):
                        self.st["ollama_offer_done"] = True
                        self.save()
                        if messagebox.askyesno(APP_NAME, t("ollama_c_offer", path=mdir,
                                                           free=max(0.0, core.disk_free_gb(mdir)),
                                                           target=self.paths.ollama_models), parent=self.r):
                            plan["ollama_redirect"] = True
                            mdir = self.paths.ollama_models
                    items.append((lm["name"], lm["size"], mdir))
                    plan["llm_pull"] = llm
        if items and not self._confirm_downloads(items):
            return None
        return plan

    def _confirm_downloads(self, items):
        lines = [t("confirm_head"), ""]
        drives = {}
        for name, size, dest in items:
            lines.append(t("confirm_item", name=name, size=size, path=dest))
            drives[core.drive_label(dest)] = max(0.0, core.disk_free_gb(dest))
        lines += ["", t("confirm_free", drives=",  ".join(f"{d} {f:.1f} GB" for d, f in drives.items())), "",
                  t("confirm_q")]
        return messagebox.askyesno(t("confirm_title"), "\n".join(lines), parent=self.r)

    # ------------------------------------------------------------- installs / downloads (worker)
    def _install_engine(self):
        self.status("status_install")
        self.log(t("log_engine_install", path=self.paths.runtime))
        info = core.install_engine(self.paths, self.hw, self.log, t, self.bytes_progress, self.abort)
        self.engine_info = info
        self.hw.cuda_usable = bool(info.get("cuda_ok"))
        self.log(t("log_engine_done", torch=info.get("torch", "?"), mode="CUDA" if self.hw.cuda_usable else "CPU"))
        if self.hw.vendor == "nvidia" and not self.hw.cuda_usable:
            self.log(t("log_cuda_fail"))
        self.post(self.refresh_lists)

    def _download_image_model(self, m):
        self.status("status_download")
        self.log(t("log_download_start", name=m["name"], size=f"~{m['size_gb']:.1f} GB",
                   path=core.image_model_dir(self.paths.models, m)))
        core.download_image_model(self.paths.models, m, self.bytes_progress, self.abort)
        self.log(t("log_download_done", name=m["name"]))
        self.progress(100)
        self.post(self.refresh_lists)

    def _ensure_upscaler(self):
        if core.is_upscaler_installed(self.paths.models):
            return
        u = core.UPSCALER
        dest = core.upscaler_path(self.paths.models)
        self.status("status_download")
        self.log(t("log_download_start", name=u["name"], size=u["size"], path=dest))
        core.download_file(u["url"], dest, self.bytes_progress, self.abort, need_gb=u["size_gb"])
        self.log(t("log_download_done", name=u["name"]))

    def _ensure_peft(self, plan):
        """LoRA yükleme/eğitim için PEFT. diffusers PEFT'ten önce yüklendiyse yeniden başlatma gerekir."""
        if not core.peft_installed(self.paths):
            if not plan.get("peft") and not plan.get("engine"):
                return False
            if not core.peft_installed(self.paths):
                self.status("status_install")
                core.install_peft(self.paths, self.log, t, self.bytes_progress, self.abort)
                self.log(t("log_peft_done"))
        if not core.peft_ready_in_process():
            self.log(t("log_peft_restart"))
            self.post(self._peft_restart_prompt)
            return False
        return True

    def _peft_restart_prompt(self):
        if messagebox.askyesno(t("restart_title"), t("peft_restart_q"), parent=self.r):
            self.restart_program()

    def _ensure_captioner(self):
        if core.is_captioner_installed(self.paths.models):
            return
        c = core.CAPTIONER
        self.status("status_download")
        self.log(t("log_download_start", name=c["name"], size=c["size"], path=core.captioner_dir(self.paths.models)))
        core.download_captioner(self.paths.models, self.bytes_progress, self.abort)
        self.log(t("log_download_done", name=c["name"]))
        self.post(self.refresh_lists)

    def captioner_ok(self, parent=None):
        """Kurulu motor Florence-2'yi tanıyor mu? (Motor henüz kurulmadıysa yeni kurulum en güncel sürümü getirir.)"""
        if not core.engine_status(self.paths):
            return True
        if os.path.isdir(os.path.join(self.paths.site_packages(), "transformers", "models", "florence2")):
            return True
        messagebox.showinfo(APP_NAME, t("cap_unsupported"), parent=parent or self.r)
        return False

    def free_vram_for_training(self):
        """Eğitim/okuma öncesi: görsel motoru, büyütücü ve yerel LLM ekran kartı belleğinden çıkarılır."""
        self.engine.unload()
        core.unload_upscaler()
        try:
            lc = self.client("local")
            if lc.alive(timeout=0.5):
                lc.unload_all()
        except Exception:
            pass
        core.free_gpu_memory()

    # ------------------------------------------------------------- LoRA seçimi
    def open_lora_picker(self):
        if self.lora_win is not None and self.lora_win.alive():
            self.lora_win.w.lift()
            return
        self.lora_win = LoraPickerWindow(self)

    def _lora_plan(self, m):
        """Seçili LoRA'lar → (uygulanacaklar [(kimlik, yol, güç, öğe)], stil kartları, atlananlar [(ad, neden)])."""
        items = {i["id"]: i for i in core.scan_loras(self.paths)}
        specs, cards, skipped = [], [], []
        for sel in self.st.get("loras", []):
            it = items.get(sel["id"])
            if not it:
                skipped.append((sel["id"].split(":", 1)[-1], "missing"))
                continue
            if it["kind"] == "card":
                cards.append(it)
                continue
            if core.lora_compat(it["base"], m["key"]) == "no":
                skipped.append((it["name"], "incompatible"))
                continue
            specs.append((it["id"], it["path"], float(sel.get("w", 0.8)), it))
        if self.v_pref_auto.get() and m["key"] == "sd15":
            pref = os.path.join(core.own_lora_dir(self.paths), core.PREF_LORA_NAME + ".safetensors")
            if os.path.exists(pref) and not any(sp[1] == pref for sp in specs):
                it = items.get("own:" + core.PREF_LORA_NAME + ".safetensors") or \
                    {"name": "AILS tercih", "origin": "own", "trigger": "", "kind": "lora"}
                specs.append(("own:" + core.PREF_LORA_NAME + ".safetensors", pref,
                              float(self.v_pref_weight.get()), it))
        return specs, cards, skipped

    def update_lora_label(self):
        if not hasattr(self, "lbl_loras"):
            return
        sel = self.st.get("loras", [])
        if not sel:
            self.lbl_loras.config(text=t("lora_none"), fg="#888888")
            return
        items = {i["id"]: i for i in core.scan_loras(self.paths)}
        m = self.model()
        parts, bad = [], False
        for x in sel:
            it = items.get(x["id"])
            if not it:
                parts.append("⚠ " + x["id"].split(":", 1)[-1])
                bad = True
                continue
            if it["kind"] == "card":
                parts.append("📝 " + it["name"])
                continue
            comp = core.lora_compat(it["base"], m["key"])
            mark = "⚠ " if comp == "no" else (core.learn_icon(it["learn"]) + " " if it.get("learn") else "")
            bad = bad or comp == "no"
            parts.append(f"{mark}{it['name']} ({x['w']:.2f})")
        self.lbl_loras.config(text="🧩 " + ", ".join(parts) + ("   " + t("lora_incompat_note") if bad else ""),
                              fg=WARN if bad else BLUE)

    # ------------------------------------------------------------- ödül / ceza (👍 / 👎)
    def on_rate(self, r):
        it = self.selected_item()
        if not it:
            messagebox.showinfo(APP_NAME, t("no_image"), parent=self.r)
            return
        key = os.path.abspath(it["path"])
        cur = self.ratings.get(key, {}).get("rating", 0)
        new = 0 if cur == r else r
        rec = core.add_feedback(self.paths, it["path"], it.get("meta") or {}, new)
        if new:
            self.ratings[key] = rec
        else:
            self.ratings.pop(key, None)
        self.log(t("log_rated_up" if new > 0 else "log_rated_down" if new < 0 else "log_rated_clear"))
        self._update_rate_buttons()
        self._refresh_caption(it)
        if self.style_win is not None and self.style_win.alive():
            self.style_win.update_state()

    def _rating_of(self, it):
        return self.ratings.get(os.path.abspath(it["path"]), {}).get("rating", 0) if it else 0

    def _update_rate_buttons(self):
        if not hasattr(self, "b_like"):
            return
        r = self._rating_of(self.selected_item())
        self.b_like.config(text="👍✓" if r > 0 else "👍")
        self.b_dislike.config(text="👎✓" if r < 0 else "👎")

    def _caption_text(self, path, meta):
        if meta and meta.get("stage") == "sample":
            base = t("gal_sample", steps=meta.get("steps", ""))
        elif meta:
            base = f"{meta.get('width', '')}×{meta.get('height', '')}"
        else:
            base = os.path.basename(path)[:18]
        r = self.ratings.get(os.path.abspath(path), {}).get("rating", 0)
        return base + (" 👍" if r > 0 else " 👎" if r < 0 else "")

    def _refresh_caption(self, it):
        try:
            it["cap"].config(text=self._caption_text(it["path"], it.get("meta")))
        except Exception:
            pass

    def _ensure_detailer(self):
        if core.is_detailer_installed(self.paths):
            return
        self.status("status_download")
        self.log(t("log_det_install", path=core.detector_dir(self.paths.models)))
        core.install_detector(self.paths, self.log, t, self.bytes_progress, self.abort)
        self.log(t("log_download_done", name=t("detailer_name")))
        self.post(self.refresh_lists)

    def _ensure_local_llm(self, name, plan):
        """Yerel Ollama: gerekirse (onaylıysa) kur / başlat / model konumunu yönlendir / modeli indir."""
        c = self.client("local")
        if not c.alive(timeout=1.0, force=True):
            if not core.ollama_installed(self.paths):
                if IS_WIN and plan.get("ollama_install"):
                    self.status("status_download")
                    self.log(t("log_ollama_install", path=self.paths.ollama_dir))
                    ok = core.install_ollama_windows(self.paths, self.log, self.bytes_progress, self.abort)
                    self.log(t("log_ollama_env_set", path=self.paths.ollama_models))
                    if not ok:
                        self.log(t("log_ollama_fail"))
                        return None
                else:
                    self.log(t("log_ollama_fail") if IS_WIN else t("log_ollama_manual"))
                    return None
            else:
                if plan.get("ollama_redirect"):
                    os.makedirs(self.paths.ollama_models, exist_ok=True)
                    core.set_user_env_var("OLLAMA_MODELS", self.paths.ollama_models)
                    self.log(t("log_ollama_env_set", path=self.paths.ollama_models))
                self.log(t("log_ollama_start"))
                if not core.start_local_ollama(self.paths, c):
                    self.log(t("log_ollama_fail"))
                    return None
        elif plan.get("ollama_redirect"):
            os.makedirs(self.paths.ollama_models, exist_ok=True)
            core.set_user_env_var("OLLAMA_MODELS", self.paths.ollama_models)
            self.log(t("log_ollama_env_set", path=self.paths.ollama_models))
            self.log(t("log_ollama_restart"))
            if not core.restart_local_ollama(self.paths, c):
                self.log(t("log_ollama_fail"))
                return None
        self.llm_loc = core.ollama_models_dir()
        names = c.installed_models()
        self.llm_installed["local"] = names
        if not core.model_in_list(name, names):
            if plan.get("llm_pull") != name:
                return None           # kullanıcı bu indirmeyi onaylamadı
            lm = core.LLM_MODEL_BY_KEY.get(name, {"size_gb": 3.0})
            core.check_space(self.llm_loc[0], lm["size_gb"] + 1.0)
            self.status("status_download")
            self.log(t("log_llm_pull", name=name))
            c.pull(name, self.bytes_progress, self.abort)
            self.llm_installed["local"] = c.installed_models()
            self.log(t("log_download_done", name=name))
        return c

    def _ensure_remote_llm(self, host, name):
        """Ağdaki Ollama: ulaşılamazsa yerel Ollama'ya düş; model yoksa o bilgisayara indirmek için sor."""
        c = self.client(host)
        if not c.alive(timeout=3.0):
            self.log(t("remote_fail", host=host))
            lc = self.client("local")
            if lc.alive(timeout=0.8) and core.model_in_list(name, lc.installed_models()):
                self.log(t("remote_fallback"))
                return lc
            return None
        names = c.installed_models()
        self.llm_installed[host] = names
        if not core.model_in_list(name, names):
            lm = core.LLM_MODEL_BY_KEY.get(name, {"name": name, "size": "?"})
            ok = self.call_ui(lambda: messagebox.askyesno(APP_NAME, t("confirm_remote_pull", name=lm["name"], host=host,
                                                                     size=lm["size"]), parent=self.r))
            if not ok:
                return None
            self.status("status_download")
            self.log(t("log_llm_pull", name=name))
            c.pull(name, self.bytes_progress, self.abort)
            self.llm_installed[host] = c.installed_models()
            self.log(t("log_download_done", name=name))
        return c

    # ------------------------------------------------------------- download / delete buttons
    def on_download(self):
        item = self.v_manage.get()
        if not item:
            return
        if item == "engine":
            if core.engine_status(self.paths):
                if not messagebox.askyesno(APP_NAME, t("engine_reinstall_q"), parent=self.r):
                    return
                if "torch" in sys.modules:        # motor bu süreçte yüklü: dosyaları kilitli, yeniden başlatmada onar
                    self._schedule_runtime_reset()
                    return
                core.remove_engine_marker(self.paths)
                plan = {"engine": True}
            else:
                plan = self._plan(engine=True)
        elif item.startswith("img:"):
            m = core.IMAGE_MODEL_BY_KEY[item[4:]]
            if core.is_image_model_installed(self.paths.models, m):
                messagebox.showinfo(APP_NAME, t("already_installed", name=m["name"]), parent=self.r)
                return
            plan = self._plan(model=m)
        elif item.startswith("up:"):
            if core.is_upscaler_installed(self.paths.models):
                messagebox.showinfo(APP_NAME, t("already_installed", name=core.UPSCALER["name"]), parent=self.r)
                return
            plan = self._plan(upscaler=True)
        elif item.startswith("cap:"):
            if core.is_captioner_installed(self.paths.models):
                messagebox.showinfo(APP_NAME, t("already_installed", name=core.CAPTIONER["name"]), parent=self.r)
                return
            if not self.captioner_ok():
                return
            plan = self._plan(engine=True, captioner=True)
        elif item.startswith("det:"):
            if core.is_detailer_installed(self.paths):
                messagebox.showinfo(APP_NAME, t("already_installed", name=t("detailer_name")), parent=self.r)
                return
            plan = self._plan(engine=True, detailer=True)
        else:
            name = item[4:]
            if self.v_llm_host.get() == "local":
                plan = self._plan(llm=name)
                if plan is not None and not plan and core.model_in_list(name, self.llm_installed.get("local", [])):
                    messagebox.showinfo(APP_NAME, t("already_installed", name=name), parent=self.r)
                    return
            else:
                plan = {}
        if plan is None:
            return
        self.start_job("download", self.job_download, item, plan)

    def job_download(self, item, plan):
        if item == "engine":
            self._install_engine()
        elif item.startswith("img:"):
            self._download_image_model(core.IMAGE_MODEL_BY_KEY[item[4:]])
        elif item.startswith("up:"):
            self._ensure_upscaler()
        elif item.startswith("det:"):
            if not core.engine_status(self.paths):
                if not plan.get("engine"):
                    raise JobFail()
                self._install_engine()
            self._ensure_detailer()
        elif item.startswith("cap:"):
            if not core.engine_status(self.paths):
                if not plan.get("engine"):
                    raise JobFail()
                self._install_engine()
            if plan.get("captioner"):
                self._ensure_captioner()
        else:
            name, host = item[4:], self.v_llm_host.get()
            if host == "local":
                c = self._ensure_local_llm(name, plan)
            else:
                c = self._ensure_remote_llm(host, name)
            if c is None:
                raise JobFail()
        self.status("status_done")

    def on_delete(self):
        item = self.v_manage.get()
        if not item or self.running():
            return
        if item == "engine":
            if not core.engine_status(self.paths) and not os.path.isdir(self.paths.venv):
                return
            if messagebox.askyesno(APP_NAME, t("confirm_delete_engine"), parent=self.r):
                self._schedule_runtime_reset()
            return
        if item.startswith("llm:"):
            name, host = item[4:], self.v_llm_host.get()
            if host == "local":
                if not messagebox.askyesno(APP_NAME, t("confirm_delete", name=name), parent=self.r):
                    return
            elif not messagebox.askyesno(APP_NAME, t("confirm_remote_delete", name=name, host=host), parent=self.r):
                return
            self.start_job("delete", self.job_delete_llm, host, name)
            return
        if item.startswith("img:"):
            m = core.IMAGE_MODEL_BY_KEY[item[4:]]
            name, path = m["name"], core.image_model_dir(self.paths.models, m)
        elif item.startswith("det:"):
            name, path = t("detailer_name"), core.detector_dir(self.paths.models)
        elif item.startswith("cap:"):
            name, path = core.CAPTIONER["name"], core.captioner_dir(self.paths.models)
        else:
            name, path = core.UPSCALER["name"], core.upscaler_path(self.paths.models)
        if not os.path.exists(path):
            return
        if not messagebox.askyesno(APP_NAME, t("confirm_delete", name=name), parent=self.r):
            return
        try:
            if item.startswith("img:") and self.engine.key == item[4:]:
                self.engine.unload()
            if item.startswith("up:"):
                core.unload_upscaler()
            if item.startswith("det:"):
                core.remove_detector(self.paths)
            core.delete_path(path)
            self.log(t("log_deleted", name=name))
        except Exception as e:
            self.log(friendly_error(e))
        self.refresh_lists()

    def job_delete_llm(self, host, name):
        c = self.client(host)
        if host == "local" and not c.alive(timeout=1.0, force=True):
            if not core.start_local_ollama(self.paths, c):
                self.log(t("log_ollama_fail"))
                raise JobFail()
        c.delete(name)
        self.llm_installed[host] = c.installed_models()
        self.log(t("log_deleted", name=name))

    def _schedule_runtime_reset(self):
        cfg = load_pointer()
        cfg["pending_runtime_reset"] = True
        save_pointer(cfg)
        if messagebox.askyesno(t("restart_title"), t("restart_q"), parent=self.r):
            self.restart_program()

    # ------------------------------------------------------------- generation
    def on_start(self):
        if self.running():
            self.abort.set()
            return
        text = self.txt_prompt.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo(APP_NAME, t("no_prompt"), parent=self.r)
            return
        m = self.model()
        want_fix = self.v_detailer.get() and (self.v_fix_faces.get() or self.v_fix_hands.get())
        specs, cards, skipped = self._lora_plan(m)
        plan = self._plan(engine=True, model=m, llm=self.v_llm.get() if self.v_use_llm.get() else None,
                          detailer=want_fix, peft=bool(specs))
        if plan is None:
            return
        self.save()
        try:
            seed = int(self.v_seed.get())
        except ValueError:
            seed = 12345
        params = {
            "text": text, "extra": self.txt_extra.get("1.0", "end").strip(), "model": m["key"],
            "llm": self.v_llm.get(), "llm_host": self.v_llm_host.get(), "use_llm": self.v_use_llm.get(),
            "show_prompt": self.v_show_prompt.get(), "aspect": self.v_aspect.get(), "orient": self.v_orient.get(),
            "res": self.v_res.get(), "quality": self.v_quality.get(), "creativity": self.v_creat.get(),
            "creativity_auto": self.v_creat_auto.get(), "seed_random": self.v_seed_random.get(), "seed": seed,
            "count": int(self.v_count.get() or 1), "outdir": self.v_outdir.get(), "plan": plan,
            "steps_manual": self._steps_manual(), "sampler": self.v_sampler.get(), "hires": self.v_hires.get(),
            "detailer": want_fix, "fix_faces": self.v_fix_faces.get(), "fix_hands": self.v_fix_hands.get(),
            "auto_neg": self.v_auto_neg.get(), "save_base": self.v_save_base.get(),
            "lora_specs": specs, "cards": cards, "lora_skipped": skipped,
        }
        self.start_job("generate", self.job_generate, params)

    def _prepare_prompt(self, p, m):
        """LLM katmanı: çeviri + pozitif/negatif + otomatik ayarlar. Sonra yereldeki LLM'i VRAM'den boşaltır."""
        auto_fields = set()
        if p["aspect"] == "auto":
            auto_fields.add("aspect")
        if p["orient"] == "auto":
            auto_fields.add("orientation")
        if p["creativity_auto"] and m["cfg"]:
            auto_fields.add("creativity")
        raw_res = {"positive": p["text"], "negative": p["extra"] if m["neg"] else ""}
        res = raw_res
        if p["use_llm"]:
            self.status("status_llm")
            try:
                if p["llm_host"] == "local":
                    c = self._ensure_local_llm(p["llm"], p["plan"])
                else:
                    c = self._ensure_remote_llm(p["llm_host"], p["llm"])
                if c is not None:
                    if c.is_local:
                        self.engine.park()
                    system, user, schema = core.build_llm_request(m, p["text"], p["extra"], auto_fields)
                    self.status("status_llm")
                    raw = c.chat_json(p["llm"], system, user, schema)
                    res = core.sanitize_llm_result(raw, auto_fields)
                    if not res["positive"]:
                        raise ValueError("empty prompt")
                    self.log(t("log_llm_prompt", p=res["positive"]))
                    if m["neg"] and res["negative"]:
                        self.log(t("log_llm_neg", n=res["negative"]))
                    chosen = [f"{k}={res[k]}" for k in ("aspect", "orientation", "creativity") if k in res]
                    if chosen:
                        self.log(t("log_llm_settings", s=", ".join(chosen)))
                else:
                    self.log(t("log_ollama_fail"))
            except core.AbortedError:
                raise
            except Exception as e:
                self._write_file_log(traceback.format_exc())
                self.log(t("log_llm_fail", err=friendly_error(e)))
                res = raw_res
        # Yerel VRAM'de LLM bırakma (ağdaki sunuculara asla dokunulmaz).
        lc = self.client("local")
        if lc.alive(timeout=0.5):
            names = lc.unload_all()
            if names:
                self.log(t("log_llm_unloaded", names=", ".join(names)))
        aspect = p["aspect"] if p["aspect"] != "auto" else res.get("aspect", "1:1")
        orient = p["orient"] if p["orient"] != "auto" else res.get("orientation", "landscape")
        creat = p["creativity"] if not p["creativity_auto"] else res.get("creativity", 5)
        return res["positive"], res["negative"], aspect, orient, creat

    def _preview_dialog(self, pos, neg, m):
        dlg = tk.Toplevel(self.r)
        dlg.title(t("preview_title"))
        dlg.configure(bg=CARD_BG)
        dlg.geometry("720x460")
        dlg.transient(self.r)
        dlg.grab_set()
        result = {"ok": False}
        tk.Label(dlg, text=t("preview_pos"), bg=CARD_BG, fg=BLUE, font=self.f.b10, anchor="w").pack(fill="x", padx=12, pady=(10, 2))
        tp = tk.Text(dlg, height=8, font=self.f.n10, bg=FRM, fg=TXT, insertbackground=TXT, wrap="word")
        tp.insert("1.0", pos)
        tp.pack(fill="both", expand=True, padx=12)
        tn = None
        if m["neg"]:
            tk.Label(dlg, text=t("preview_neg"), bg=CARD_BG, fg=BLUE, font=self.f.b10, anchor="w").pack(fill="x", padx=12, pady=(8, 2))
            tn = tk.Text(dlg, height=4, font=self.f.n10, bg=FRM, fg=TXT, insertbackground=TXT, wrap="word")
            tn.insert("1.0", neg)
            tn.pack(fill="both", expand=True, padx=12)
        else:
            tk.Label(dlg, text=t("preview_neg_na"), bg=CARD_BG, fg="#888888", font=self.f.n9, anchor="w").pack(fill="x", padx=12, pady=(8, 2))
        btns = tk.Frame(dlg, bg=CARD_BG)
        btns.pack(fill="x", padx=12, pady=10)

        def ok():
            result.update(ok=True, pos=tp.get("1.0", "end").strip(), neg=tn.get("1.0", "end").strip() if tn else "")
            dlg.destroy()

        tk.Button(btns, text=t("preview_ok"), bg=ORANGE, fg="white", font=self.f.b11, relief="flat", command=ok,
                  cursor="hand2").pack(side="left", fill="x", expand=True, ipady=5)
        tk.Button(btns, text=t("preview_cancel"), bg=RED, fg="white", font=self.f.b11, relief="flat",
                  command=dlg.destroy, cursor="hand2").pack(side="left", padx=(8, 0), ipady=5, ipadx=12)
        dlg.protocol("WM_DELETE_WINDOW", dlg.destroy)
        self.r.wait_window(dlg)
        return result

    def job_generate(self, p):
        m = core.IMAGE_MODEL_BY_KEY[p["model"]]
        plan = p["plan"]
        if not core.engine_status(self.paths):
            if not plan.get("engine"):
                raise JobFail()
            self._install_engine()
        if not core.is_image_model_installed(self.paths.models, m):
            if not plan.get("model"):
                raise JobFail()
            self._download_image_model(m)
        if p["detailer"] and plan.get("detailer") and not core.is_detailer_installed(self.paths):
            try:
                self._ensure_detailer()
            except (core.AbortedError, core.DiskFullError):
                raise
            except Exception as e:          # düzeltici kurulamazsa görsel yine de üretilir
                self._write_file_log(traceback.format_exc())
                self.log(t("log_det_unavailable", err=friendly_error(e)))

        specs = list(p.get("lora_specs") or [])
        for name, why in p.get("lora_skipped") or []:
            self.log(t("log_lora_skip_" + why, name=name))
        if specs and not self._ensure_peft(plan):
            self.log(t("log_lora_off"))
            specs = []
        pos, neg, aspect, orient, creat = self._prepare_prompt(p, m)
        lead = [core.lora_trigger_text(sp[3]) for sp in specs] + [c.get("style", "") for c in p.get("cards") or []]
        lead = [x for x in lead if x]
        if lead:
            pos = core.merge_prompt_lists(", ".join(lead), pos)
        if m["neg"]:
            avoid = ", ".join(c.get("avoid", "") for c in p.get("cards") or [] if c.get("avoid"))
            if avoid:
                neg = core.merge_prompt_lists(neg, avoid)
        if m["neg"] and p["auto_neg"]:
            neg = core.merge_prompt_lists(neg, core.ANATOMY_NEGATIVE)
        if p["show_prompt"]:
            r = self.call_ui(self._preview_dialog, pos, neg, m)
            if not r or not r.get("ok"):
                raise core.AbortedError("cancel")
            pos, neg = r["pos"], r["neg"]

        res = p["res"] if p["res"] != "auto" else "normal"
        quality = p["quality"] if p["quality"] != "auto" else "balanced"
        w, h = core.compute_size(m["base_px"], aspect, orient, res)
        steps = core.resolve_steps(m, quality, p["steps_manual"])
        cfg = core.creativity_to_cfg(creat) if m["cfg"] else m["fixed_cfg"]
        profile = self.effective_profile(m)
        path = core.image_model_dir(self.paths.models, m)
        os.makedirs(p["outdir"], exist_ok=True)
        base_seed = random.randint(0, 2 ** 31 - 2) if p["seed_random"] else p["seed"]
        n = p["count"]
        do_hires = core.hires_enabled(p["hires"], m, quality, profile)
        do_fix = bool(p["detailer"]) and core.is_detailer_installed(self.paths)
        if p["detailer"] and not do_fix:
            self.log(t("log_det_skip"))
        sampler_txt = core.SAMPLER_NAMES.get("dpmpp_2m_karras" if p["sampler"] == "auto" else p["sampler"], "") \
            if m.get("samplers") else t("sampler_model")
        self.log(t("log_profile", profile=t("prof_" + profile), desc=t("prof_desc_" + profile)))
        self.log(t("log_quality", q=t("q_" + quality), steps=steps, sampler=sampler_txt,
                   hires=t("on") if do_hires else t("off"), fix=t("on") if do_fix else t("off")))
        if m.get("turbo") and steps > m["turbo"] * 2:
            self.log(t("log_turbo_steps", n=m["turbo"]))

        # İlerleme çubuğu: her görselin payı ana üretim + (varsa) ikinci geçiş + (varsa) düzeltme olarak bölünür.
        weights = [("base", 1.0)] + ([("hires", 0.7)] if do_hires else []) + ([("fix", 0.7)] if do_fix else [])
        total_w = sum(x for _k, x in weights)
        offsets, acc = {}, 0.0
        for key, wt in weights:
            offsets[key] = (acc / total_w, wt / total_w)
            acc += wt

        def make_cb(i, phase, k=0, K=1):
            off, wid = offsets[phase]

            def cb(s, S):
                frac = off + wid * ((k + s / float(max(1, S))) / max(1, K))
                self.progress((i + frac) * 100.0 / n, t("counter", i=i + 1, n=n, s=s, S=S))
            return cb

        for i in range(n):
            seed = (base_seed + i) % (2 ** 31 - 1)
            while True:
                wd = core.RamWatchdog(self.abort, self.hw.ram_total_gb)
                try:
                    with wd:
                        if self.engine.key != m["key"] or self.engine.profile != profile:
                            self.status("status_loading")
                            self.post(self.set_indeterminate, "")
                            self.log(t("log_loading", name=m["name"]))
                            t0 = time.time()
                            self.engine.load(m, path, profile, self.hw)
                            self.log(t("log_loaded", sec=time.time() - t0))
                        else:
                            self.engine.unpark()
                        if specs or self.engine._loras:
                            applied, lerrs = self.engine.set_loras([(sp[0], sp[1], sp[2]) for sp in specs])
                            if i == 0:
                                names = {sp[0]: sp[3].get("name", sp[0]) for sp in specs}
                                if applied:
                                    self.log(t("log_lora_applied", names=", ".join(f"{names[k]} ({w:.2f})"
                                                                               for k, w in applied)))
                                for k, err in lerrs:
                                    self.log(t("log_lora_fail", name=names.get(k, k), err=err))
                            lora_used = [{"name": next((sp[3].get("name") for sp in specs if sp[0] == k), k),
                                          "w": round(w, 2)} for k, w in applied]
                        else:
                            lora_used = []
                        self.status("status_gen")
                        self.log(t("log_gen_start", i=i + 1, n=n, w=w, h=h, steps=steps, seed=seed))
                        t0 = time.time()
                        img = self.engine.generate(m, pos, neg, w, h, steps, cfg, seed, make_cb(i, "base"),
                                                   self.abort, sampler=p["sampler"])
                    break
                except core.AbortedError:
                    if wd.triggered:
                        self.engine.unload()
                        self.log(t("log_ram_guard", free=wd.free_at_trigger))
                        raise core.AbortedError("ram")
                    raise
                except core.MemoryGuardError as e:
                    need, avail = e.args[0]
                    self.log(t("err_ram", need=need, avail=avail))
                    raise JobFail()
                except core.EngineBrokenError:
                    raise
                except Exception as e:
                    if not core.is_oom_error(e):
                        raise
                    self.engine.unload()
                    nxt = core.lower_profile(profile)
                    if nxt is None:
                        self.log(t("log_oom_final"))
                        raise JobFail()
                    profile = nxt
                    self.log(t("log_oom", profile=t("prof_" + profile)))
                    do_hires = do_hires and core.hires_enabled(p["hires"], m, quality, profile)

            stage = {"img": img, "hires": None, "fix": None, "loras": lora_used,
                     "cards": [c.get("name", "") for c in p.get("cards") or []]}
            meta = self._gen_meta(m, p, pos, neg, seed, steps, cfg, w, h, aspect, orient, profile, quality, stage)
            if p["save_base"] and (do_hires or do_fix):
                bpath = self._save_image(img, p["outdir"], m, seed, dict(meta, stage="base"), suffix="_base")
                self.post(self.add_to_gallery, bpath, dict(meta, stage="base"), False)
            try:
                if do_hires or do_fix:
                    self._run_passes(p, m, stage, pos, neg, steps, cfg, seed, do_hires, do_fix, make_cb, i)
            except core.AbortedError:
                # Durdurulsa bile bitmiş son hâl kaybolmasın.
                meta = self._gen_meta(m, p, pos, neg, seed, steps, cfg, w, h, aspect, orient, profile, quality, stage)
                out_path = self._save_image(stage["img"], p["outdir"], m, seed, meta)
                self.log(t("log_saved_partial", path=out_path))
                self.post(self.add_to_gallery, out_path, meta, True)
                raise
            meta = self._gen_meta(m, p, pos, neg, seed, steps, cfg, w, h, aspect, orient, profile, quality, stage)
            out_path = self._save_image(stage["img"], p["outdir"], m, seed, meta)
            self.log(t("log_saved", path=out_path, sec=time.time() - t0))
            self.post(self.add_to_gallery, out_path, meta, True)
        self.progress(100)
        self.status("status_done")

    def _gen_meta(self, m, p, pos, neg, seed, steps, cfg, w, h, aspect, orient, profile, quality, stage):
        img = stage["img"]
        meta = {"app": f"{APP_NAME} {APP_VERSION}", "model": m["name"], "model_key": m["key"], "prompt": pos,
                "negative": neg if m["neg"] else "", "user_text": p["text"], "extra": p["extra"], "seed": seed,
                "steps": steps, "cfg": cfg, "width": img.width, "height": img.height, "base_size": f"{w}x{h}",
                "aspect": aspect, "orient": orient, "profile": profile, "quality": quality,
                "sampler": core.SAMPLER_NAMES.get(self.engine.sampler, "") if m.get("samplers") else ""}
        if stage.get("hires"):
            meta["hires"] = "%dx%d" % tuple(stage["hires"])
            meta["hires_strength"] = m.get("hires_strength")
        if stage.get("fix") is not None:
            meta["fix"] = dict(stage["fix"])
        if stage.get("loras"):
            meta["loras"] = list(stage["loras"])
        if stage.get("cards"):
            meta["cards"] = list(stage["cards"])
        return meta

    def _run_passes(self, p, m, stage, pos, neg, steps, cfg, seed, do_hires, do_fix, make_cb, i):
        """İkinci geçiş + yüz/el düzeltme. Bellek ya da başka bir sorun olursa o adım atlanır, görsel korunur."""
        if not self.engine.supports_refine(m):
            self.log(t("log_pass_unsupported", name=m["name"]))
            return
        wd = core.RamWatchdog(self.abort, self.hw.ram_total_gb)
        try:
            with wd:
                if do_hires:
                    size = core.hires_size(stage["img"].width, stage["img"].height, m)
                    if size:
                        st = m.get("hires_strength", 0.45)
                        self.status("status_hires")
                        self.log(t("log_hires", w=size[0], h=size[1], s=st, steps=core.refine_steps(m, steps, st)))
                        out, size = core.run_hires(self.engine, m, stage["img"], pos, neg, steps, cfg, seed,
                                                   make_cb(i, "hires"), self.abort)
                        stage["img"], stage["hires"] = out, size
                if do_fix:
                    self.status("status_fix_detect")
                    # Yavaş profillerde (düşük VRAM / CPU) en büyük 2 yüz + 2 el; diğerlerinde 4 + 4.
                    cap = 2 if self.engine.profile in ("low", "cpu") else 4
                    regions = core.detect_regions(self.paths.models, stage["img"], m["base_px"],
                                                  faces=p["fix_faces"], hands=p["fix_hands"], max_each=cap)
                    nf = sum(1 for r in regions if r["kind"] == "face")
                    self.log(t("log_fix_found", faces=nf, hands=len(regions) - nf))
                    if regions:
                        cur = {"k": 0, "K": len(regions)}

                        def rcb(k, K, kind):
                            cur["k"] = k - 1
                            self.status("status_fix", k=k, K=K, part=t("part_" + kind))

                        def scb(s, S):
                            make_cb(i, "fix", cur["k"], cur["K"])(s, S)

                        out, counts = core.run_detailer(self.engine, m, stage["img"], self.paths.models, pos, neg,
                                                        steps, cfg, seed, region_cb=rcb, step_cb=scb,
                                                        abort_event=self.abort, regions=regions)
                        stage["img"], stage["fix"] = out, counts
                    else:
                        stage["fix"] = {"face": 0, "hand": 0}
        except core.AbortedError:
            if wd.triggered:
                self.engine.unload()
                self.log(t("log_ram_guard", free=wd.free_at_trigger))
                raise core.AbortedError("ram")
            raise
        except core.EngineBrokenError:
            raise
        except Exception as e:
            self._write_file_log(traceback.format_exc())
            if core.is_oom_error(e):
                core.free_gpu_memory()
                self.log(t("log_pass_oom"))
            else:
                self.log(t("log_pass_fail", err=friendly_error(e)))

    def _save_image(self, img, outdir, m, seed, meta, suffix=""):
        from PIL.PngImagePlugin import PngInfo
        info = PngInfo()
        a1111 = f"{meta['prompt']}\n"
        if meta.get("negative"):
            a1111 += f"Negative prompt: {meta['negative']}\n"
        a1111 += f"Steps: {meta['steps']}, "
        if meta.get("sampler"):
            a1111 += f"Sampler: {meta['sampler']}, "
        a1111 += f"CFG scale: {meta['cfg']}, Seed: {seed}, Size: {meta['width']}x{meta['height']}, Model: {m['name']}"
        if meta.get("hires"):
            a1111 += f", Hires resize: {meta['hires']}, Denoising strength: {meta.get('hires_strength')}"
        if meta.get("fix"):
            a1111 += f", Face/hand fix: faces {meta['fix'].get('face', 0)} hands {meta['fix'].get('hand', 0)}"
        if meta.get("loras"):
            a1111 += ", LoRA: " + ", ".join(f"{x['name']}:{x['w']}" for x in meta["loras"])
        info.add_text("parameters", a1111)
        info.add_text("ails", json.dumps(meta, ensure_ascii=False))
        name = time.strftime("%Y%m%d_%H%M%S") + f"_{m['key']}_{seed}{suffix}.png"
        path = os.path.join(outdir, name)
        core.check_space(outdir, 0.05)
        img.save(path, pnginfo=info)
        return path

    # ------------------------------------------------------------- upscale
    def on_upscale(self):
        it = self.selected_item()
        if not it:
            messagebox.showinfo(APP_NAME, t("no_image"), parent=self.r)
            return
        plan = self._plan(engine=True, upscaler=True)
        if plan is None:
            return
        self.start_job("upscale", self.job_upscale, it["path"], dict(it.get("meta") or {}), plan)

    def job_upscale(self, src, meta, plan):
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo
        if not core.engine_status(self.paths):
            if not plan.get("engine"):
                raise JobFail()
            self._install_engine()
        self._ensure_upscaler()
        self.status("status_upscale")
        self.engine.park()
        core.free_gpu_memory()
        device = self.hw.device if (self.hw.cuda_usable and self.hw.vram_gb >= 1.5) else "cpu"
        tile = 256 if self.hw.vram_gb >= 3 else 160
        img = Image.open(src)
        img.load()
        t0 = time.time()

        def cb(a, b):
            self.progress(a * 100.0 / b, f"{a}/{b}")

        try:
            try:
                net = core.load_upscaler(core.upscaler_path(self.paths.models), device)
                out = core.upscale_2x(img, net, device, tile=tile, progress_cb=cb, abort_event=self.abort)
            except core.EngineBrokenError:
                raise
            except Exception as e:
                if device == "cpu" or not core.is_oom_error(e):
                    raise
                core.unload_upscaler()
                self.log(t("log_oom", profile=t("prof_cpu")))
                device = "cpu"
                net = core.load_upscaler(core.upscaler_path(self.paths.models), device)
                out = core.upscale_2x(img, net, device, tile=tile, progress_cb=cb, abort_event=self.abort)
        finally:
            core.unload_upscaler()
        meta.update(width=out.width, height=out.height, upscaled="Real-ESRGAN x2plus")
        info = PngInfo()
        info.add_text("ails", json.dumps(meta, ensure_ascii=False))
        base, _ext = os.path.splitext(src)
        dst = base + "_2x.png"
        core.check_space(dst, 0.1)
        out.save(dst, pnginfo=info)
        self.log(t("log_upscale_done", path=dst, sec=time.time() - t0))
        self.post(self.add_to_gallery, dst, meta, True)
        self.progress(100)
        self.status("status_done")

    # ------------------------------------------------------------- gallery
    def _load_existing_gallery(self):
        d = self.v_outdir.get()
        if not os.path.isdir(d):
            return
        try:
            files = sorted((os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".png")),
                           key=os.path.getmtime)[-30:]
        except Exception:
            return
        for fpath in files:
            meta = {}
            try:
                from PIL import Image
                with Image.open(fpath) as im:
                    meta = json.loads(im.info.get("ails", "{}"))
            except Exception:
                pass
            self.add_to_gallery(fpath, meta, select=False)
        if self.gallery:
            self.select(len(self.gallery) - 1)

    def add_to_gallery(self, path, meta, select=True):
        try:
            from PIL import Image, ImageTk
            with Image.open(path) as im:
                im.thumbnail((140, 140))
                th = ImageTk.PhotoImage(im.copy())
        except Exception:
            return
        idx = len(self.gallery)
        card = tk.Frame(self.th_inner, bg=CARD_BG, bd=1, relief="solid", cursor="hand2")
        lbl = tk.Label(card, image=th, bg="#111111", width=144, height=144)
        lbl.pack(padx=2, pady=2)
        cap = tk.Label(card, text=self._caption_text(path, meta), bg="#0e0e0e", fg="#ffeb3b", font=self.f.b9)
        cap.pack(fill="x")
        for w in (card, lbl, cap):
            w.bind("<Button-1>", lambda e, k=idx: self.select(k))
            w.bind("<MouseWheel>", lambda e: self.th_canvas.yview_scroll(int(-e.delta / 120), "units"))
        self.thumb_refs.append(th)
        self.gallery.append({"path": path, "meta": meta, "card": card, "cap": cap})
        self._layout_thumbs()
        if select:
            self.select(idx)

    def _layout_actions(self, force_width=None):
        """Galeri butonlarını sığdığı kadar yan yana dizer, sığmayanı alt satıra alır (kesik yazı olmasın)."""
        acts = getattr(self, "_acts", None)
        if acts is None:
            return
        avail = force_width or acts.winfo_width()
        if avail <= 1:
            return
        rows, x = [[]], 0
        for b in self._act_btns:
            w = b.winfo_reqwidth() + 6
            if rows[-1] and x + w > avail:
                rows.append([])
                x = 0
            rows[-1].append(b)
            x += w
        sig = tuple(tuple(str(b) for b in r) for r in rows)
        if sig == self._act_sig:
            return
        self._act_sig = sig
        for b in self._act_btns:
            b.pack_forget()
        for fr in self._act_rows:
            fr.pack_forget()
        for i, r in enumerate(rows):
            fr = self._act_rows[i]
            fr.pack(side="top", fill="x", pady=(0 if i == 0 else 3, 0))
            for b in r:
                b.pack(in_=fr, side="left", padx=3)

    def _layout_thumbs(self):
        width = max(1, self.th_canvas.winfo_width())
        cols = max(1, width // 156)
        for n, it in enumerate(reversed(self.gallery)):       # en yeni görsel en üstte
            it["card"].grid(row=n // cols, column=n % cols, padx=4, pady=4)

    def selected_item(self):
        if self.sel_index is None or not (0 <= self.sel_index < len(self.gallery)):
            return None
        return self.gallery[self.sel_index]

    def select(self, idx):
        for i, it in enumerate(self.gallery):
            it["card"].config(bg=BLUE if i == idx else CARD_BG)
        self.sel_index = idx
        self._render_preview()
        self._show_meta()
        self._update_rate_buttons()

    def _schedule_preview(self):
        if getattr(self, "_prev_job", None):
            self.r.after_cancel(self._prev_job)
        self._prev_job = self.r.after(150, self._render_preview)

    def _render_preview(self):
        self._prev_job = None
        it = self.selected_item()
        if not it:
            return
        try:
            from PIL import Image, ImageTk
            w = max(100, self.lbl_preview.winfo_width() - 8)
            h = max(100, self.lbl_preview.winfo_height() - 8)
            with Image.open(it["path"]) as im:
                im.thumbnail((w, h))
                self.preview_ref = ImageTk.PhotoImage(im.copy())
            self.lbl_preview.config(image=self.preview_ref, text="")
        except Exception:
            self.lbl_preview.config(image="", text=os.path.basename(it["path"]))

    def _show_meta(self):
        it = self.selected_item()
        if not it:
            self.lbl_meta.config(text="")
            return
        m = it.get("meta") or {}
        if m:
            extra = ""
            if m.get("sampler"):
                extra += f" · {m['sampler']}"
            if m.get("hires"):
                extra += " · hires"
            if m.get("fix"):
                extra += " · " + t("meta_fix", f=m["fix"].get("face", 0), h=m["fix"].get("hand", 0))
            if m.get("stage") == "base":
                extra += " · " + t("meta_base")
            if m.get("loras"):
                extra += " · 🧩 " + ", ".join(f"{x.get('name')} {x.get('w')}" for x in m["loras"])
            if m.get("cards"):
                extra += " · 🎨 " + ", ".join(m["cards"])
            r = self._rating_of(it)
            if r:
                extra += " · " + (t("meta_liked") if r > 0 else t("meta_disliked"))
            txt = (f"{m.get('model', '')} | {m.get('width', '')}×{m.get('height', '')} | seed {m.get('seed', '')} | "
                   f"{m.get('steps', '')} steps{extra}\n{(m.get('prompt') or '')[:300]}")
        else:
            txt = os.path.basename(it["path"])
        self.lbl_meta.config(text=txt)

    def on_show_file(self):
        it = self.selected_item()
        if not it:
            return
        p = it["path"]
        try:
            if IS_WIN:
                subprocess.Popen(["explorer", "/select,", os.path.normpath(p)])
            else:
                self.open_path(os.path.dirname(p))
        except Exception:
            self.open_path(os.path.dirname(p))

    def on_copy_prompt(self):
        it = self.selected_item()
        if not it:
            return
        self.r.clipboard_clear()
        self.r.clipboard_append((it.get("meta") or {}).get("prompt", ""))
        self.log(t("prompt_copied"))

    def on_reuse(self):
        it = self.selected_item()
        if not it or not it.get("meta"):
            return
        m = it["meta"]
        if m.get("model_key") in core.IMAGE_MODEL_BY_KEY:
            self.v_model.set(m["model_key"])
        self.txt_prompt.delete("1.0", "end")
        self.txt_prompt.insert("1.0", m.get("user_text") or m.get("prompt", ""))
        self.txt_extra.delete("1.0", "end")
        self.txt_extra.insert("1.0", m.get("extra", ""))
        if m.get("aspect") in core.ASPECTS:
            self.v_aspect.set(m["aspect"])
        if m.get("orient") in ("landscape", "portrait"):
            self.v_orient.set(m["orient"])
        if "seed" in m:
            self.v_seed_random.set(False)
            self.v_seed.set(str(m["seed"]))

    # ------------------------------------------------------------- advanced window
    def open_advanced(self):
        if self.adv_win is not None and self.adv_win.winfo_exists():
            self.adv_win.lift()
            return
        w = tk.Toplevel(self.r)
        self.adv_win = w
        w.title(t("adv_title"))
        w.configure(bg=BG)
        w.transient(self.r)
        tk.Label(w, text=t("adv_title"), bg=BG, fg=BLUE, font=self.f.b12).pack(pady=(10, 6))
        cols = tk.Frame(w, bg=BG)
        cols.pack(fill="both", expand=True, padx=12)
        cols.grid_columnconfigure(0, weight=1)
        cols.grid_columnconfigure(1, weight=1)
        lf1 = ttk.LabelFrame(cols, text=t("adv_grp_image"))
        lf1.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        lf2 = ttk.LabelFrame(cols, text=t("adv_grp_quality"))
        lf2.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        b1 = tk.Frame(lf1, bg=BG)
        b1.pack(fill="both", expand=True, padx=10, pady=6)
        b1.grid_columnconfigure(1, weight=1)
        b2 = tk.Frame(lf2, bg=BG)
        b2.pack(fill="both", expand=True, padx=10, pady=6)
        b2.grid_columnconfigure(1, weight=1)
        WRAP = 360

        def row_label(parent, r, key):
            tk.Label(parent, text=t(key), bg=BG, fg=TXT, font=self.f.b9, anchor="w").grid(
                row=r, column=0, sticky="nw", pady=4, padx=(0, 8))

        def note(parent, r, fg="#888888"):
            lbl = tk.Label(parent, bg=BG, fg=fg, font=self.f.n9, anchor="w", justify="left", wraplength=WRAP)
            lbl.grid(row=r, column=1, sticky="w")
            return lbl

        # ---------------- sol: görsel ----------------
        row_label(b1, 0, "lbl_aspect")
        KeyCombo(b1, self.v_aspect, self._aspect_items, width=14).cb.grid(row=0, column=1, sticky="w")
        row_label(b1, 1, "lbl_orient")
        KeyCombo(b1, self.v_orient, self._orient_items, width=14).cb.grid(row=1, column=1, sticky="w")
        row_label(b1, 2, "lbl_res")
        KeyCombo(b1, self.v_res, self._res_items, width=14).cb.grid(row=2, column=1, sticky="w")
        row_label(b1, 3, "adv_size")
        lbl_size = tk.Label(b1, bg=BG, fg=GREEN, font=self.f.b10, anchor="w", justify="left", wraplength=WRAP)
        lbl_size.grid(row=3, column=1, sticky="w")

        row_label(b1, 4, "adv_creativity")
        cf = tk.Frame(b1, bg=BG)
        cf.grid(row=4, column=1, sticky="ew")
        tk.Label(cf, text=t("adv_creat_low"), bg=BG, fg="#888888", font=self.f.n9).pack(side="left")
        scale = tk.Scale(cf, from_=1, to=10, orient="horizontal", variable=self.v_creat, bg=BG, fg=TXT,
                         troughcolor=FRM, highlightthickness=0, length=140, font=self.f.n9, activebackground=BLUE)
        scale.pack(side="left", padx=6)
        tk.Label(cf, text=t("adv_creat_high"), bg=BG, fg="#888888", font=self.f.n9).pack(side="left")
        chk_ca = dark_check(b1, self.v_creat_auto, self.f, fg=GREEN)
        chk_ca.config(text=t("auto"))
        chk_ca.grid(row=5, column=1, sticky="w")
        lbl_cna = note(b1, 6, WARN)

        row_label(b1, 7, "adv_seed")
        sf = tk.Frame(b1, bg=BG)
        sf.grid(row=7, column=1, sticky="w")
        ent = tk.Entry(sf, textvariable=self.v_seed, bg=FRM, fg=TXT, insertbackground=TXT, width=14, font=self.f.n10)
        ent.pack(side="left")
        chk_sr = dark_check(sf, self.v_seed_random, self.f, fg=GREEN)
        chk_sr.config(text=t("adv_seed_random"))
        chk_sr.pack(side="left", padx=8)
        row_label(b1, 8, "lbl_count")
        ttk.Combobox(b1, textvariable=self.v_count, values=[str(i) for i in range(1, 9)], state="readonly",
                     width=4).grid(row=8, column=1, sticky="w")

        # ---------------- sağ: kalite ve düzeltme ----------------
        row_label(b2, 0, "adv_quality")
        kq = KeyCombo(b2, self.v_quality, self._quality_items, width=26)
        kq.cb.grid(row=0, column=1, sticky="w")
        row_label(b2, 1, "adv_steps_manual")
        stf = tk.Frame(b2, bg=BG)
        stf.grid(row=1, column=1, sticky="w")
        vcmd = (w.register(lambda v: v == "" or (v.isdigit() and int(v) <= core.MAX_STEPS)), "%P")
        spin = tk.Spinbox(stf, from_=0, to=core.MAX_STEPS, textvariable=self.v_steps_manual, width=5,
                          validate="key", validatecommand=vcmd, bg=FRM, fg=TXT, insertbackground=TXT,
                          buttonbackground=BTN, font=self.f.n10, relief="flat")
        spin.pack(side="left")
        tk.Label(stf, text=t("adv_steps_zero"), bg=BG, fg="#888888", font=self.f.n9).pack(side="left", padx=6)
        lbl_steps = note(b2, 2)

        row_label(b2, 3, "adv_sampler")
        ks = KeyCombo(b2, self.v_sampler, self._sampler_items, width=26)
        ks.cb.grid(row=3, column=1, sticky="w")
        lbl_sampler = note(b2, 4)

        row_label(b2, 5, "adv_hires")
        kh = KeyCombo(b2, self.v_hires, self._hires_items, width=26)
        kh.cb.grid(row=5, column=1, sticky="w")
        lbl_hires = note(b2, 6)

        row_label(b2, 7, "adv_fix")
        ff = tk.Frame(b2, bg=BG)
        ff.grid(row=7, column=1, sticky="w")
        chk_fix = dark_check(ff, self.v_detailer, self.f, fg=BLUE)
        chk_fix.config(text=t("adv_fix_on"))
        chk_fix.pack(side="left")
        chk_faces = dark_check(ff, self.v_fix_faces, self.f, fg=TXT)
        chk_faces.config(text=t("adv_fix_faces"))
        chk_faces.pack(side="left", padx=(10, 0))
        chk_hands = dark_check(ff, self.v_fix_hands, self.f, fg=TXT)
        chk_hands.config(text=t("adv_fix_hands"))
        chk_hands.pack(side="left", padx=(6, 0))
        lbl_fix = note(b2, 8)

        chk_neg = dark_check(b2, self.v_auto_neg, self.f, fg=GREEN)
        chk_neg.config(text=t("adv_auto_neg"))
        chk_neg.grid(row=9, column=0, columnspan=2, sticky="w", pady=(6, 0))
        chk_base = dark_check(b2, self.v_save_base, self.f, fg=TXT)
        chk_base.config(text=t("adv_save_base"))
        chk_base.grid(row=10, column=0, columnspan=2, sticky="w")

        row_label(b2, 11, "adv_pref")
        pf = tk.Frame(b2, bg=BG)
        pf.grid(row=11, column=1, sticky="w", pady=(6, 0))
        chk_pref = dark_check(pf, self.v_pref_auto, self.f, fg=GREEN)
        chk_pref.config(text=t("adv_pref_auto"))
        chk_pref.pack(side="left")
        tk.Scale(pf, from_=0.0, to=1.0, resolution=0.05, orient="horizontal", variable=self.v_pref_weight, length=150,
                 bg=BG, fg=TXT, troughcolor=FRM, highlightthickness=0, font=self.f.n9,
                 activebackground=BLUE).pack(side="left", padx=6)
        lbl_pref = note(b2, 12)

        lbl_warn = tk.Label(w, bg=BG, fg=WARN, font=self.f.b9, anchor="w", justify="left", wraplength=760)
        lbl_warn.pack(fill="x", padx=16, pady=(6, 0))

        def update(*_a):
            if not w.winfo_exists():
                return
            m = self.model()
            a = self.v_aspect.get() if self.v_aspect.get() != "auto" else "1:1"
            o = self.v_orient.get() if self.v_orient.get() != "auto" else "landscape"
            r = self.v_res.get() if self.v_res.get() != "auto" else "normal"
            ww, hh = core.compute_size(m["base_px"], a, o, r)
            auto_note = f"  ({t('auto')})" if "auto" in (self.v_aspect.get(), self.v_orient.get()) else ""
            lbl_size.config(text=f"{ww} × {hh} px — {m['name']}{auto_note}")
            if m["cfg"]:
                scale.config(state="disabled" if self.v_creat_auto.get() else "normal")
                chk_ca.config(state="normal")
                lbl_cna.config(text=f"CFG ≈ {core.creativity_to_cfg(self.v_creat.get())}")
            else:
                scale.config(state="disabled")
                chk_ca.config(state="disabled")
                lbl_cna.config(text=t("adv_creat_na"))
            ent.config(state="disabled" if self.v_seed_random.get() else "normal")
            kq.refresh()
            prof = self.effective_profile(m)
            q = self.v_quality.get() if self.v_quality.get() != "auto" else "balanced"
            manual = self._steps_manual()
            steps = core.resolve_steps(m, q, manual)
            txt = t("adv_steps_manual_info", n=steps) if manual else t("adv_steps_preset_info", n=steps)
            if m.get("turbo"):
                txt += "\n" + t("adv_steps_turbo", n=m["turbo"])
            lbl_steps.config(text=txt)
            if m.get("samplers"):
                ks.cb.config(state="readonly")
                lbl_sampler.config(text=t("adv_sampler_info"))
            else:
                ks.cb.config(state="disabled")
                lbl_sampler.config(text=t("adv_sampler_na"))
            on = core.hires_enabled(self.v_hires.get(), m, q, prof)
            hs = core.hires_size(ww, hh, m) if on else None
            if hs:
                lbl_hires.config(text=t("adv_hires_info", w=hs[0], h=hs[1], s=m.get("hires_strength", 0.45)), fg=GREEN)
            else:
                lbl_hires.config(text=t("adv_hires_off_info"), fg="#888888")
            fix_on = self.v_detailer.get()
            for c in (chk_faces, chk_hands):
                c.config(state="normal" if fix_on else "disabled")
            if not fix_on:
                lbl_fix.config(text=t("adv_fix_off_info"), fg="#888888")
            elif core.is_detailer_installed(self.paths):
                lbl_fix.config(text=t("adv_fix_ready"), fg=GREEN)
            else:
                lbl_fix.config(text=t("adv_fix_missing", size=core.DETAILER["size"]), fg=WARN)
            chk_neg.config(state="normal" if m["neg"] else "disabled")
            pref = os.path.join(core.own_lora_dir(self.paths), core.PREF_LORA_NAME + ".json")
            if m["key"] != "sd15":
                lbl_pref.config(text=t("adv_pref_sd15_only"), fg="#888888")
            elif os.path.exists(pref):
                try:
                    with open(pref, encoding="utf-8") as fh:
                        pi = json.load(fh)
                except Exception:
                    pi = {}
                lbl_pref.config(text=t("adv_pref_info", steps=pi.get("steps", 0), judged=pi.get("judged", 0),
                                       date=pi.get("updated", "")), fg=GREEN)
            else:
                lbl_pref.config(text=t("adv_pref_none"), fg="#888888")
            warn = []
            if r == "high" and prof in ("low", "cpu"):
                warn.append(t("adv_warn_high"))
            if hs and prof in ("low", "cpu"):
                warn.append(t("adv_warn_hires_slow"))
            lbl_warn.config(text="\n".join(warn))

        traces = []
        for v in (self.v_aspect, self.v_orient, self.v_res, self.v_creat, self.v_creat_auto, self.v_seed_random,
                  self.v_model, self.v_quality, self.v_steps_manual, self.v_sampler, self.v_hires, self.v_detailer,
                  self.v_fix_faces, self.v_fix_hands, self.v_profile):
            traces.append((v, v.trace_add("write", update)))

        def reset():
            for v, val in ((self.v_aspect, "auto"), (self.v_orient, "auto"), (self.v_res, "auto"),
                           (self.v_quality, "auto"), (self.v_creat, 5), (self.v_creat_auto, True),
                           (self.v_seed_random, True), (self.v_count, "1"), (self.v_steps_manual, "0"),
                           (self.v_sampler, "auto"), (self.v_hires, "auto"), (self.v_detailer, True),
                           (self.v_fix_faces, True), (self.v_fix_hands, True), (self.v_auto_neg, True),
                           (self.v_save_base, False), (self.v_pref_auto, True), (self.v_pref_weight, 0.6)):
                v.set(val)

        def close():
            for v, tid in traces:
                try:
                    v.trace_remove("write", tid)
                except Exception:
                    pass
            if not str(self.v_steps_manual.get()).strip():
                self.v_steps_manual.set("0")
            self.adv_win = None
            self.save()
            w.destroy()

        self._adv_close = close
        btns = tk.Frame(w, bg=BG)
        btns.pack(fill="x", padx=16, pady=12)
        ttk.Button(btns, text=t("adv_reset"), command=reset).pack(side="left")
        ttk.Button(btns, text=t("adv_close"), command=close).pack(side="right")
        w.protocol("WM_DELETE_WINDOW", close)
        update()
        w.update_idletasks()
        w.minsize(w.winfo_reqwidth(), w.winfo_reqheight())

    # ------------------------------------------------------------- v2 hazırlık penceresi
    def open_style_window(self):
        if self.style_win is not None and self.style_win.alive():
            self.style_win.w.lift()
            return
        self.style_win = StyleStudioWindow(self)

    # ------------------------------------------------------------- data folder / misc
    def on_data_dir(self):
        if self.running():
            messagebox.showinfo(APP_NAME, t("busy"), parent=self.r)
            return
        cur = self.paths.root
        if not messagebox.askyesno(t("data_dir_title"), t("data_dir_info", path=cur,
                                                          free=max(0.0, core.disk_free_gb(cur))), parent=self.r):
            return
        d = filedialog.askdirectory(parent=self.r, title=t("data_dir_title"))
        if not d:
            return
        new = os.path.abspath(d)
        if os.path.dirname(new) == new or (IS_WIN and len(new) <= 3):
            new = os.path.join(new, core.DEFAULT_ROOT_NAME)
        nc, cc = os.path.normcase(new), os.path.normcase(cur)
        if nc == cc:
            messagebox.showinfo(t("data_dir_title"), t("data_dir_same"), parent=self.r)
            return
        if nc.startswith(cc + os.sep):
            messagebox.showwarning(t("data_dir_title"), t("data_dir_inside"), parent=self.r)
            return
        if not core.is_writable_dir(new):
            messagebox.showwarning(t("data_dir_title"), t("loc_not_writable"), parent=self.r)
            return
        size = core.data_move_size(cur) / core.GB
        free = core.disk_free_gb(new)
        if not core.same_drive(new, cur) and 0 <= free < size + 1.0:
            messagebox.showwarning(t("data_dir_title"), t("err_disk_full", drive=core.drive_label(new),
                                                          need=size + 1.0, free=free), parent=self.r)
            return
        if not messagebox.askyesno(t("data_dir_title"), t("data_dir_move_q", size=size, path=new), parent=self.r):
            return
        cfg = load_pointer()
        cfg["pending_move"] = {"from": cur, "to": new}
        save_pointer(cfg)
        self.restart_program()

    def restart_program(self):
        self.save()
        # Başlatıcı, programı ilk açan Python'la (veri klasörünün DIŞINDAKİ) yeniden açılır: veri klasörü taşınacaksa
        # programa özel Python'un dosyaları kullanımda olmasın.
        py = None
        lp = load_pointer().get("launcher_python")
        for cand in (lp, core.base_python(), sys.executable):
            if cand and os.path.exists(cand) and _outside(cand, self.paths.root):
                py = cand
                break
        py = py or core.base_python() or sys.executable
        env = core.sub_env(drop_ours=True)
        try:
            subprocess.Popen([py, SCRIPT_PATH, "--wait-pid", str(os.getpid())], cwd=APP_DIR, env=env)
        except Exception as e:
            self.log(friendly_error(e))
            return
        self.on_close()

    def choose_outdir(self):
        d = filedialog.askdirectory(initialdir=self.v_outdir.get(), parent=self.r)
        if d:
            self.v_outdir.set(d)
            self.save()

    def open_path(self, p):
        try:
            os.makedirs(p, exist_ok=True)
            if IS_WIN:
                os.startfile(p)  # noqa
            elif sys.platform == "darwin":
                subprocess.Popen(["open", p])
            else:
                subprocess.Popen(["xdg-open", p])
        except Exception:
            pass

    def show_licenses(self):
        w = tk.Toplevel(self.r)
        w.title(t("licenses_title"))
        w.configure(bg=BG)
        w.geometry("780x560")
        txt = tk.Text(w, bg="#111111", fg=TXT, font=self.f.n9, wrap="word", relief="flat")
        txt.pack(fill="both", expand=True, padx=10, pady=10)
        lines = [t("licenses_title"), "", t("lic_intro"), "", "— Models —"]
        for m in core.IMAGE_MODELS:
            lines.append(f"• {m['name']}  ({m['repo']}) — {m['license']}")
        for m in core.LLM_MODELS:
            lines.append(f"• {m['name']}  (Ollama: {m['key']}) — {m['license']}")
        lines.append(f"• {core.UPSCALER['name']} (xinntao/Real-ESRGAN) — {core.UPSCALER['license']}")
        lines.append("• YuNet face detector (opencv/opencv_zoo) — MIT")
        lines.append("• MediaPipe palm detector (opencv/opencv_zoo, Google) — Apache-2.0")
        lines.append("• lbpcascade_animeface (nagadomi) — MIT")
        lines += ["", t("lic_rail"), "", "— Software —",
                  "• Python — PSF License", "• PyTorch — BSD-3-Clause", "• Hugging Face diffusers / transformers / "
                  "accelerate / huggingface_hub / safetensors — Apache-2.0", "• Ollama — MIT",
                  "• Pillow — MIT-CMU (HPND)", "• NumPy — BSD-3-Clause", "• psutil — BSD-3-Clause",
                  "• requests — Apache-2.0", "• OpenCV (opencv-python-headless) — Apache-2.0",
                  "• NVIDIA CUDA runtime (via PyTorch wheels) — NVIDIA EULA", "",
                  t("lic_addons")]
        txt.insert("1.0", "\n".join(lines))
        txt.config(state="disabled")

    # ------------------------------------------------------------- shutdown (SRT ile aynı davranış)
    def _trigger_shutdown(self):
        self._shutdown_active = True
        try:
            if IS_WIN:
                subprocess.Popen(["shutdown", "/s", "/t", "60"], **core.no_window_kwargs())
            else:
                subprocess.Popen(["shutdown", "-h", "+1"])
        except Exception:
            return
        dlg = tk.Toplevel(self.r)
        dlg.title(t("shutdown_title"))
        dlg.geometry("460x200")
        dlg.configure(bg=CARD_BG)
        dlg.attributes("-topmost", True)
        dlg.resizable(False, False)
        tk.Label(dlg, text=t("shutdown_msg"), bg=CARD_BG, fg=WARN, font=self.f.b10, justify="center",
                 pady=16, wraplength=430).pack(fill="x", padx=12)

        def abort():
            self.v_shutdown.set(False)
            self._cancel_shutdown()
            if dlg.winfo_exists():
                dlg.destroy()

        tk.Button(dlg, text=t("shutdown_cancel"), bg="#c62828", fg="white", font=self.f.b11, relief="flat",
                  cursor="hand2", command=abort).pack(fill="x", padx=24, pady=8, ipady=8)

        def poll():
            if not dlg.winfo_exists():
                return
            if not self.v_shutdown.get() or not self._shutdown_active:
                abort()
                return
            dlg.after(400, poll)

        dlg.protocol("WM_DELETE_WINDOW", abort)
        dlg.after(400, poll)

    def _cancel_shutdown(self):
        if not self._shutdown_active:
            return
        self._shutdown_active = False
        try:
            if IS_WIN:
                subprocess.Popen(["shutdown", "/a"], **core.no_window_kwargs())
            else:
                subprocess.Popen(["shutdown", "-c"])
        except Exception:
            pass

    def on_close(self):
        self.abort.set()
        self.save()
        self._cancel_shutdown()
        try:
            self.engine.unload()
        except Exception:
            pass
        for c in list(self.clients.values()):
            c.close()
        try:
            self.r.destroy()
        except Exception:
            pass
        os._exit(0)


# =============================================================================
# Giriş noktaları
# =============================================================================
def gui_main():
    cfg = load_pointer()
    if not cfg.get("data_root"):
        return launcher_main()
    paths = core.DataPaths(cfg["data_root"])
    core.apply_env(paths)
    T.set(LANGS.get(cfg.get("lang") or "English", "en"))
    root = tk.Tk()
    ImageStudio(root, paths)
    root.mainloop()
    return 0


def main():
    if "--ails-child" in sys.argv:
        sys.exit(gui_main() or 0)
    cfg = load_pointer()
    if cfg.get("data_root") and not cfg.get("pending_move") and not cfg.get("pending_runtime_reset") and \
            core.running_in_venv(core.DataPaths(cfg["data_root"])):
        sys.exit(gui_main() or 0)
    sys.exit(launcher_main() or 0)


if __name__ == "__main__":
    main()
