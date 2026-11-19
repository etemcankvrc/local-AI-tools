import os, sys, re, threading, time, base64, json, subprocess, queue, io, ctypes, socket, shutil, tempfile, atexit, gc, hashlib, zlib, codecs, importlib, types, uuid
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote_plus, urlparse
import requests
from requests.adapters import HTTPAdapter

APP_NAME = "AI Local Translator"
APP_VERSION = "1.0.6"

# PyInstaller (_PYI_*, _MEI*) kalıntılarını temizler ve alt süreç DLL çakışmalarını önler.
for _k in list(os.environ.keys()):
    if _k.upper().startswith(("_PYI", "_MEI", "PYINSTALLER")):
        os.environ.pop(_k, None)


# Geçici klasör ve çıkışta Ollama RAM/VRAM temizleme sistemi.
TEMP_DIR = tempfile.mkdtemp(prefix="ai_local_tr_")
ACTIVE_HOST = "http://localhost:11434"
APP_DATA_DIR = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".local", "share"), "AI_Local_Translator")

def unload_all_ollama_models(ollama_base_url=None, fast_exit=False):
    base_u = (ollama_base_url or ACTIVE_HOST).rstrip("/")
    t_out = 1.2 if fast_exit else 3.0
    try:
        r = requests.get(f"{base_u}/api/ps", timeout=t_out)
        if r.status_code == 200:
            for m in r.json().get("models", []):
                m_name = m.get("name")
                if m_name:
                    try:
                        requests.post(f"{base_u}/api/generate", json={"model": m_name, "keep_alive": 0}, timeout=t_out)
                    except:
                        pass
    except:
        pass
    if fast_exit and os.name == "nt" and ("localhost" in base_u or "127.0.0.1" in base_u):
        try:
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            subprocess.run(["taskkill", "/F", "/IM", "ollama_llama_server.exe", "/T"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1.5, startupinfo=si)
        except:
            pass

_CLEANUP_DONE = False

def cleanup_on_exit():
    # Hem atexit hem de pencere kapanışı tarafından çağrılır; iki kez çalışıp çıkışı yavaşlatmasın.
    global _CLEANUP_DONE
    if _CLEANUP_DONE:
        return
    _CLEANUP_DONE = True
    try:
        unload_all_ollama_models(ACTIVE_HOST, fast_exit=True)
    except:
        pass
    try:
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR, ignore_errors=True)
    except:
        pass

atexit.register(cleanup_on_exit)

# Bilgisayarı kapatma zamanlayıcısı ve iptal sistemi.
SHUTDOWN_TIMER_ACTIVE = False

def cancel_pc_shutdown():
    global SHUTDOWN_TIMER_ACTIVE
    SHUTDOWN_TIMER_ACTIVE = False
    if os.name == "nt":
        try:
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            subprocess.Popen(["shutdown", "/a"], startupinfo=si, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            pass
    else:
        try:
            subprocess.Popen(["shutdown", "-c"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            pass

def trigger_pc_shutdown_with_cancel(parent_win, chk_var, ui_lang="English"):
    global SHUTDOWN_TIMER_ACTIVE
    if not chk_var.get():
        return
    SHUTDOWN_TIMER_ACTIVE = True
    if os.name == "nt":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        subprocess.Popen(["shutdown", "/s", "/t", "60"], startupinfo=si)
    else:
        subprocess.Popen(["shutdown", "-h", "+1"])

    def _show_cancel_dlg():
        if not parent_win.winfo_exists():
            return
        dlg = tk.Toplevel(parent_win)
        dlg.title("⚠️ Shutdown Scheduled" if ui_lang != "Türkçe" else "⚠️ Bilgisayar Kapatılıyor")
        dlg.geometry("440x190")
        dlg.configure(bg="#181818")
        dlg.attributes("-topmost", True)
        dlg.resizable(False, False)

        msg = (
            "🔌 İşlem tamamlandı! Bilgisayar 60 saniye içinde kapanacak.\nİptal etmek için aşağıdaki butona basın veya kutucuğu kaldırın."
            if ui_lang == "Türkçe" else
            "🔌 Process finished! PC will shut down in 60 seconds.\nClick the button below or uncheck the box to cancel."
        )
        tk.Label(dlg, text=msg, bg="#181818", fg="#ffb74d", font=("Segoe UI", 10, "bold"), justify="center", pady=16).pack(fill="x", padx=12)

        def _do_abort():
            chk_var.set(False)
            cancel_pc_shutdown()
            if dlg.winfo_exists():
                dlg.destroy()

        btn_txt = "❌ KAPATMAYI İPTAL ET (DURDUR)" if ui_lang == "Türkçe" else "❌ CANCEL SHUTDOWN NOW"
        tk.Button(dlg, text=btn_txt, bg="#c62828", fg="white", font=("Segoe UI", 11, "bold"), relief="flat", cursor="hand2", command=_do_abort).pack(fill="x", padx=24, pady=8, ipady=8)

        def _poll_chk():
            if not dlg.winfo_exists():
                return
            if not chk_var.get() or not SHUTDOWN_TIMER_ACTIVE:
                cancel_pc_shutdown()
                dlg.destroy()
                return
            dlg.after(400, _poll_chk)

        dlg.protocol("WM_DELETE_WINDOW", _do_abort)
        dlg.after(400, _poll_chk)

    parent_win.after(0, _show_cancel_dlg)

# NVIDIA CUDA ve cuDNN DLL yollarını sisteme tanıtır.
def register_cuda_dlls():
    if os.name != "nt":
        return
    search_roots = ([sys._MEIPASS] if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS') else []) + list(sys.path)
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS') and hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(sys._MEIPASS)
        except:
            pass
    for sp in search_roots:
        if not sp or not os.path.isdir(sp):
            continue
        nv_root = os.path.join(sp, "nvidia")
        if os.path.isdir(nv_root):
            for sub in ("cublas", "cudnn", "cuda_runtime"):
                bin_dir = os.path.join(nv_root, sub, "bin")
                if os.path.isdir(bin_dir):
                    if bin_dir not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                    try:
                        os.add_dll_directory(bin_dir)
                    except:
                        pass

register_cuda_dlls()

# HTTP oturumu, FFmpeg yolu ve güvenli dosya adı temizleyicisi.
def create_http_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"})
    adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=2)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

HTTP = create_http_session()

def get_ffmpeg_path():
    candidates = []
    if getattr(sys, 'frozen', False):
        base_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
        candidates.extend([base_dir, os.path.dirname(sys.executable)])
    candidates.extend([os.getcwd(), r"C:\ffmpeg\bin", r"C:\ffmpeg"])
    for d in candidates:
        exe_p = os.path.join(d, "ffmpeg.exe")
        if os.path.exists(exe_p):
            if d not in os.environ.get("PATH", ""):
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            return exe_p
    return shutil.which("ffmpeg")

FFMPEG_EXE = get_ffmpeg_path()

# Windows MAX_PATH (260 karakter) sınırını aşmamak için dosya adlarını 85 karakterle sınırlar.
def sanitize_filename(name, max_len=85):
    cleaned = re.sub(r'[\x00-\x1f\\/*?:"<>|]+', "_", str(name or "Video"))
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ._")
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].strip(" ._")
    return cleaned if cleaned else "Video"

# yt-dlp 2025.11+ sürümleri YouTube'un JavaScript sorgularını çözmek için harici bir JS çalışma
# ortamına (Deno önerilir, Node da olur) ihtiyaç duyar. Uygulama klasöründe veya PATH'te arar.
def get_js_runtime_for_ytdlp():
    names = (("deno.exe", "deno"), ("node.exe", "node")) if os.name == "nt" else (("deno", "deno"), ("node", "node"))
    dirs = []
    if getattr(sys, 'frozen', False):
        dirs += [getattr(sys, '_MEIPASS', ""), os.path.dirname(sys.executable)]
    try:
        dirs.append(os.path.dirname(os.path.abspath(__file__)))
    except Exception:
        pass
    dirs.append(os.getcwd())
    for d in dirs:
        if not d:
            continue
        for fname, key in names:
            p = os.path.join(d, fname)
            if os.path.isfile(p):
                return {key: {"path": p}}
    if shutil.which("deno"):
        return {"deno": {}}
    if shutil.which("node"):
        return {"node": {}}
    return None

# Farklı karakter kodlamalarına sahip SRT dosyalarını bozulmadan okur.
# DÜZELTME: Eski sürüm BOM'suz Türkçe ANSI (cp1254) dosyaları "utf-16" ile çözmeye çalışıyordu;
# dosya uzunluğu çift sayı olunca bu "başarılı" sayılıp anlamsız Çince benzeri karakterler üretiyordu.
def _decode_text_bytes_smart(raw_bytes):
    if raw_bytes.startswith(codecs.BOM_UTF8):
        return raw_bytes[len(codecs.BOM_UTF8):].decode("utf-8", errors="replace")
    if raw_bytes.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        return raw_bytes.decode("utf-32", errors="replace")
    if raw_bytes.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return raw_bytes.decode("utf-16", errors="replace")
    sample = raw_bytes[:4096]
    if len(sample) >= 8:
        half = len(sample) / 2.0
        even_nul = sample[0::2].count(0)
        odd_nul = sample[1::2].count(0)
        if odd_nul > half * 0.3 and even_nul < half * 0.05:
            return raw_bytes.decode("utf-16-le", errors="replace")
        if even_nul > half * 0.3 and odd_nul < half * 0.05:
            return raw_bytes.decode("utf-16-be", errors="replace")
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        pass
    enc = None
    try:
        from charset_normalizer import from_bytes
        best = from_bytes(raw_bytes).best()
        if best is not None:
            enc = (best.encoding or "").lower().replace("-", "_")
    except Exception:
        enc = None
    turkish_specific = any(b in raw_bytes for b in (b"\xd0", b"\xdd", b"\xde", b"\xf0", b"\xfd", b"\xfe"))
    latin_family = {"", "cp1252", "latin_1", "iso8859_1", "iso8859_15", "iso8859_9", "cp1254", "cp1250", "iso8859_2", "mac_roman", "cp850", "cp437", "ascii"}
    if not enc or (enc in latin_family and turkish_specific):
        enc = "cp1254"
    for cand in (enc, "cp1254", "cp1252"):
        try:
            return raw_bytes.decode(cand)
        except Exception:
            continue
    return raw_bytes.decode("latin-1", errors="replace")

def read_srt_file_smart(file_path):
    with open(file_path, "rb") as f:
        raw_bytes = f.read()
    txt = _decode_text_bytes_smart(raw_bytes)
    return txt.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿").replace("\x00", "")

try:
    from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageOps, ImageTk
    PIL_OK = True
except ImportError:
    PIL_OK = False

# LİSANS DÜZELTMESİ: faster-whisper modül yüklenirken "av" (PyAV) kütüphanesini ister. PyAV'ın hazır
# Windows paketleri GPL lisanslı x264/x265 DLL'leri içerir; bunlar ücretli/kapalı kaynak exe'ye
# gömülürse tüm uygulama GPL yükümlülüğüne girer. Ses artık harici ffmpeg.exe ile çözülüp
# faster-whisper'a numpy dizisi olarak veriliyor; PyAV pakete eklenmediğinde yerine zararsız bir
# yer tutucu konur (build betiğinde: --exclude-module av).
def _install_av_stub_if_missing():
    try:
        import av  # noqa: F401
        return False
    except Exception:
        stub = types.ModuleType("av")
        stub.__ai_local_stub__ = True
        def _pyav_unavailable(*_a, **_k):
            raise RuntimeError("PyAV is not bundled; audio is decoded with FFmpeg instead.")
        stub.open = _pyav_unavailable
        sys.modules["av"] = stub
        return True

AV_STUBBED = _install_av_stub_if_missing()

try:
    import yt_dlp
    YTDLP_OK = True
except Exception:
    YTDLP_OK = False

try:
    import ctranslate2
    from faster_whisper import WhisperModel
    WHISPER_OK = True
except Exception:
    WHISPER_OK = False

VA = YTDLP_OK and WHISPER_OK

# Whisper GPU/CPU yükleyici ve doğrulayıcı.
def get_best_whisper_device():
    if not WHISPER_OK:
        return ("cpu", "int8")
    try:
        if ctranslate2.get_cuda_device_count() > 0:
            return ("cuda", "int8_float16")
    except:
        pass
    return ("cpu", "int8")

def load_whisper_auto(model_size, force_cpu=False):
    cpu_threads = max(4, (os.cpu_count() or 4) // 2)
    if not force_cpu:
        dev, c_type = get_best_whisper_device()
        if dev == "cuda":
            for ct in (c_type, "int8", "float16"):
                try:
                    wm = WhisperModel(model_size, device="cuda", compute_type=ct)
                    return wm, "GPU (CUDA)"
                except:
                    continue
    wm = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=cpu_threads)
    return wm, "CPU"

def verify_gpu_ready():
    register_cuda_dlls()
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return True, "CUDA GPU Ready"
    except Exception as e:
        return False, str(e)
    return False, "NVIDIA GPU / CUDA (cuBLAS/cuDNN) not detected."

def _hidden_proc_kwargs():
    if os.name == "nt":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        return {"startupinfo": si, "creationflags": subprocess.CREATE_NO_WINDOW}
    return {}

# Sesi harici FFmpeg ile 16 kHz mono float32 numpy dizisine çevirir (PyAV gerektirmez).
def decode_audio_ffmpeg(media_path, sampling_rate=16000, abort_check=None, timeout_sec=5400):
    import numpy as np
    if not FFMPEG_EXE:
        raise RuntimeError("FFmpeg not found (ffmpeg.exe)")
    cmd = [FFMPEG_EXE, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", media_path,
           "-vn", "-sn", "-dn", "-ac", "1", "-ar", str(int(sampling_rate)), "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    err_file = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err_file, **_hidden_proc_kwargs())
    buf = bytearray()
    t0 = time.time()
    try:
        while True:
            if abort_check and abort_check():
                proc.kill()
                return None
            chunk = proc.stdout.read(1 << 20)
            if not chunk:
                break
            buf.extend(chunk)
            if time.time() - t0 > timeout_sec:
                proc.kill()
                raise RuntimeError("FFmpeg audio decode timeout")
        proc.wait(timeout=30)
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
    err_txt = ""
    try:
        err_file.seek(0)
        err_txt = err_file.read().decode("utf-8", errors="ignore").strip()
        err_file.close()
    except Exception:
        pass
    usable = len(buf) - (len(buf) % 4)
    if usable <= 0:
        raise RuntimeError("FFmpeg could not decode audio: " + (err_txt[-240:] or "no audio stream"))
    return np.frombuffer(buf, dtype=np.float32, count=usable // 4)

# Live Web translation (PySide6/QtWebEngine) is a PRO-only feature and is NOT part of this
# open-source edition. This build focuses on SRT translation and video-to-SRT translation.
PA = False

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Windows Görev Çubuğu (Taskbar) ilerleme çubuğu sınıfı.
class WinTaskbarProgress:
    TBPF_NOPROGRESS = 0x0
    TBPF_NORMAL = 0x2
    def __init__(self):
        self.ptr = ctypes.c_void_p()
        self.ok = False
        if os.name != "nt":
            return
        try:
            ole32 = ctypes.windll.ole32
            ole32.CoInitialize(None)
            class GUID(ctypes.Structure):
                _fields_ = [("Data1", ctypes.c_ulong), ("Data2", ctypes.c_ushort), ("Data3", ctypes.c_ushort), ("Data4", ctypes.c_ubyte * 8)]
            clsid = GUID(0x56FDF344, 0xFD6D, 0x11D0, (ctypes.c_ubyte * 8)(0x95, 0x8A, 0x00, 0x60, 0x97, 0xC9, 0xA0, 0x90))
            iid = GUID(0xEA1AFB91, 0x9E28, 0x4B86, (ctypes.c_ubyte * 8)(0x90, 0xE9, 0x9E, 0x9F, 0x8A, 0x5E, 0xEF, 0xAF))
            if ole32.CoCreateInstance(ctypes.byref(clsid), None, 1, ctypes.byref(iid), ctypes.byref(self.ptr)) == 0 and self.ptr.value:
                vtable = ctypes.cast(self.ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
                ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p)(vtable[3])(self.ptr)
                self.ok = True
        except:
            self.ok = False

    def _get_top_hwnd(self, w):
        try:
            hwnd = w if isinstance(w, int) else int(w.winfo_id())
            p = ctypes.windll.user32.GetParent(hwnd)
            return p if p else hwnd
        except:
            return 0

    def set_progress(self, w, cur, tot=100, state=2):
        if not self.ok or not self.ptr.value:
            return
        try:
            hwnd = self._get_top_hwnd(w)
            if not hwnd:
                return
            vt = ctypes.cast(self.ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)(vt[10])(self.ptr, ctypes.c_void_p(hwnd), state)
            if state != self.TBPF_NOPROGRESS:
                ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_ulonglong)(vt[9])(self.ptr, ctypes.c_void_p(hwnd), max(0, int(cur)), max(1, int(tot)))
        except:
            pass

    def clear(self, w):
        self.set_progress(w, 0, 100, self.TBPF_NOPROGRESS)

TASKBAR = WinTaskbarProgress()

# Ollama sunucu tespiti ve otomatik yeniden bağlanma sistemi.
URL = f"{ACTIVE_HOST}/api/chat"
T_URL = f"{ACTIVE_HOST}/api/tags"
P_URL = f"{ACTIVE_HOST}/api/pull"
D_URL = f"{ACTIVE_HOST}/api/delete"
LAN_LOCK = threading.Lock()
MODEL_CACHE = {"models": [], "ts": 0.0, "checked": False}

def set_active_ollama_host(host_url):
    global ACTIVE_HOST, URL, T_URL, P_URL, D_URL
    ACTIVE_HOST = host_url.rstrip("/")
    URL = f"{ACTIVE_HOST}/api/chat"
    T_URL = f"{ACTIVE_HOST}/api/tags"
    P_URL = f"{ACTIVE_HOST}/api/pull"
    D_URL = f"{ACTIVE_HOST}/api/delete"

def _probe_ollama_ip(ip_str, timeout_sec=0.6):
    try:
        with socket.create_connection((ip_str, 11434), timeout=0.35):
            pass
        r = requests.get(f"http://{ip_str}:11434/api/tags", timeout=timeout_sec)
        if r.status_code == 200:
            models = [m.get("name", "") for m in r.json().get("models", []) if m.get("name")]
            return (ip_str, models)
    except:
        pass
    return None

def _is_private_lan_ip(ip):
    try:
        a, b, _c, _d = [int(x) for x in ip.split(".")]
    except Exception:
        return False
    return a == 10 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168)

def get_arp_neighbor_ips():
    ips = []
    try:
        si = None
        if os.name == "nt":
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        out = subprocess.check_output(["arp", "-a"], universal_newlines=True, timeout=1.5, startupinfo=si)
        for line in out.splitlines():
            if "dynamic" in line.lower() or "dinamik" in line.lower():
                m = re.search(r"(\d+\.\d+\.\d+\.\d+)", line)
                if m:
                    ip = m.group(1)
                    # DÜZELTME: "172." ile başlayan tüm adresler değil, yalnızca özel ağ aralıkları (172.16-31) taranır.
                    if _is_private_lan_ip(ip) and not ip.endswith((".255", ".1", ".0")):
                        ips.append(ip)
    except:
        pass
    return ips[:4]

def get_cached_models():
    return list(MODEL_CACHE["models"])

def get_installed_ollama_models(force_refresh=False):
    with LAN_LOCK:
        now = time.time()
        if not force_refresh and MODEL_CACHE["checked"] and (now - MODEL_CACHE["ts"] < 25.0):
            return list(MODEL_CACHE["models"])

        MODEL_CACHE["checked"] = True
        MODEL_CACHE["ts"] = now

        res_local = _probe_ollama_ip("127.0.0.1", 1.2)
        if res_local is not None:
            set_active_ollama_host("http://localhost:11434")
            MODEL_CACHE["models"] = res_local[1]
            return list(res_local[1])

        for ip in get_arp_neighbor_ips():
            out = _probe_ollama_ip(ip, 0.6)
            if out and out[1]:
                set_active_ollama_host(f"http://{out[0]}:11434")
                MODEL_CACHE["models"] = out[1]
                return list(out[1])

        return list(MODEL_CACHE["models"])

def wait_until_ollama_ready(abort_check=None, status_cb=None, max_wait_sec=90):
    start_t = time.time()
    tried_spawn = False
    while True:
        if abort_check and abort_check():
            return False
        try:
            r = requests.get(T_URL, timeout=2.5)
            if r.status_code == 200:
                return True
        except:
            pass
        if not tried_spawn and ("localhost" in ACTIVE_HOST or "127.0.0.1" in ACTIVE_HOST):
            tried_spawn = True
            try:
                ollama_bin = shutil.which("ollama") or os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe")
                if ollama_bin and os.path.exists(ollama_bin):
                    si = subprocess.STARTUPINFO() if os.name == "nt" else None
                    if si:
                        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    cflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                    subprocess.Popen([ollama_bin, "serve"], startupinfo=si, creationflags=cflags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except:
                pass
        if status_cb:
            try:
                status_cb("⏳ Ollama yanıt bekleniyor / yeniden bağlanılıyor...")
            except:
                pass
        if time.time() - start_t > max_wait_sec:
            return False
        time.sleep(2.0)

# Model yetenekleri (/api/show -> "capabilities": vision, thinking...) ve güvenli sohbet çağrısı.
_MODEL_CAPS_CACHE = {}
_MODEL_CAPS_LOCK = threading.Lock()
_VISION_NAME_HINTS = ("vision", "-vl", "vl:", "llava", "bakllava", "minicpm-v", "moondream", "gemma3:", "gemma3n", "gemma4",
                      "translategemma", "qwen2.5vl", "qwen3-vl", "mistral-small3", "llama4", "pixtral", "granite3.2-vision")

class OllamaError(Exception):
    pass

class OllamaModelMissing(OllamaError):
    pass

def get_model_capabilities(model, session=None, timeout=6.0):
    key = (ACTIVE_HOST, model)
    with _MODEL_CAPS_LOCK:
        if key in _MODEL_CAPS_CACHE:
            return _MODEL_CAPS_CACHE[key]
    caps = None
    try:
        r = (session or HTTP).post(f"{ACTIVE_HOST}/api/show", json={"model": model, "name": model}, timeout=timeout)
        if r.status_code == 200:
            caps = [str(c).lower() for c in (r.json().get("capabilities") or [])]
    except Exception:
        caps = None
    if caps is not None:
        with _MODEL_CAPS_LOCK:
            _MODEL_CAPS_CACHE[key] = caps
    return caps

def is_vision_model_name(model):
    m = str(model or "").lower()
    if m.startswith(("gemma3:1b", "gemma3:270m")):
        return False
    return any(h in m for h in _VISION_NAME_HINTS)

def model_supports_vision(model, session=None):
    caps = get_model_capabilities(model, session)
    if caps:
        return "vision" in caps
    return is_vision_model_name(model)

def model_supports_thinking(model, session=None):
    caps = get_model_capabilities(model, session)
    return bool(caps) and "thinking" in caps

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

def strip_reasoning(text):
    if not text:
        return ""
    t = _THINK_BLOCK_RE.sub("", str(text))
    # Yanıt kesildiyse kapanmamış <think> bloğu kalır; tamamen at (eskiden çeviri yerine düşünce metni basılıyordu).
    t = re.sub(r"<think>.*\Z", "", t, flags=re.DOTALL | re.IGNORECASE)
    if re.search(r"</think>", t, flags=re.IGNORECASE):
        t = re.split(r"</think>", t, flags=re.IGNORECASE)[-1]
    t = re.sub(r"<\|channel\|?>.*?<\|?channel\|>", "", t, flags=re.DOTALL)
    return t.strip()

def ollama_chat(model, messages, session=None, options=None, timeout=180, keep_alive="30m", disable_thinking=True):
    """Ollama /api/chat çağrısı.
    - Düşünen (reasoning) modellerde düşünmeyi kapatır (hız + kesik yanıt sorunu).
    - Desteklemeyen modelde 400 'does not support thinking' gelirse alanı kaldırıp tekrar dener.
    - <think> bloklarını temizler. HTTP hatalarında OllamaError fırlatır."""
    sess = session or HTTP
    payload = {"model": model, "messages": messages, "stream": False, "keep_alive": keep_alive}
    if options:
        payload["options"] = options
    if disable_thinking and model_supports_thinking(model, sess):
        payload["think"] = False
    for attempt in range(3):
        r = sess.post(URL, json=payload, timeout=timeout)
        if r.status_code == 400 and "think" in payload and "think" in (r.text or "").lower():
            payload.pop("think", None)
            with _MODEL_CAPS_LOCK:
                old = _MODEL_CAPS_CACHE.get((ACTIVE_HOST, model)) or []
                _MODEL_CAPS_CACHE[(ACTIVE_HOST, model)] = [c for c in old if c != "thinking"]
            continue
        if r.status_code != 200:
            try:
                err = str(r.json().get("error", ""))
            except Exception:
                err = (r.text or "")[:240]
            if r.status_code == 404 or "not found" in err.lower():
                raise OllamaModelMissing(f"'{model}': {err}")
            raise OllamaError(f"HTTP {r.status_code}: {err}")
        data = r.json()
        content = strip_reasoning((data.get("message") or {}).get("content") or "")
        if not content and "think" in payload and attempt == 0:
            payload.pop("think", None)
            continue
        return content
    return ""

def ollama_preload(model, session=None, timeout=420):
    # Büyük modeller ilk istekte diske/VRAM'e yüklenirken zaman aşımına düşmesin diye önceden yükler.
    try:
        (session or HTTP).post(f"{ACTIVE_HOST}/api/generate", json={"model": model, "keep_alive": "30m"}, timeout=timeout)
        return True
    except Exception:
        return False

# Çeviri dilleri ve model katalogları.
L = {
    "Auto": ("Auto", "auto"),
    "Turkish": ("Turkish", "tr"),
    "English": ("English", "en"),
    "German": ("German", "de"),
    "French": ("French", "fr"),
    "Japanese": ("Japanese", "ja"),
    "Korean": ("Korean", "ko"),
    "Chinese": ("Chinese", "zh"),
    "Spanish": ("Spanish", "es"),
    "Russian": ("Russian", "ru")
}

# LİSANS DÜZELTMESİ: Ücretli (Patreon abonelikli) bir üründe yalnızca ticari kullanıma izin veren
# lisanslı modeller önerilir/tek tıkla indirilir. Ticari kullanıma kapalı veya belirsiz olanlar
# (aya-expanse: CC-BY-NC-4.0, qwen2.5:3b: Qwen Research License, minicpm-v: kayıt şartlı,
# llava / llava-llama3 / bakllava: ticari olmayan LLaVA eğitim verisi) katalogdan çıkarıldı.
# Ollama kütüphanesinde artık bulunmayan "qwen2-vl:7b" yerine "qwen2.5vl:7b" kondu.
# Kullanıcı bu modelleri Ollama ile kendisi kurarsa uygulama yine kullanır, listede lisans uyarısı gösterir.
# Biçim: (kod, boyut, lisans, görsel_okur, gruplar, açıklama_TR, açıklama_EN)
#   gruplar: "web"=web/makale, "ocr"=manga/webtoon görsel, "video"=video, "srt"=altyazı
MODEL_CATALOG = [
    ("translategemma:12b", "~8.1 GB", "Gemma ToU", True, ("web", "video", "srt", "ocr"),
     "🌍 Google Çeviri Uzmanı - 55 Dilde En Doğal Çeviri, Görseldeki Yazıyı da Okur",
     "🌍 Google Translation Specialist - Most Natural in 55 Langs, Also Reads Images"),
    ("translategemma:4b", "~3.3 GB", "Gemma ToU", True, ("web", "video", "srt", "ocr"),
     "⚡ Hafif Çeviri Uzmanı - Düşük VRAM'de Hızlı ve Doğal",
     "⚡ Light Translation Specialist - Fast & Natural on Low VRAM"),
    ("translategemma:27b", "~17 GB", "Gemma ToU", True, ("web", "video", "srt"),
     "👑 Stüdyo Çeviri - 55 Dilde En Yüksek Kalite",
     "👑 Studio Translation - Top Quality in 55 Langs"),
    ("qwen2.5:14b", "~9.0 GB", "Apache-2.0", False, ("web", "video", "srt"),
     "🏆 Kusursuz SRT & Anime - Uzun Cümle ve Bağlam Korumalı",
     "🏆 Flawless SRT & Anime - Best Context Preservation"),
    ("qwen2.5:32b", "~20 GB", "Apache-2.0", False, ("web", "video", "srt"),
     "👑 Stüdyo Kalitesi - Profesyonel Dizi, Anime ve Film",
     "👑 Studio Grade - Pro Series, Anime & Film"),
    ("qwen2.5:7b", "~4.7 GB", "Apache-2.0", False, ("web", "video", "srt"),
     "⚡ Dengeli & Hızlı - Doğal ve Seri Çeviri",
     "⚡ Balanced & Fast - Natural & Rapid Translation"),
    ("qwen3:8b", "~5.2 GB", "Apache-2.0", False, ("web", "video", "srt"),
     "🧠 Yeni Nesil Çok Dilli - 100+ Dil (Düşünme Kapalı Hızlı Mod)",
     "🧠 Next-Gen Multilingual - 100+ Langs (Fast, Thinking Off)"),
    ("qwen3:4b", "~2.5 GB", "Apache-2.0", False, ("web", "video", "srt"),
     "🪶 Hafif & Hızlı - Zayıf Donanımda Toplu Çeviri",
     "🪶 Light & Fast - Batch Translation on Low-End PCs"),
    ("qwen3:14b", "~9.3 GB", "Apache-2.0", False, ("web", "video", "srt"),
     "🎯 İleri Çok Dilli - Karmaşık Metin ve Diyaloglar",
     "🎯 Advanced Multilingual - Complex Text & Dialogue"),
    ("gemma2:9b", "~5.4 GB", "Gemma ToU", False, ("web", "video", "srt"),
     "🎭 Sinematik - Batı Dillerinden En Doğal Cümle Akışı",
     "🎭 Cinematic - Most Natural Western-Language Flow"),
    ("gemma2:27b", "~16 GB", "Gemma ToU", False, ("web", "video", "srt"),
     "🏛️ Edebiyat & Belgesel Pro - Uzun Cümlelerde Akıcılık",
     "🏛️ Literature & Docs Pro - Fluent Long Sentences"),
    ("gemma4:12b", "~8 GB", "Apache-2.0", True, ("web", "video", "srt", "ocr"),
     "💎 Gemma 4 - Çok Dilli + Görsel, Yüksek Kalite",
     "💎 Gemma 4 - Multilingual + Vision, High Quality"),
    ("gemma4:26b", "~18 GB", "Apache-2.0", True, ("web", "video", "srt", "ocr"),
     "🚀 Gemma 4 MoE - Büyük Model Kalitesi, Hızlı Çıkarım",
     "🚀 Gemma 4 MoE - Big-Model Quality, Fast Inference"),
    ("mistral-nemo:12b", "~7.1 GB", "Apache-2.0", False, ("web", "video", "srt"),
     "🧭 Uzun Bağlam - Karakter İsimlerini Unutmayan Model",
     "🧭 Long Context - Remembers Names & Plot"),
    ("phi4:14b", "~9.1 GB", "MIT", False, ("web", "video", "srt"),
     "🔬 Bilim & Teknik - Terimleri Doğru Çeviren Model",
     "🔬 Science & Tech - Accurate Terminology"),
    ("llama3.1:8b", "~4.9 GB", "Llama 3.1 License", False, ("web", "video", "srt"),
     "🛠️ Batı Dilleri - YouTube, Podcast ve Forumlar",
     "🛠️ Western Langs - YouTube, Podcasts & Forums"),
    ("mistral:7b", "~4.1 GB", "Apache-2.0", False, ("web", "video", "srt"),
     "🚀 Seri Batı Dilleri - Maksimum Hız",
     "🚀 Rapid Western Langs - Max Speed"),
    ("deepseek-r1:8b", "~5.2 GB", "MIT", False, ("web", "video", "srt"),
     "🧠 Analitik - Karmaşık Konular (Düşünme Kapalı)",
     "🧠 Analytical - Complex Topics (Thinking Off)"),
    ("deepseek-r1:14b", "~9.0 GB", "MIT", False, ("web", "video", "srt"),
     "🧠 Derin Bağlam - Teknik ve Felsefi Metinler",
     "🧠 Deep Context - Technical & Philosophical"),
    ("qwen2.5vl:7b", "~6.0 GB", "Apache-2.0", True, ("ocr",),
     "🎌 Manga/Manhwa OCR Kralı - Japonca/Korece/Çince Balonlar",
     "🎌 Manga/Manhwa OCR King - JP/KR/CN Bubbles"),
    ("qwen3-vl:8b-instruct", "~6.2 GB", "Apache-2.0", True, ("ocr",),
     "🔥 Yeni Nesil Görsel - Dikey Yazı ve Küçük Balonlar",
     "🔥 Next-Gen Vision - Vertical Text & Tiny Bubbles"),
    ("qwen3-vl:4b-instruct", "~3.3 GB", "Apache-2.0", True, ("ocr",),
     "⚡ Hafif Görsel OCR - Düşük VRAM",
     "⚡ Light Vision OCR - Low VRAM"),
    ("gemma3:12b", "~8.1 GB", "Gemma ToU", True, ("ocr",),
     "👁️ Çok Dilli Görsel - 140+ Dil, Dergi ve Tabelalar",
     "👁️ Multilingual Vision - 140+ Langs, Magazines & Signs"),
    ("gemma3:4b", "~3.3 GB", "Gemma ToU", True, ("ocr",),
     "🪶 Hafif Görsel - Hızlı Balon Okuma",
     "🪶 Light Vision - Fast Bubble Reading"),
    ("llama3.2-vision:11b", "~7.8 GB", "Llama 3.2 License", True, ("ocr",),
     "👁️ Yüksek Çözünürlüklü OCR - Çizgi Roman ve Tablo",
     "👁️ High-Res OCR - Comics & Tables"),
]

# Kullanıcının kendisinin kurabileceği ama ticari kullanımı kısıtlı/belirsiz modeller (listede uyarı gösterilir).
NON_COMMERCIAL_MODELS = {
    "aya-expanse": "CC-BY-NC-4.0",
    "aya": "CC-BY-NC-4.0",
    "command-r": "CC-BY-NC-4.0",
    "command-r7b": "CC-BY-NC-4.0",
    "c4ai-command-r": "CC-BY-NC-4.0",
    "qwen2.5:3b": "Qwen Research License",
    "qwen2.5vl:3b": "Qwen Research License",
    "minicpm-v": "MiniCPM Model License",
    "llava": "LLaVA (non-commercial data)",
    "llava-llama3": "LLaVA (non-commercial data)",
    "llava-phi3": "LLaVA (non-commercial data)",
    "bakllava": "LLaVA (non-commercial data)",
}

def model_license_warning(model_code):
    m = str(model_code or "").strip().lower()
    base = m.split(":")[0]
    for key, lic in NON_COMMERCIAL_MODELS.items():
        if ":" in key:
            if m == key or m.startswith(key + "-"):
                return lic
        elif base == key:
            return lic
    return None

def catalog_license(model_code):
    for code, _sz, lic, *_rest in MODEL_CATALOG:
        if code == model_code:
            return lic
    return None

_GROUP_HEADERS = {
    "web": {
        "Türkçe": ["|--- WEB SAYFASI / MAKALE / FORUM ÇEVİRİSİ ---|", "|--- MANGA / WEBTOON (OCR + GÖRSEL OKUYAN MODELLER) ---|"],
        "English": ["|--- WEB PAGE / ARTICLE / FORUM TRANSLATION ---|", "|--- MANGA / WEBTOON (OCR + VISION MODELS) ---|"],
    },
    "video": {
        "Türkçe": ["|--- VİDEO / ANİME / FİLM KONUŞMA ÇEVİRİSİ (TİCARİ KULLANIMA UYGUN LİSANSLAR) ---|"],
        "English": ["|--- VIDEO / ANIME / MOVIE SPEECH TRANSLATION (COMMERCIAL-FRIENDLY LICENSES) ---|"],
    },
    "srt": {
        "Türkçe": ["|--- PROFESYONEL DİZİ / FİLM / ANİME SRT ÇEVİRİ MODELLERİ ---|"],
        "English": ["|--- PRO SERIES / MOVIE / ANIME SRT SUBTITLE MODELS ---|"],
    },
    "manage": {
        "Türkçe": ["|--- METİN, ALTYAZI VE VİDEO MODELLERİ (İNDİR / SİL) ---|", "|--- MANGA, WEBTOON VE GÖRSEL OCR MODELLERİ (İNDİR / SİL) ---|"],
        "English": ["|--- TEXT, SUBTITLE & VIDEO MODELS (DOWNLOAD / DELETE) ---|", "|--- MANGA, WEBTOON & VISION OCR MODELS (DOWNLOAD / DELETE) ---|"],
    },
}

def _catalog_entry_text(entry, lang_key, with_size):
    code, size, lic, _vis, _groups, d_tr, d_en = entry
    desc = d_tr if lang_key == "Türkçe" else d_en
    if with_size:
        return f"{code} (📦 {size} | {desc} | 📜 {lic})"
    return f"{code} ({desc} | 📜 {lic})"

def _build_model_lists():
    web, video, srt, manage = {}, {}, {}, {}
    for lang_key in ("Türkçe", "English"):
        text_entries = [e for e in MODEL_CATALOG if "web" in e[4]]
        ocr_entries = [e for e in MODEL_CATALOG if "ocr" in e[4] and "web" not in e[4]]
        h = _GROUP_HEADERS["web"][lang_key]
        web[lang_key] = [h[0]] + [_catalog_entry_text(e, lang_key, False) for e in text_entries] + \
                        [h[1]] + [_catalog_entry_text(e, lang_key, False) for e in ocr_entries]
        video[lang_key] = _GROUP_HEADERS["video"][lang_key] + [_catalog_entry_text(e, lang_key, False) for e in MODEL_CATALOG if "video" in e[4]]
        srt[lang_key] = _GROUP_HEADERS["srt"][lang_key] + [_catalog_entry_text(e, lang_key, False) for e in MODEL_CATALOG if "srt" in e[4]]
        hm = _GROUP_HEADERS["manage"][lang_key]
        txt_only = [e for e in MODEL_CATALOG if "ocr" not in e[4] or "srt" in e[4]]
        vis_only = [e for e in MODEL_CATALOG if "ocr" in e[4] and "srt" not in e[4]]
        manage[lang_key] = [hm[0]] + [_catalog_entry_text(e, lang_key, True) for e in txt_only] + \
                           [hm[1]] + [_catalog_entry_text(e, lang_key, True) for e in vis_only]
    return web, video, srt, manage

MODELS_WEB, MODELS_VIDEO, MODELS_SRT, MODELS_MANAGE = _build_model_lists()

def _build_desc_map(lang):
    lang_key = "Türkçe" if lang == "Türkçe" else "English"
    exact_map, base_map = {}, {}
    for src_dct in (MODELS_MANAGE, MODELS_WEB, MODELS_VIDEO, MODELS_SRT):
        for entry in src_dct.get(lang_key, []):
            if entry.startswith("|"):
                continue
            parts = entry.split(" ", 1)
            if len(parts) == 2:
                code, desc = parts[0], parts[1]
                exact_map.setdefault(code, desc)
                base_map.setdefault(code.split(":")[0], desc)
    return exact_map, base_map

# Ollama'da etiketsiz kurulan modeller ":latest" olarak görünür; hangi boyuta karşılık geldiğini bilir.
LATEST_ALIASES = {
    "qwen2.5": "7b", "qwen3": "8b", "gemma2": "9b", "gemma3": "4b", "gemma4": "e4b", "translategemma": "4b",
    "llama3.1": "8b", "mistral": "7b", "mistral-nemo": "12b", "phi4": "14b", "deepseek-r1": "8b",
    "qwen2.5vl": "7b", "llama3.2-vision": "11b",
}

def _installed_alias_for(code, installed):
    base, _, tag = str(code).partition(":")
    if tag and f"{base}:latest" in installed and LATEST_ALIASES.get(base) == tag:
        return f"{base}:latest"
    return None

def get_loc_list(dct, lang, include_custom=True, allow_vision=False):
    # DÜZELTME: Eskiden önbellek boşken burada ağ taraması yapılıyordu; arayüz (UI) iş parçacığında
    # olduğu için açılışta pencere birkaç saniye donuyordu. Liste artık yalnızca önbellekten kurulur,
    # arka plandaki tarama bitince arayüz yenilenir.
    # DÜZELTME: ":latest" eşleştirmesi aynı modeli listede birden çok kez (farklı açıklamalarla) gösteriyordu.
    lang_key = "Türkçe" if lang == "Türkçe" else "English"
    raw_list = list(dct.get(lang_key, dct["Türkçe"]))
    installed = get_cached_models()
    is_manage_list = (dct is MODELS_MANAGE)
    show_headers = is_manage_list or (dct is MODELS_WEB)

    exact_map, _base_map = _build_desc_map(lang)
    final_list = []
    covered_installed = set()

    for opt in raw_list:
        if opt.startswith("|"):
            if show_headers:
                final_list.append(opt)
            continue
        code, desc = opt.split(" ", 1)
        if code in installed:
            if code in covered_installed and not is_manage_list:
                continue
            prefix = "✅ " if is_manage_list else ""
            final_list.append(f"{code} {prefix}{desc}")
            covered_installed.add(code)
            continue
        alias = _installed_alias_for(code, installed)
        if alias and alias not in covered_installed:
            prefix = "✅ " if is_manage_list else ""
            final_list.append(f"{alias} {prefix}{desc}")
            covered_installed.add(alias)
        elif is_manage_list:
            final_list.append(opt)

    if include_custom:
        custom_lbl = "(✅ Yüklü Özel Model)" if lang == "Türkçe" else "(✅ Installed Custom Model)"
        for im in installed:
            if im in covered_installed:
                continue
            if not allow_vision and is_vision_model_name(im) and not im.startswith(("translategemma", "gemma4")):
                continue
            base = im.split(":")[0]
            desc = exact_map.get(im)
            if not desc and im.endswith(":latest") and base in LATEST_ALIASES:
                desc = exact_map.get(f"{base}:{LATEST_ALIASES[base]}")
            desc = desc or custom_lbl
            warn = model_license_warning(im)
            if warn:
                desc = f"{desc} ⚠️ {'Ticari kullanım kısıtlı' if lang == 'Türkçe' else 'Restricted commercial use'}: {warn}"
            final_list.append(f"{im} {desc}")

    # Altında öğe kalmayan başlıkları temizle (manage listesinde tüm katalog gösterildiği için gerekmez).
    if show_headers and not is_manage_list:
        cleaned = []
        for idx, opt in enumerate(final_list):
            if opt.startswith("|"):
                nxt = final_list[idx + 1] if idx + 1 < len(final_list) else None
                if nxt is None or nxt.startswith("|"):
                    continue
            cleaned.append(opt)
        final_list = cleaned

    if not final_list or all(o.startswith("|") for o in final_list):
        empty_msg = "⚠️ Kurulu model yok (6. Bölümden indirin)" if lang == "Türkçe" else "⚠️ No installed models (Download from Sec. 6)"
        return [empty_msg]
    return final_list

def resolve_installed_model_name(selected_code, installed_list=None):
    # DÜZELTME: Seçilen model kurulu değilse eskiden aynı ailedeki BAŞKA bir boyut (ör. 7b yerine 32b)
    # sessizce kullanılıyordu. Artık yalnızca birebir aynı model veya ":latest" eşdeğeri kabul edilir.
    if installed_list is None:
        installed_list = get_cached_models()
    if selected_code in installed_list:
        return selected_code
    alias = _installed_alias_for(selected_code, installed_list)
    if alias:
        return alias
    if ":" not in selected_code and f"{selected_code}:latest" in installed_list:
        return f"{selected_code}:latest"
    return selected_code

def pick_best_installed_option(options_list, installed_names):
    for opt in options_list:
        if opt.startswith("|") or opt.startswith("⚠️"):
            continue
        code = opt.split(" ")[0]
        if code in installed_names:
            return opt
    for opt in options_list:
        if not opt.startswith("|"):
            return opt
    return options_list[0] if options_list else ""

# Çok dilli arayüz metinleri.
UIT = {
    "English": [
        "🌐 WEB PAGE TRANSLATION",
        "🎬 Multi-Video & Audio Translation + SRT",
        "1. Select SRT Files (Multi-Queue Supported)",
        "2. Output Folder",
        "3. Languages",
        "4. AI & Chunk Settings",
        "5. Style / Prompt",
        "6. Download / Delete Model (Size, Features & License)",
        "7. UI Language",
        "🚀 START TRANSLATION QUEUE",
        "Model:",
        "Source:",
        "Target:",
        "Select SRTs",
        "Select Folder",
        "System Logs",
        "Chunk:",
        "🔌 Shutdown PC When Done"
    ],
    "Türkçe": [
        "🌐 İNTERNET SAYFASI ÇEVİRİSİ",
        "🎬 Çoklu Video & Ses Çevirisi + SRT",
        "1. Hazır SRT Dosyaları Seçimi (Çoklu Kuyruk Destekli)",
        "2. Çıktı Klasörü Seçimi",
        "3. Çeviri Dilleri",
        "4. Yapay Zeka ve Parça Ayarları",
        "5. Çeviri Üslubu / Özel Talimat",
        "6. Model İndir / Sil (Boyut, Özellik & Lisans)",
        "7. Arayüz Dili",
        "🚀 ÇEVİRİ KUYRUĞUNU BAŞLAT",
        "Model:",
        "Kaynak:",
        "Hedef:",
        "Çoklu SRT Seç",
        "Klasör Seç",
        "Sistem Kayıtları (Log)",
        "Parça:",
        "🔌 İşlem Bittiğinde Bilgisayarı Kapat"
    ],
    "日本語": [
        "🌐 ウェブページ翻訳",
        "🎬 複数ビデオ・音声翻訳 + SRT",
        "1. SRTファイルを選択 (複数対応)",
        "2. 出力フォルダ",
        "3. 翻訳言語",
        "4. AIとチャンク設定",
        "5. 翻訳スタイル",
        "6. モデルの管理 (サイズ情報)",
        "7. UI言語",
        "🚀 翻訳キューを開始",
        "モデル:",
        "元の言語:",
        "対象言語:",
        "SRT選択",
        "フォルダ選択",
        "システムログ",
        "チャンク:",
        "🔌 完了後にPCをシャットダウン"
    ]
}

STATS_LOC = {
    "Türkçe": ["Metin:", "Görsel:", "İlerleme:"],
    "English": ["Text:", "Image:", "Progress:"],
    "日本語": ["テキスト:", "画像:", "進行:"]
}

VID_UI_LOC = {
    "Türkçe": {
        "f1": "1. Çoklu Video / Ses ve Oynatma Listesi Kuyruğu (Hızlı İndirme + Yanına 2 SRT)",
        "add_loc": "📂 Bilgisayardan Çoklu Video/Ses Ekle",
        "clear": "🗑️ Tüm Kuyruğu Temizle",
        "url_lbl": "Video veya Oynatma Listesi (Playlist) Linki:",
        "add_url": "➕ Link Ekle ve Analiz Et",
        "analyzing": "🔍 Link analiz ediliyor: {u}",
        "pl_tag": "📂 {title} (Toplam {n} Video)",
        "vid_tag": "🎬 {title}",
        "raw_tag": "🔗 {u}",
        "loc_tag": "💻 {title}",
        "f3": "3. Whisper Ses Kalitesi (GPU / CPU)",
        "f4": "4. Asenkron AI Çeviri Ayarları (Tüm Video Kuyruğunu Sırayla Eksiksiz Çevirir)",
        "chk_srt": "Çeviri SRT Dosyasını da Oluştur (Orijinal SRT Silinmez, Yan Yana Durur)",
        "prompt": "Özel Talimat:",
        "start": "🚀 TÜM VİDEO KUYRUĞUNU İŞLE",
        "shutdown": "🔌 İşlem Bittiğinde Bilgisayarı Kapat",
        "f5": "5. Canlı Video Kuyruğu Kapakları (16:9 Soldan Sağa Bulanıklık Açılma Animasyonu)",
        "dev_lbl": "Ses Motoru Cihazı:",
        "gpu_warn": "⚠️ (!) GPU seçildiğinde RAM/VRAM koruması için videolar tek tek sırayla işlenir!"
    },
    "English": {
        "f1": "1. Multi-Video / Audio & Playlist Queue (Fast Download + Dual SRT)",
        "add_loc": "📂 Add Local Video/Audio Files",
        "clear": "🗑️ Clear Entire Queue",
        "url_lbl": "Video or Playlist URL:",
        "add_url": "➕ Add & Analyze URL",
        "analyzing": "🔍 Analyzing URL: {u}",
        "pl_tag": "📂 {title} (Total {n} Videos)",
        "vid_tag": "🎬 {title}",
        "raw_tag": "🔗 {u}",
        "loc_tag": "💻 {title}",
        "f3": "3. Whisper Audio Quality (GPU / CPU)",
        "f4": "4. Async AI Translation Settings (Translates Entire Video Queue Completely)",
        "chk_srt": "Generate Translated SRT File (Original SRT is Kept Alongside)",
        "prompt": "Custom Prompt:",
        "start": "🚀 PROCESS ENTIRE VIDEO QUEUE",
        "shutdown": "🔌 Shutdown PC When Done",
        "f5": "5. Live Video Queue Thumbnails (16:9 Left-to-Right Blur-to-Clear Animation)",
        "dev_lbl": "Speech Engine Device:",
        "gpu_warn": "⚠️ (!) When GPU is selected, videos are processed one by one to protect RAM/VRAM!"
    },
    "日本語": {
        "f1": "1. 複数動画・音声・再生リストキュー (高速DL + 2つのSRT出力)",
        "add_loc": "📂 PCから動画/音声を追加",
        "clear": "🗑️ キューをすべて消去",
        "url_lbl": "動画または再生リストのURL:",
        "add_url": "➕ URLを追加して解析",
        "analyzing": "🔍 URLを解析中: {u}",
        "pl_tag": "📂 {title} (全 {n} 本の動画)",
        "vid_tag": "🎬 {title}",
        "raw_tag": "🔗 {u}",
        "loc_tag": "💻 {title}",
        "f3": "3. Whisper 音声品質 (GPU / CPU)",
        "f4": "4. 非同期AI翻訳設定 (音声認識とAI翻訳の処理)",
        "chk_srt": "翻訳SRTファイルも作成する (元のSRTも保持されます)",
        "prompt": "カスタム指示:",
        "start": "🚀 すべての動画キューを処理開始",
        "shutdown": "🔌 完了後にPCをシャットダウン",
        "f5": "5. ライブ動画サムネイル (16:9 進行アニメーション)",
        "dev_lbl": "音声認識デバイス:",
        "gpu_warn": "⚠️ (!) GPU選択時はメモリ保護のため1つずつ順番に処理されます！"
    }
}

CARD_LOC = {
    "Türkçe": {
        "wait": "⏳ Kuyrukta Bekliyor...",
        "dl": "📥 İndiriliyor: %{p} | {sp}",
        "wh_Wait": "⏳ Ses Analizi Sırasında...",
        "wh": "🎙️ Sesten Yazıya [{dev}]: %{p} | Satır: {ln}",
        "ai_wait": "⏳ AI Çeviri Sırasında...",
        "ai": "🌐 AI Çeviriyor: %{p} ({c}/{t})",
        "retry": "🔄 Bağlantı bekleniyor / Tekrar deneniyor ({c}/{t})...",
        "done": "✅ Tamamlandı (%100)",
        "err": "❌ Hata: {e}"
    },
    "English": {
        "wait": "⏳ Waiting in Queue...",
        "dl": "📥 Downloading: {p}% | {sp}",
        "wh_Wait": "⏳ Queued for Audio Analysis...",
        "wh": "🎙️ Speech-to-Text [{dev}]: {p}% | Line: {ln}",
        "ai_wait": "⏳ Queued for AI Translation...",
        "ai": "🌐 AI Translating: {p}% ({c}/{t})",
        "retry": "🔄 Waiting for AI / Retrying chunk ({c}/{t})...",
        "done": "✅ Completed (100%)",
        "err": "❌ Error: {e}"
    },
    "日本語": {
        "wait": "⏳ 待機中...",
        "dl": "📥 ダウンロード中: {p}% | {sp}",
        "wh_Wait": "⏳ 音声解析待機中...",
        "wh": "🎙️ 音声認識 [{dev}]: {p}% | 行: {ln}",
        "ai_wait": "⏳ AI翻訳待機中...",
        "ai": "🌐 AI翻訳中: {p}% ({c}/{t})",
        "retry": "🔄 AI再試行中 ({c}/{t})...",
        "done": "✅ 完了 (100%)",
        "err": "❌ エラー: {e}"
    }
}

# Yeni eklenen arayüz metinleri (durdurma butonları, lisans uyarıları vb.).
EXTRA_UI = {
    "Türkçe": {
        "stop": "⏹ DURDUR",
        "stopping": "⏳ Durduruluyor...",
        "licenses": "📜 Lisanslar",
        "lic_title": "Lisans Uyarısı",
        "lic_warn": "'{m}' modelinin lisansı ({lic}) ticari kullanımı kısıtlar.\n\nBu modeli ücretli ürününüzle birlikte önermeniz/dağıtmanız lisans ihlali olabilir. Yalnızca kişisel/ticari olmayan kullanım içindir.\n\nYine de indirilsin mi?",
        "no_model": "Lütfen geçerli ve kurulu bir model seçin.",
        "model_missing": "Seçilen model Ollama'da bulunamadı: {m}\nModeli 6. bölümden indirin.",
        "notices_missing": "THIRD_PARTY_NOTICES.txt dosyası uygulama klasöründe bulunamadı.",
        "gpu_install_frozen": "GPU kütüphaneleri paketlenmiş sürümde otomatik kurulamaz.\nLütfen NVIDIA sürücünüzü güncelleyin ve uygulamanın GPU paketini kullanın.",
        "gpu_installing": "⏳ GPU kütüphaneleri arka planda kuruluyor (birkaç dakika sürebilir)...",
        "gpu_install_done": "✅ GPU kütüphaneleri kuruldu.",
        "gpu_install_fail": "❌ GPU kütüphaneleri kurulamadı: {e}",
    },
    "English": {
        "stop": "⏹ STOP",
        "stopping": "⏳ Stopping...",
        "licenses": "📜 Licenses",
        "lic_title": "License Warning",
        "lic_warn": "The license of '{m}' ({lic}) restricts commercial use.\n\nOffering/distributing it with your paid product may violate its license. It is for personal / non-commercial use only.\n\nDownload anyway?",
        "no_model": "Please select a valid installed model.",
        "model_missing": "The selected model was not found in Ollama: {m}\nDownload it from section 6.",
        "notices_missing": "THIRD_PARTY_NOTICES.txt was not found in the application folder.",
        "gpu_install_frozen": "GPU libraries cannot be installed automatically in the packaged build.\nPlease update your NVIDIA driver and use the GPU package of the app.",
        "gpu_installing": "⏳ Installing GPU libraries in the background (may take a few minutes)...",
        "gpu_install_done": "✅ GPU libraries installed.",
        "gpu_install_fail": "❌ GPU libraries could not be installed: {e}",
    },
}

def xui(lang, key, **fmt):
    tbl = EXTRA_UI.get(lang) or EXTRA_UI["English"]
    txt = tbl.get(key) or EXTRA_UI["English"].get(key, key)
    return txt.format(**fmt) if fmt else txt

def find_notices_file():
    names = ("THIRD_PARTY_NOTICES.txt", "THIRD_PARTY_NOTICES.md")
    dirs = []
    if getattr(sys, 'frozen', False):
        dirs += [os.path.dirname(sys.executable), getattr(sys, '_MEIPASS', "")]
    try:
        dirs.append(os.path.dirname(os.path.abspath(__file__)))
    except Exception:
        pass
    dirs.append(os.getcwd())
    for d in dirs:
        for n in names:
            p = os.path.join(d, n) if d else ""
            if p and os.path.isfile(p):
                return p
    return None

def open_path_with_os(path):
    try:
        if os.name == "nt":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        return True
    except Exception:
        return False

BG, FRM, TXT, LBL, BTN = "#121212", "#1e1e1e", "#ffffff", "#cccccc", "#333333"

def set_dark(rt):
    s = ttk.Style(rt)
    s.theme_use("clam")
    s.configure("TFrame", background=BG)
    s.configure("TLabelframe", background=BG, foreground=TXT)
    s.configure("TLabelframe.Label", background=BG, foreground=TXT, font=("Segoe UI", 10, "bold"))
    s.configure("TLabel", background=BG, foreground=LBL)
    s.configure("TRadiobutton", background=BG, foreground=LBL)
    s.configure("TCheckbutton", background=BG, foreground=LBL)
    s.configure("TButton", background=BTN, foreground=TXT, font=("Segoe UI", 9))
    s.map("TButton", background=[("active", "#555555")])
    rt.configure(bg=BG)

def get_icon_path():
    candidates = [
        os.path.join(os.getcwd(), "simge.ico"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "simge.ico")
    ]
    if getattr(sys, 'frozen', False):
        candidates.insert(0, os.path.join(getattr(sys, '_MEIPASS', os.path.dirname(sys.executable)), "simge.ico"))
        candidates.insert(1, os.path.join(os.path.dirname(sys.executable), "simge.ico"))
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def set_icon(win):
    try:
        ic = get_icon_path()
        if ic:
            win.iconbitmap(ic)
    except:
        pass

def load_header_icon_photo(size=(28, 28)):
    if not PIL_OK:
        return None
    try:
        ic = get_icon_path()
        if ic and os.path.exists(ic):
            im = Image.open(ic).convert("RGBA").resize(size, Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(im)
    except:
        pass
    return None

# Altyazı zamanlama ve segment daraltma motoru.
def format_srt_time(sec):
    # DÜZELTME: Milisaniye 1000'e yuvarlanınca saniye artmıyordu (ör. 1.9996 -> 00:00:01,999 yerine 00:00:02,000).
    total_ms = max(0, int(round(float(sec) * 1000.0)))
    h, rem = divmod(total_ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def parse_srt_time_to_sec(t_str):
    try:
        t_str = t_str.strip().replace(".", ",")
        hms, ms = t_str.split(",")
        h, m, s = hms.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms[:3].ljust(3, "0")) / 1000.0
    except:
        return None

def clamp_srt_timestamp_line(ts_line, text_content):
    if "-->" not in ts_line:
        return ts_line
    parts = ts_line.split("-->")
    if len(parts) != 2:
        return ts_line
    st = parse_srt_time_to_sec(parts[0])
    en = parse_srt_time_to_sec(parts[1])
    if st is None or en is None or en <= st:
        return ts_line
    char_len = len(re.sub(r"\s+", " ", text_content.strip()))
    max_allowed = min(8.0, max(2.6, 1.6 + char_len * 0.11))
    if (en - st) > max_allowed:
        en = st + max_allowed
    return f"{format_srt_time(st)} --> {format_srt_time(en)}"

def build_tight_whisper_segments(whisper_segment, filler_re):
    words = getattr(whisper_segment, "words", None)
    results = []

    if words:
        valid_words = [w for w in words if getattr(w, "word", "").strip()]
        if valid_words:
            chunks = []
            cur_w = []
            for w in valid_words:
                if not cur_w:
                    cur_w.append(w)
                    continue
                gap = max(0.0, float(w.start) - float(cur_w[-1].end))
                cur_dur = float(w.end) - float(cur_w[0].start)
                cur_chars = sum(len(x.word) for x in cur_w)
                if gap >= 1.15 or cur_dur >= 5.5 or cur_chars >= 85:
                    chunks.append(cur_w)
                    cur_w = [w]
                else:
                    cur_w.append(w)
            if cur_w:
                chunks.append(cur_w)

            for idx, ch in enumerate(chunks):
                txt = "".join(x.word for x in ch).strip()
                txt = re.sub(r"\s+", " ", txt)
                if not txt or filler_re.match(txt):
                    continue
                st_sec = max(0.0, float(ch[0].start) - 0.05)
                actual_word_end = float(ch[-1].end)
                min_read_dur = min(6.5, max(1.5, 1.0 + len(txt) * 0.08))
                en_sec = max(actual_word_end + 0.45, st_sec + min_read_dur)
                en_sec = min(en_sec, actual_word_end + 1.65)

                if idx + 1 < len(chunks):
                    next_start = max(0.0, float(chunks[idx + 1][0].start))
                    if en_sec > next_start - 0.02:
                        en_sec = max(actual_word_end, next_start - 0.02)
                results.append((st_sec, en_sec, txt))
            return results

    tx = whisper_segment.text.strip()
    if not tx or filler_re.match(tx):
        return []
    st_sec = max(0.0, float(whisper_segment.start))
    seg_end = float(whisper_segment.end)
    max_dur = min(7.5, max(2.2, 1.4 + len(tx) * 0.09))
    en_sec = min(seg_end + 0.35, max(seg_end, st_sec + max_dur))
    return [(st_sec, en_sec, tx)]

# ASS/HTML etiketlerini ({\an8}, <font>) temizleyen SRT ayrıştırıcı ve satır korumalı çeviri motoru.
_TS_LINE_RE = re.compile(r"^\s*(\d{1,2}:\d{1,2}:\d{1,2}[,.:]\d{1,3})\s*-->\s*(\d{1,2}:\d{1,2}:\d{1,2}[,.:]\d{1,3})(.*)$")
_ASS_TAG_RE = re.compile(r"\{\\[^}]*\}")
_HTML_TAG_RE = re.compile(r"</?(?:font|i|b|u|s)\b[^>]*>", re.IGNORECASE)

def clean_subtitle_tags(txt_str):
    s = _ASS_TAG_RE.sub("", str(txt_str or ""))
    s = _HTML_TAG_RE.sub("", s)
    return s.strip()

def normalize_srt_timestamp_line(ts_line):
    m = _TS_LINE_RE.match(ts_line or "")
    if not m:
        return (ts_line or "").strip()
    def _fix(t):
        t = t.strip()
        parts = re.split(r"[:,.]", t)
        if len(parts) != 4:
            return t
        h, mi, s, ms = parts
        return f"{int(h):02d}:{int(mi):02d}:{int(s):02d},{ms[:3].ljust(3, '0')}"
    tail = (m.group(3) or "").strip()
    return f"{_fix(m.group(1))} --> {_fix(m.group(2))}" + (f" {tail}" if tail else "")

def _extract_subtitle_style(raw_txt):
    raw = str(raw_txt or "").strip()
    m = re.match(r"^\s*(\{\\an\d\})", raw)
    pos_tag = m.group(1) if m else ""
    body = _ASS_TAG_RE.sub("", raw).strip()
    italic = bool(re.match(r"^<i>", body, re.I) and re.search(r"</i>$", body, re.I) and body.lower().count("<i>") == 1)
    lines = [ln for ln in clean_subtitle_tags(raw).split("\n") if ln.strip()]
    dialogue = len(lines) >= 2 and all(ln.lstrip().startswith(("-", "–", "—")) for ln in lines)
    return {"pos": pos_tag, "italic": italic, "lines": max(1, len(lines)), "dialogue": dialogue}

def parse_srt_blocks(raw_srt, clamp=False):
    """SRT metnini (sıra, zaman, temiz_metin, stil) demetlerine ayırır.
    DÜZELTME 1: Orijinal zaman damgaları artık KORUNUR. Eski sürüm her altyazının süresini metin
    uzunluğuna göre kısaltıyordu (clamp); çevrilen dosyanın zamanlaması bozuluyordu. (clamp=True ile eski davranış)
    DÜZELTME 2: Blok arasında boş satır olmayan bozuk SRT dosyaları artık doğru ayrıştırılır."""
    norm_srt = str(raw_srt or "").replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")
    lines = norm_srt.split("\n")
    ts_idx = [k for k, ln in enumerate(lines) if _TS_LINE_RE.match(ln)]
    parsed = []
    seq = 1
    for j, k in enumerate(ts_idx):
        end = ts_idx[j + 1] if j + 1 < len(ts_idx) else len(lines)
        if j + 1 < len(ts_idx):
            p = end - 1
            while p > k and not lines[p].strip():
                p -= 1
            if p > k and lines[p].strip().isdigit():
                prev_blank = (p - 1 <= k) or not lines[p - 1].strip()
                cur_num = lines[k - 1].strip() if k > 0 else ""
                expected = cur_num.isdigit() and int(lines[p].strip()) == int(cur_num) + 1
                if prev_blank or expected:
                    end = p
        text_lines = [ln.strip() for ln in lines[k + 1:end] if ln.strip()]
        if not text_lines:
            continue
        raw_text = "\n".join(text_lines)
        txt_content = clean_subtitle_tags(raw_text)
        if not txt_content:
            continue
        ts_line = normalize_srt_timestamp_line(lines[k])
        if clamp:
            ts_line = clamp_srt_timestamp_line(ts_line, txt_content)
        parsed.append((seq, ts_line, txt_content, _extract_subtitle_style(raw_text)))
        seq += 1
    return parsed

def _has_translatable_letters(text_str):
    return bool(re.search(r"[^\W\d_]", str(text_str or ""), flags=re.UNICODE))

def _norm_cmp(s):
    return re.sub(r"\s+", " ", str(s or "").strip()).lower()

def _is_line_untranslated(orig_txt, cand_txt, src_lang=None, tgt_lang=None):
    if not cand_txt or not str(cand_txt).strip():
        return True
    if src_lang and tgt_lang and src_lang == tgt_lang:
        return False
    o_clean, c_clean = _norm_cmp(orig_txt), _norm_cmp(cand_txt)
    return o_clean == c_clean and len(o_clean) > 3 and _has_translatable_letters(o_clean)

_LABEL_LINE_RE = re.compile(r"^\s*(?:sure|of course|here(?:'s| is| are)|translation|translated text|çeviri|tercüme|türkçe(?: çeviri)?|output)\b[^\n:]{0,60}:\s*$", re.IGNORECASE)
_NOTE_LINE_RE = re.compile(r"^\s*\(?\s*(?:note|not|explanation|açıklama|t/n|tn|translator'?s? note|çevirmen notu)\b", re.IGNORECASE)

def clean_model_output(text, single_line=True):
    """Model yanıtını temizler: düşünce blokları, 'Translation:' gibi etiketler, tırnaklar.
    DÜZELTME: Eskiden **kalın** yazılmış çeviri tamamen siliniyordu; artık yalnızca yıldızlar kaldırılır."""
    t = strip_reasoning(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t, flags=re.DOTALL)
    t = re.sub(r"^\s*(?:translation|çeviri|tercüme)\s*[:：]\s*", "", t, flags=re.IGNORECASE)
    out_lines = []
    for ln in t.split("\n"):
        s = ln.strip()
        if not s or _LABEL_LINE_RE.match(s) or _NOTE_LINE_RE.match(s):
            continue
        s = re.sub(r"^\s*(?:\[\d{1,4}\]|<<<\d{1,4}>>>|\(\d{1,4}\))\s*", "", s)
        out_lines.append(s)
    if single_line:
        t = " ".join(out_lines)
    else:
        t = "\n".join(out_lines)
    t = t.strip().strip('"\'«»“”„`').strip()
    return t

_NUM_LINE_RE = re.compile(r"^\s*(?:\[|\(|<<<|【)?\s*(\d{1,4})\s*(?:\]|\)|>>>|】|\.|:|：|-)\s*(.*)$")

def _parse_numbered_response(resp_text, count, global_offset=None):
    got = {}
    unnumbered = []
    for line in str(resp_text or "").split("\n"):
        s = line.strip()
        if not s:
            continue
        m = _NUM_LINE_RE.match(s)
        if m:
            k = int(m.group(1))
            val = clean_model_output(m.group(2))
            if not val:
                continue
            if 1 <= k <= count:
                got.setdefault(k, val)
            elif global_offset and 1 <= (k - global_offset) <= count:
                got.setdefault(k - global_offset, val)
        elif not _LABEL_LINE_RE.match(s):
            unnumbered.append(clean_model_output(s))
    if not got and len(unnumbered) == count:
        got = {i: v for i, v in enumerate(unnumbered, 1) if v}
    return got

def _lang_code_for(name, default="auto"):
    try:
        return L[name][1]
    except Exception:
        return default

def _translategemma_messages(model, text, src_lang, tgt_lang, custom_prompt="", numbered_count=0, images=None):
    s_name = src_lang if src_lang and src_lang != "Auto" else "multilingual (auto-detected)"
    s_code = _lang_code_for(src_lang) if src_lang and src_lang != "Auto" else "auto"
    t_code = _lang_code_for(tgt_lang, "en")
    extra = ""
    if numbered_count:
        extra += f" The text has {numbered_count} numbered lines; keep every [number] prefix exactly and output exactly one line per number."
    if custom_prompt:
        extra += f" Style: {custom_prompt}."
    what = "the text in the image" if images else f"the following {s_name} text"
    content = (f"You are a professional {s_name} ({s_code}) to {tgt_lang} ({t_code}) translator. "
               f"Your goal is to accurately convey the meaning and nuances of the original {s_name} text "
               f"while adhering to {tgt_lang} grammar, vocabulary, and cultural sensitivities.{extra} "
               f"Produce only the {tgt_lang} translation, without any additional explanations or commentary. "
               f"Please translate {what} into {tgt_lang}:\n\n\n{text}")
    msg = {"role": "user", "content": content}
    if images:
        msg["images"] = images
    return [msg]

def _is_translategemma(model):
    return str(model or "").lower().startswith("translategemma")

def _batch_messages(model, payload_text, count, src_lang, tgt_lang, custom_prompt, purpose):
    if _is_translategemma(model):
        return _translategemma_messages(model, payload_text, src_lang, tgt_lang, custom_prompt, numbered_count=count)
    if src_lang == "Auto":
        lang_instruction = (f"The input lines may contain MULTIPLE different languages mixed together. "
                            f"Detect the language of EACH line individually and translate EVERY single line completely into {tgt_lang}.")
    else:
        lang_instruction = f"Translate every line from {src_lang} (and any mixed foreign phrases) completely into {tgt_lang}."
    role = {
        "subtitle": "You are a professional multilingual subtitle translator.",
        "manga": ("You are a professional manga / manhwa / webtoon translator. Each line is the OCR text of one speech "
                  "bubble or caption; it may contain small OCR mistakes, fix them silently. Use natural, conversational language."),
        "web": "You are a professional multilingual web page translator. Keep names, numbers and links unchanged.",
    }.get(purpose, "You are a professional multilingual translator.")
    sys_prompt = (
        f"{role} {lang_instruction}\n"
        "CRITICAL RULES:\n"
        f"1. Keep the exact [1] to [{count}] prefix at the start of each line.\n"
        f"2. Translate EVERY line into {tgt_lang}. Never skip or merge lines.\n"
        "3. Output ONLY the translated lines with their [number] prefix."
    )
    if custom_prompt:
        sys_prompt += f"\n4. CONTEXT / STYLE INSTRUCTION: {custom_prompt}"
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": payload_text}]

def _single_messages(model, text, src_lang, tgt_lang, custom_prompt, purpose, keep_lines):
    if _is_translategemma(model):
        return _translategemma_messages(model, text, src_lang, tgt_lang, custom_prompt)
    what = {"subtitle": "subtitle line", "manga": "manga speech bubble text (may contain OCR mistakes)", "web": "web page text"}.get(purpose, "text")
    src_part = f" from {src_lang}" if src_lang and src_lang != "Auto" else ""
    sys_s = (f"You are a professional translator. Translate the user's {what}{src_part} into natural {tgt_lang}. "
             + ("Keep the paragraph structure. " if keep_lines else f"Output ONLY the final {tgt_lang} translation on a single line ")
             + "without quotes, prefixes, notes or explanations.")
    if custom_prompt:
        sys_s += f" Style: {custom_prompt}"
    return [{"role": "system", "content": sys_s}, {"role": "user", "content": text}]

def translate_single_text(orig_txt, real_model, src_lang, tgt_lang, custom_prompt="", session=None, abort_check=None,
                          status_cb=None, purpose="subtitle", keep_lines=False, max_attempts=4):
    sess = session or HTTP
    if keep_lines:
        clean_in = clean_subtitle_tags(orig_txt).strip()
    else:
        clean_in = clean_subtitle_tags(orig_txt).replace("\n", " ").strip()
    if not _has_translatable_letters(clean_in):
        return clean_in
    messages = _single_messages(real_model, clean_in, src_lang, tgt_lang, custom_prompt, purpose, keep_lines)
    echo_count = 0
    last_ans = ""
    for retry in range(max_attempts):
        if abort_check and abort_check():
            return orig_txt
        try:
            ans = ollama_chat(real_model, messages, session=sess,
                              options={"temperature": min(0.4, 0.1 + retry * 0.08), "repeat_penalty": 1.1},
                              timeout=180)
            ans = clean_model_output(ans, single_line=not keep_lines)
            if ans:
                last_ans = ans
                if not _is_line_untranslated(clean_in, ans, src_lang, tgt_lang):
                    return ans
                echo_count += 1
                if echo_count >= 2:
                    return ans  # metin zaten hedef dilde olabilir; sonsuz tekrar yapma
        except OllamaModelMissing:
            raise
        except Exception:
            if abort_check and abort_check():
                return orig_txt
            wait_until_ollama_ready(abort_check=abort_check, status_cb=status_cb, max_wait_sec=45)
            time.sleep(1.0)
    return last_ans or orig_txt

_translate_single_line_guaranteed = translate_single_text  # eski adla uyumluluk

def _estimate_tokens(text):
    s = str(text or "")
    wide = sum(1 for ch in s if ord(ch) > 0x2E80)
    return int((len(s) - wide) / 3.5) + wide + 4

def _split_by_token_budget(texts, budget=1400, max_items=100):
    """Ollama'nın varsayılan bağlam penceresi (num_ctx) taşmasın diye büyük parçaları böler;
    taşma olunca model son satırları sessizce atıyordu."""
    groups, cur, cur_tok = [], [], 0
    for i, t in enumerate(texts):
        tk = _estimate_tokens(t)
        if cur and (cur_tok + tk > budget or len(cur) >= max_items):
            groups.append(cur)
            cur, cur_tok = [], 0
        cur.append(i)
        cur_tok += tk
    if cur:
        groups.append(cur)
    return groups

def translate_lines_batch(texts, real_model, src_lang, tgt_lang, custom_prompt="", session=None, abort_check=None,
                          status_cb=None, purpose="subtitle", max_attempts=6, single_fallback=True, global_offset=None):
    """Satır listesini numaralı tek istekle çevirir; eksik veya çevrilmemiş satırları tek tek tamamlar.
    Dönüş: girdiyle aynı uzunlukta liste (çevrilemeyen satır için orijinal metin)."""
    sess = session or HTTP
    n = len(texts)
    if n == 0:
        return []
    clean = [clean_subtitle_tags(t).replace("\n", " ").strip() for t in texts]
    results = [None] * n
    todo = [i for i in range(n) if _has_translatable_letters(clean[i])]
    for i in range(n):
        if i not in todo:
            results[i] = clean[i]

    for group in _split_by_token_budget([clean[i] for i in todo]):
        if abort_check and abort_check():
            break
        idxs = [todo[g] for g in group]
        if len(idxs) == 1 and single_fallback:
            continue  # tek satır -> doğrudan tekli çeviriye
        payload = "\n".join(f"[{k}] {clean[i]}" for k, i in enumerate(idxs, 1))
        messages = _batch_messages(real_model, payload, len(idxs), src_lang, tgt_lang, custom_prompt, purpose)
        got = {}
        attempt = 0
        while attempt < max_attempts:
            if abort_check and abort_check():
                break
            attempt += 1
            try:
                resp = ollama_chat(real_model, messages, session=sess,
                                   options={"temperature": min(0.35, 0.1 + (attempt - 1) * 0.05), "repeat_penalty": 1.1},
                                   timeout=300)
                for k, v in _parse_numbered_response(resp, len(idxs), global_offset).items():
                    if not _is_line_untranslated(clean[idxs[k - 1]], v, src_lang, tgt_lang):
                        got.setdefault(k, v)
                if len(got) >= max(1, int(len(idxs) * 0.7)):
                    break
            except OllamaModelMissing:
                raise
            except Exception:
                if abort_check and abort_check():
                    break
                wait_until_ollama_ready(abort_check=abort_check, status_cb=status_cb, max_wait_sec=60)
                time.sleep(1.0)
        for k, i in enumerate(idxs, 1):
            if got.get(k):
                results[i] = got[k]

    for i in range(n):
        if results[i] is not None:
            continue
        if abort_check and abort_check():
            results[i] = clean[i]
            continue
        if single_fallback:
            results[i] = translate_single_text(clean[i], real_model, src_lang, tgt_lang, custom_prompt=custom_prompt,
                                               session=sess, abort_check=abort_check, status_cb=status_cb, purpose=purpose)
        else:
            results[i] = clean[i]
    return [r if (r is not None and str(r).strip()) else clean[i] for i, r in enumerate(results)]

def _rebalance_subtitle_lines(text, meta, max_line=42):
    t = str(text or "").strip()
    if not meta or "\n" in t:
        return t
    if meta.get("dialogue"):
        parts = [p.strip() for p in re.split(r"\s+(?=[-–—]\s*\S)", t) if p.strip()]
        if len(parts) == meta.get("lines", 2):
            return "\n".join(parts)
    if meta.get("lines", 1) < 2 or len(t) <= max_line:
        return t
    mid = len(t) // 2
    best = None
    for i, ch in enumerate(t):
        if ch == " " and (best is None or abs(i - mid) < abs(best - mid)):
            best = i
    if best is None:
        return t
    return t[:best].rstrip() + "\n" + t[best + 1:].lstrip()

def _apply_subtitle_style(text, meta):
    out = str(text or "")
    if meta and meta.get("italic"):
        out = f"<i>{out}</i>"
    if meta and meta.get("pos"):
        out = meta["pos"] + out
    return out

def translate_srt_chunk_locked(chunk_items, real_model, src_lang, tgt_lang, custom_prompt="", session=None, abort_check=None, status_cb=None):
    texts = [it[2] for it in chunk_items]
    offset = None
    try:
        offset = int(chunk_items[0][0]) - 1 if chunk_items else None
    except Exception:
        offset = None
    outs = translate_lines_batch(texts, real_model, src_lang, tgt_lang, custom_prompt, session=session,
                                 abort_check=abort_check, status_cb=status_cb, purpose="subtitle", global_offset=offset)
    rebuilt_blocks = []
    for item, tr_txt in zip(chunk_items, outs):
        seq_id, ts_line, orig_txt = item[0], item[1], item[2]
        meta = item[3] if len(item) > 3 and isinstance(item[3], dict) else {}
        final_txt = tr_txt.strip() if tr_txt and tr_txt.strip() else orig_txt
        final_txt = _apply_subtitle_style(_rebalance_subtitle_lines(final_txt, meta), meta)
        rebuilt_blocks.append(f"{seq_id}\n{ts_line}\n{final_txt}")
    return "\n\n".join(rebuilt_blocks)

# [Open-source edition] Manga/Webtoon OCR and live-web translation modules are removed here.
# This edition provides SRT translation and video-to-SRT (transcribe + translate) only.

# Çoklu Video ve Ses Çevirmeni (Uzun Başlık Korumalı İndirici + FFmpeg WAV Yedekli Whisper + Çift SRT).
class VideoTranslator:
    def __init__(self, rt, ui_lang="English"):
        self.w = tk.Toplevel(rt)
        self.w.title("🎬 AI Local Translator - Multi-Video & Audio Translator (GPU / CPU)")
        self.w.geometry("1080x940")
        self.w.minsize(920, 760)
        set_dark(self.w)
        set_icon(self.w)
        self.ui = ui_lang
        self.op = ""
        self.groups = []
        self.jobs = []
        self.next_gid = 1
        self.running = False
        self.abort_flag = False
        self.resize_after_id = None
        self.last_canvas_w = 0
        self.var_shutdown_vid = tk.BooleanVar(value=False)
        self.var_shutdown_vid.trace_add("write", self._on_shutdown_toggle_live)
        self.whisper_dev_mode = tk.StringVar(value="CPU")
        if not VA:
            messagebox.showerror("Error", "yt-dlp and faster-whisper libraries are missing.")
            self.w.destroy()
            return
        self.w.protocol("WM_DELETE_WINDOW", self.on_close_window)
        self.bui()

    def _on_shutdown_toggle_live(self, *args):
        if not self.var_shutdown_vid.get():
            cancel_pc_shutdown()

    def on_close_window(self):
        self.abort_flag = True
        self.running = False
        self.var_shutdown_vid.set(False)
        cancel_pc_shutdown()
        TASKBAR.clear(self.w)
        threading.Thread(target=lambda: unload_all_ollama_models(ACTIVE_HOST, fast_exit=True), daemon=True).start()
        self.w.destroy()

    def on_device_mode_change(self, *args):
        vloc = VID_UI_LOC.get(self.ui, VID_UI_LOC["English"])
        if self.whisper_dev_mode.get() == "GPU":
            ok, msg = verify_gpu_ready()
            if not ok:
                # DÜZELTME: Paketlenmiş exe'de sys.executable uygulamanın kendisidir; "exe -m pip" uygulamanın
                # ikinci bir kopyasını açıyordu. Ayrıca kurulum arayüz iş parçacığında dakikalarca donma yapıyordu.
                if getattr(sys, 'frozen', False):
                    messagebox.showwarning("GPU", f"{msg}\n\n{xui(self.ui, 'gpu_install_frozen')}", parent=self.w)
                    self.whisper_dev_mode.set("CPU")
                    self.lbl_gpu_warn.config(text="")
                    return
                ans = messagebox.askyesno(
                    "GPU Libraries Missing",
                    f"CUDA components required for GPU acceleration were not found:\n({msg})\n\n"
                    "Install nvidia-cublas-cu12 and nvidia-cudnn-cu12 automatically now?",
                    parent=self.w
                )
                if not ans:
                    self.whisper_dev_mode.set("CPU")
                    self.lbl_gpu_warn.config(text="")
                    return
                self.lbl_gpu_warn.config(text=xui(self.ui, "gpu_installing"))

                def _bg_install():
                    try:
                        p = subprocess.run([sys.executable, "-m", "pip", "install", "nvidia-cublas-cu12", "nvidia-cudnn-cu12"],
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1800, **_hidden_proc_kwargs())
                        register_cuda_dlls()
                        txt = xui(self.ui, "gpu_install_done") if p.returncode == 0 else xui(self.ui, "gpu_install_fail", e=p.returncode)
                    except Exception as e:
                        txt = xui(self.ui, "gpu_install_fail", e=str(e)[:60])
                    if self.w.winfo_exists():
                        self.w.after(0, lambda: self.lbl_gpu_warn.config(text=txt + "  " + vloc["gpu_warn"]))
                threading.Thread(target=_bg_install, daemon=True).start()
                return
            self.lbl_gpu_warn.config(text=vloc["gpu_warn"])
        else:
            self.lbl_gpu_warn.config(text="")

    def bui(self):
        t = UIT.get(self.ui, UIT["English"])
        vloc = VID_UI_LOC.get(self.ui, VID_UI_LOC["English"])

        self.w.grid_columnconfigure(0, weight=1)
        self.w.grid_rowconfigure(1, weight=2, minsize=145)
        self.w.grid_rowconfigure(5, weight=6, minsize=340)

        dev_bar = tk.Frame(self.w, bg="#161925", highlightthickness=1, highlightbackground="#2b3248")
        dev_bar.grid(row=0, column=0, sticky="ew", padx=12, pady=(6, 2))
        tk.Label(dev_bar, text=vloc["dev_lbl"], bg="#161925", fg="#00aaff", font=("Segoe UI", 9, "bold")).pack(side="left", padx=(10, 6), pady=5)
        tk.Radiobutton(
            dev_bar, text="🖥️ CPU", variable=self.whisper_dev_mode, value="CPU",
            bg="#161925", fg="#ffffff", selectcolor=FRM, activebackground="#161925",
            font=("Segoe UI", 9, "bold"), command=self.on_device_mode_change
        ).pack(side="left", padx=4)
        tk.Radiobutton(
            dev_bar, text="🎮 GPU (NVIDIA CUDA)", variable=self.whisper_dev_mode, value="GPU",
            bg="#161925", fg="#00e676", selectcolor=FRM, activebackground="#161925",
            font=("Segoe UI", 9, "bold"), command=self.on_device_mode_change
        ).pack(side="left", padx=6)
        self.lbl_gpu_warn = tk.Label(dev_bar, text="", bg="#161925", fg="#ffb74d", font=("Segoe UI", 9, "bold"))
        self.lbl_gpu_warn.pack(side="left", padx=10)

        sf = ttk.LabelFrame(self.w, text=vloc["f1"])
        sf.grid(row=1, column=0, sticky="nsew", padx=12, pady=3)

        top_bar = tk.Frame(sf, bg=BG)
        top_bar.pack(fill="x", padx=8, pady=3)
        ttk.Button(top_bar, text=vloc["add_loc"], command=self.add_local_files).pack(side="left")
        ttk.Label(top_bar, text=vloc["url_lbl"]).pack(side="left", padx=(12, 4))
        self.url_ent = tk.Entry(top_bar, bg=FRM, fg=TXT, insertbackground=TXT, font=("Segoe UI", 9))
        self.url_ent.pack(side="left", fill="x", expand=True, padx=4, ipady=2)
        self.url_ent.bind("<Return>", lambda e: self.add_url_from_entry())
        self.url_ent.bind("<Control-v>", self._on_url_paste)
        ttk.Button(top_bar, text=vloc["add_url"], command=self.add_url_from_entry).pack(side="left", padx=4)
        ttk.Button(top_bar, text=vloc["clear"], command=self.clear_queue).pack(side="right")

        q_box_frame = tk.Frame(sf, bg="#161616", bd=1, relief="sunken")
        q_box_frame.pack(fill="both", expand=True, padx=8, pady=(2, 5))
        self.chip_canvas = tk.Canvas(q_box_frame, bg="#161616", height=96, highlightthickness=0)
        self.chip_scroll = ttk.Scrollbar(q_box_frame, orient="vertical", command=self.chip_canvas.yview)
        self.chip_canvas.configure(yscrollcommand=self.chip_scroll.set)
        self.chip_scroll.pack(side="right", fill="y")
        self.chip_canvas.pack(side="left", fill="both", expand=True)
        self.chip_inner = tk.Frame(self.chip_canvas, bg="#161616")
        self.chip_win = self.chip_canvas.create_window((0, 0), window=self.chip_inner, anchor="nw")
        self.chip_inner.bind("<Configure>", lambda e: self.chip_canvas.configure(scrollregion=self.chip_canvas.bbox("all")))
        self.chip_canvas.bind("<Configure>", lambda e: self.chip_canvas.itemconfig(self.chip_win, width=e.width))

        mid_row = tk.Frame(self.w, bg=BG)
        mid_row.grid(row=2, column=0, sticky="ew", padx=12, pady=2)
        of = ttk.LabelFrame(mid_row, text=t[3])
        of.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self.ol = ttk.Label(of, text="...")
        self.ol.pack(side="left", padx=8, pady=4)
        ttk.Button(of, text=t[14], command=self.so).pack(side="right", padx=8, pady=4)

        wf = ttk.LabelFrame(mid_row, text=vloc["f3"])
        wf.pack(side="right", fill="both", padx=(6, 0))
        self.wv = tk.StringVar(value="small")
        for tx, v in [("Tiny", "tiny"), ("Base", "base"), ("Small", "small"), ("Medium", "medium"), ("Large", "large-v3")]:
            ttk.Radiobutton(wf, text=tx, variable=self.wv, value=v).pack(side="left", padx=4, pady=4)

        af = ttk.LabelFrame(self.w, text=vloc["f4"])
        af.grid(row=3, column=0, sticky="ew", padx=12, pady=3)
        self.dt = tk.BooleanVar(value=True)
        ttk.Checkbutton(af, text=vloc["chk_srt"], variable=self.dt, command=self.ta).pack(anchor="w", padx=8, pady=2)

        aif = tk.Frame(af, bg=BG)
        aif.pack(fill="x", padx=8, pady=3)
        tk.Label(aif, text=t[10], bg=BG, fg=TXT).pack(side="left")

        get_installed_ollama_models(force_refresh=True)
        m_list = get_loc_list(MODELS_VIDEO, self.ui, include_custom=True, allow_vision=False)
        installed = get_cached_models()
        self.am = ttk.Combobox(aif, values=m_list, state="readonly")
        self.am.set(pick_best_installed_option(m_list, installed))
        self.am.pack(side="left", padx=5, fill="x", expand=True)

        self.cz = ttk.Combobox(aif, values=[5, 10, 20, 50, 100], state="readonly", width=5)
        self.cz.set(20)
        self.cz.pack(side="right", padx=4)
        tk.Label(aif, text=t[16], bg=BG, fg=TXT).pack(side="right", padx=2)

        self.at = ttk.Combobox(aif, values=list(L.keys()), state="readonly", width=10)
        self.at.set("Turkish")
        self.at.pack(side="right", padx=4)
        tk.Label(aif, text=t[12], bg=BG, fg=TXT).pack(side="right", padx=2)

        self.as_ = ttk.Combobox(aif, values=list(L.keys()), state="readonly", width=10)
        self.as_.set("Auto")
        self.as_.pack(side="right", padx=4)
        tk.Label(aif, text=t[11], bg=BG, fg=TXT).pack(side="right", padx=2)

        pf = tk.Frame(af, bg=BG)
        pf.pack(fill="x", padx=8, pady=3)
        tk.Label(pf, text=vloc["prompt"], bg=BG, fg=TXT).pack(side="left")
        self.vp = ttk.Entry(pf)
        self.vp.pack(side="left", padx=5, fill="x", expand=True)

        act_row = tk.Frame(self.w, bg=BG)
        act_row.grid(row=4, column=0, sticky="ew", padx=12, pady=4)
        self.sb_btn = tk.Button(act_row, text=vloc["start"], bg="#D83B01", fg="white", font=("Segoe UI", 11, "bold"), command=self.sp)
        self.sb_btn.pack(side="left", fill="x", expand=True, ipadx=25, ipady=5)
        self.chk_shutdown = tk.Checkbutton(
            act_row, text=vloc["shutdown"], variable=self.var_shutdown_vid,
            bg=BG, fg="#ffb74d", selectcolor=FRM, activebackground=BG, activeforeground="#ffb74d",
            font=("Segoe UI", 10, "bold")
        )
        self.chk_shutdown.pack(side="right", padx=10)

        gf = ttk.LabelFrame(self.w, text=vloc["f5"])
        gf.grid(row=5, column=0, sticky="nsew", padx=12, pady=(2, 10))
        self.canvas = tk.Canvas(gf, bg="#0d0d0d", highlightthickness=0)
        self.v_scroll = ttk.Scrollbar(gf, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.v_scroll.set)
        self.v_scroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.cards_frame = tk.Frame(self.canvas, bg="#0d0d0d")
        self.canvas_win = self.canvas.create_window((0, 0), window=self.cards_frame, anchor="nw")
        self.cards_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.chip_or_cards_bbox()))
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.ta()

    def chip_or_cards_bbox(self):
        return self.canvas.bbox("all")

    def _on_canvas_resize(self, event):
        self.canvas.itemconfig(self.canvas_win, width=event.width)
        if abs(event.width - self.last_canvas_w) > 36:
            self.last_canvas_w = event.width
            if self.resize_after_id:
                try:
                    self.w.after_cancel(self.resize_after_id)
                except:
                    pass
            self.resize_after_id = self.w.after(180, self._relayout_existing_cards_16_9)

    def _relayout_existing_cards_16_9(self):
        if not self.jobs or not self.w.winfo_exists():
            return
        cols, cw, ch = self._calc_card_geometry(len(self.jobs))
        for c in range(cols):
            self.cards_frame.grid_columnconfigure(c, weight=1)
        for idx, job in enumerate(self.jobs):
            r_idx, c_idx = idx // cols, idx % cols
            job["w"], job["h"] = cw, ch
            try:
                job["card_fr"].config(width=cw)
                job["card_fr"].grid(row=r_idx, column=c_idx, padx=8, pady=8, sticky="n")
                if job.get("raw_pil"):
                    self._set_job_base_image(job, job["raw_pil"])
                    self._render_job_card(job, force=True)
            except:
                pass

    def _on_url_paste(self, event):
        try:
            clip = self.w.clipboard_get().strip()
            if "\n" in clip:
                for line in clip.splitlines():
                    u = line.strip()
                    if u:
                        self._enqueue_url_group(u)
                return "break"
            self.w.after(40, self.add_url_from_entry)
        except:
            pass

    def add_url_from_entry(self):
        if self.running:
            return
        raw = self.url_ent.get().strip()
        if not raw:
            return
        self.url_ent.delete(0, tk.END)
        for line in raw.splitlines():
            u = line.strip()
            if u:
                self._enqueue_url_group(u)

    def _create_chip_row(self, gid, initial_text):
        row = tk.Frame(self.chip_inner, bg="#252526", bd=1, relief="flat")
        row.pack(fill="x", padx=4, pady=2)
        lbl = tk.Label(row, text=initial_text, bg="#252526", fg="#ffffff", font=("Segoe UI", 9, "bold"), anchor="w", padx=6)
        lbl.pack(side="left", fill="x", expand=True)
        btn_del = tk.Button(
            row, text="✖", bg="#b71c1c", fg="white", font=("Segoe UI", 8, "bold"),
            bd=0, padx=7, pady=1, cursor="hand2", command=lambda g=gid: self.remove_group(g)
        )
        btn_del.pack(side="right", padx=3, pady=2)
        return row, lbl

    def _enqueue_url_group(self, url):
        vloc = VID_UI_LOC.get(self.ui, VID_UI_LOC["English"])
        gid = self.next_gid
        self.next_gid += 1
        row_w, lbl_w = self._create_chip_row(gid, vloc["analyzing"].format(u=url[:65]))
        grp = {
            "gid": gid, "type": "url", "raw": url, "row": row_w, "lbl": lbl_w,
            "resolving": True,
            "items": [{"type": "url", "src": url, "title": url, "thumb": None}]
        }
        self.groups.append(grp)
        self._rebuild_cards_grid()
        threading.Thread(target=self._resolve_url_metadata, args=(grp,), daemon=True).start()

    def _resolve_url_metadata(self, grp):
        vloc = VID_UI_LOC.get(self.ui, VID_UI_LOC["English"])
        url = grp["raw"]
        # GÜVENLİK DÜZELTMESİ: SSL sertifika doğrulaması artık açık ('nocheckcertificate' kaldırıldı).
        opts = {'extract_flat': 'in_playlist', 'quiet': True, 'no_warnings': True, 'socket_timeout': 10}
        js_rt = get_js_runtime_for_ytdlp()
        if js_rt:
            opts['js_runtimes'] = js_rt
        try:
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url, download=False)
            except Exception as e:
                if "CERTIFICATE_VERIFY_FAILED" not in str(e) and "certificate verify failed" not in str(e).lower():
                    raise
                # Sistem sertifika deposu bozuk bilgisayarlarda (kurumsal proxy vb.) yalnızca bu durumda doğrulamasız dene.
                with yt_dlp.YoutubeDL(dict(opts, nocheckcertificate=True)) as ydl:
                    info = ydl.extract_info(url, download=False)
            if not info or grp not in self.groups:
                grp["resolving"] = False
                return
            if 'entries' in info and info['entries'] is not None:
                entries = [e for e in info['entries'] if e]
                if entries:
                    pl_title = info.get('title') or info.get('playlist_title') or "Playlist"
                    new_items = []
                    for idx, ent in enumerate(entries):
                        v_url = ent.get('url') or ent.get('webpage_url') or ""
                        if v_url and not v_url.startswith("http"):
                            ie_key = (ent.get('ie_key') or "").lower()
                            if "youtube" in ie_key or len(v_url) == 11:
                                v_url = f"https://www.youtube.com/watch?v={v_url}"
                        if not v_url:
                            v_url = url
                        v_title = ent.get('title') or f"{pl_title} #{idx + 1}"
                        thumbs = ent.get('thumbnails') or []
                        v_thumb = ent.get('thumbnail') or (thumbs[-1].get('url') if thumbs else None)
                        new_items.append({"type": "url", "src": v_url, "title": v_title, "thumb": v_thumb})
                    grp["items"] = new_items
                    grp["resolving"] = False
                    chip_txt = vloc["pl_tag"].format(title=pl_title, n=len(new_items))
                    self.w.after(0, lambda: self._apply_resolved_group(grp, chip_txt))
                    return
            v_title = info.get('title') or url
            thumbs = info.get('thumbnails') or []
            v_thumb = info.get('thumbnail') or (thumbs[-1].get('url') if thumbs else None)
            grp["items"] = [{"type": "url", "src": info.get('webpage_url') or url, "title": v_title, "thumb": v_thumb}]
            grp["resolving"] = False
            chip_txt = vloc["vid_tag"].format(title=v_title)
            self.w.after(0, lambda: self._apply_resolved_group(grp, chip_txt))
        except:
            grp["resolving"] = False
            if grp in self.groups:
                chip_txt = vloc["raw_tag"].format(u=url)
                self.w.after(0, lambda: self._apply_resolved_group(grp, chip_txt))

    def _apply_resolved_group(self, grp, chip_txt):
        if grp not in self.groups or not self.w.winfo_exists():
            return
        try:
            grp["lbl"].config(text=chip_txt)
        except:
            pass
        if not self.running:
            self._rebuild_cards_grid()

    def add_local_files(self):
        if self.running:
            return
        vloc = VID_UI_LOC.get(self.ui, VID_UI_LOC["English"])
        files = filedialog.askopenfilenames(
            title="Video / Audio",
            filetypes=[("Media Files", "*.mp4 *.mkv *.webm *.avi *.mov *.mp3 *.wav *.m4a *.flac"), ("All Files", "*.*")]
        )
        if files:
            for f in files:
                gid = self.next_gid
                self.next_gid += 1
                t_name = os.path.splitext(os.path.basename(f))[0]
                row_w, lbl_w = self._create_chip_row(gid, vloc["loc_tag"].format(title=t_name))
                grp = {
                    "gid": gid, "type": "local", "raw": f, "row": row_w, "lbl": lbl_w,
                    "resolving": False,
                    "items": [{"type": "local", "src": f, "title": t_name, "thumb": None}]
                }
                self.groups.append(grp)
            self._rebuild_cards_grid()

    def remove_group(self, gid):
        if self.running:
            return
        target = next((g for g in self.groups if g["gid"] == gid), None)
        if target:
            try:
                target["row"].destroy()
            except:
                pass
            self.groups.remove(target)
            self._rebuild_cards_grid()

    def clear_queue(self):
        if self.running:
            return
        for g in self.groups:
            try:
                g["row"].destroy()
            except:
                pass
        self.groups.clear()
        self._rebuild_cards_grid()

    def so(self):
        d = filedialog.askdirectory()
        if d:
            self.op = d
            self.ol.config(text=d)

    def ta(self):
        s = "readonly" if self.dt.get() else "disabled"
        e = "normal" if self.dt.get() else "disabled"
        self.as_.config(state="readonly")
        for w in [self.am, self.at, self.cz]:
            w.config(state=s)
        self.vp.config(state=e)

    def _calc_card_geometry(self, total_count):
        cw = max(760, self.canvas.winfo_width() - 36)
        if total_count <= 1:
            cols = 1
            card_w = min(560, max(380, int(cw * 0.56)))
        elif total_count == 2:
            cols = 2
            card_w = min(460, max(300, (cw - 36) // 2))
        elif total_count <= 6:
            cols = 3 if cw >= 980 else 2
            card_w = min(380, max(250, (cw - (cols + 1) * 16) // cols))
        else:
            cols = 4 if cw >= 1280 else 3
            card_w = min(330, max(220, (cw - (cols + 1) * 14) // cols))
        card_h = max(124, int(round(card_w * 9.0 / 16.0)))
        return cols, card_w, card_h

    def _create_placeholder_pil(self, w, h):
        im = Image.new("RGB", (w, h), (26, 32, 44))
        dr = ImageDraw.Draw(im)
        for y in range(0, h, 4):
            c = int(22 + (y / float(max(1, h))) * 34)
            dr.line([(0, y), (w, y)], fill=(c, c + 6, c + 16))
        return im

    def _fit_16_9_pil(self, src_img, w, h):
        if not PIL_OK or src_img is None:
            return None
        return ImageOps.fit(src_img.convert("RGB"), (w, h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

    def _extract_local_thumb_pil(self, video_path, w, h):
        if not PIL_OK or not FFMPEG_EXE:
            return None
        try:
            cmd = [FFMPEG_EXE, "-y", "-ss", "00:00:03", "-i", video_path, "-frames:v", "1", "-q:v", "4", "-f", "image2pipe", "-vcodec", "mjpeg", "-"]
            si = None
            if os.name == "nt":
                si = subprocess.STARTUPINFO()
                si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=6, startupinfo=si)
            if proc.stdout and len(proc.stdout) > 500:
                return Image.open(io.BytesIO(proc.stdout)).convert("RGB")
        except:
            pass
        return None

    def _download_thumb_pil(self, thumb_url, w, h):
        if not PIL_OK or not thumb_url:
            return None
        try:
            r = HTTP.get(thumb_url, timeout=6)
            if r.status_code == 200 and len(r.content) > 500:
                return Image.open(io.BytesIO(r.content)).convert("RGB")
        except:
            pass
        return None

    def _set_job_base_image(self, job, pil_img):
        if not PIL_OK or pil_img is None:
            return
        job["raw_pil"] = pil_img
        w, h = job["w"], job["h"]
        clear_im = self._fit_16_9_pil(pil_img, w, h)
        blur_im = ImageEnhance.Brightness(clear_im.filter(ImageFilter.GaussianBlur(radius=9))).enhance(0.42)
        job["clear_pil"] = clear_im
        job["blur_pil"] = blur_im
        job["last_drawn_pct"] = -5.0

    def _render_job_card(self, job, force=False):
        if not self.w.winfo_exists():
            return
        pct = max(0.0, min(100.0, float(job.get("pct", 0.0))))
        status_txt = job.get("status_text", "")
        title_txt = job.get("title", "Video")
        w, h = job["w"], job["h"]

        max_ch = max(24, w // 8)
        short_title = title_txt if len(title_txt) <= max_ch else title_txt[:max_ch - 3] + "..."
        try:
            job["title_lbl"].config(text=f"🎬 {short_title}")
            job["status_lbl"].config(text=status_txt, fg="#00e676" if pct >= 100 else ("#ff5252" if "❌" in status_txt else "#ffeb3b"))
        except:
            return

        if PIL_OK and job.get("clear_pil") and job.get("blur_pil"):
            if force or abs(pct - job.get("last_drawn_pct", -5.0)) >= 0.8 or pct in (0.0, 100.0):
                job["last_drawn_pct"] = pct
                comp = job["blur_pil"].copy()
                split_x = int((pct / 100.0) * w)
                if split_x > 0:
                    comp.paste(job["clear_pil"].crop((0, 0, split_x, h)), (0, 0))
                dr = ImageDraw.Draw(comp)
                if 0 < split_x < w:
                    dr.line([(split_x, 0), (split_x, h)], fill=(0, 215, 255), width=3)
                bar_w = int((pct / 100.0) * w)
                dr.rectangle([(0, h - 5), (w, h)], fill=(20, 20, 20))
                if bar_w > 0:
                    dr.rectangle([(0, h - 5), (bar_w, h)], fill=(0, 230, 118) if pct >= 100 else (0, 175, 255))
                tk_im = ImageTk.PhotoImage(comp)
                job["tk_img"] = tk_im
                job["img_lbl"].config(image=tk_im, width=w, height=h)

        self._update_overall_taskbar()

    def _update_overall_taskbar(self):
        if not self.jobs:
            TASKBAR.clear(self.w)
            return
        total_pct = sum(float(j.get("pct", 0.0)) for j in self.jobs)
        TASKBAR.set_progress(self.w, int(total_pct), len(self.jobs) * 100, WinTaskbarProgress.TBPF_NORMAL)

    def _rebuild_cards_grid(self):
        if self.running:
            return
        for w in self.cards_frame.winfo_children():
            w.destroy()
        self.jobs.clear()

        flat_items = []
        for g in self.groups:
            for it in g["items"]:
                flat_items.append(it)
        if not flat_items:
            TASKBAR.clear(self.w)
            return

        cols, cw, ch = self._calc_card_geometry(len(flat_items))
        for c in range(4):
            self.cards_frame.grid_columnconfigure(c, weight=1 if c < cols else 0)
        c_loc = CARD_LOC.get(self.ui, CARD_LOC["English"])

        for idx, it in enumerate(flat_items):
            r_idx, c_idx = idx // cols, idx % cols
            card_fr = tk.Frame(self.cards_frame, bg="#181818", bd=1, relief="solid", width=cw)
            card_fr.grid(row=r_idx, column=c_idx, padx=8, pady=8, sticky="n")

            t_lbl = tk.Label(card_fr, text="", bg="#0e0e0e", fg="#ffffff", font=("Segoe UI", 9, "bold"), anchor="w", padx=6, pady=3)
            t_lbl.pack(fill="x")
            im_lbl = tk.Label(card_fr, bg="#111111", bd=0, width=cw, height=ch)
            im_lbl.pack()
            s_lbl = tk.Label(card_fr, text="", bg="#0e0e0e", fg="#ffeb3b", font=("Segoe UI", 9, "bold"), anchor="w", padx=6, pady=3)
            s_lbl.pack(fill="x")

            job = {
                "id": idx,
                "card_fr": card_fr,
                "type": it["type"],
                "src": it["src"],
                "title": it["title"],
                "thumb_url": it.get("thumb"),
                "media_path": it["src"] if it["type"] == "local" else "",
                "w": cw,
                "h": ch,
                "pct": 0.0,
                "last_drawn_pct": -5.0,
                "status_text": c_loc["wait"],
                "title_lbl": t_lbl,
                "img_lbl": im_lbl,
                "status_lbl": s_lbl,
                "raw_pil": None,
                "clear_pil": None,
                "blur_pil": None,
                "tk_img": None
            }
            if PIL_OK:
                self._set_job_base_image(job, self._create_placeholder_pil(cw, ch))
            self.jobs.append(job)
            self._render_job_card(job, force=True)

            if it["type"] == "local":
                threading.Thread(target=self._async_load_local_thumb, args=(job,), daemon=True).start()
            elif it.get("thumb"):
                threading.Thread(target=self._async_load_url_thumb, args=(job, it["thumb"]), daemon=True).start()

    def _async_load_local_thumb(self, job):
        im = self._extract_local_thumb_pil(job["src"], job["w"], job["h"])
        if im and job in self.jobs:
            self._set_job_base_image(job, im)
            self.w.after(0, lambda j=job: self._render_job_card(j, force=True))

    def _async_load_url_thumb(self, job, th_url):
        im = self._download_thumb_pil(th_url, job["w"], job["h"])
        if im and job in self.jobs:
            self._set_job_base_image(job, im)
            self.w.after(0, lambda j=job: self._render_job_card(j, force=True))

    def _set_job_state(self, job, pct, status_str, force=False):
        job["pct"] = pct
        job["status_text"] = status_str
        if self.w.winfo_exists():
            self.w.after(0, lambda j=job, f=force: self._render_job_card(j, force=f))

    def sp(self):
        # YENİ: Başlat butonu çalışırken "DURDUR" butonuna dönüşür (eskiden durdurmanın tek yolu pencereyi kapatmaktı).
        if self.running:
            self.abort_flag = True
            self.sb_btn.config(text=xui(self.ui, "stopping"), state="disabled")
            return
        if self.url_ent.get().strip():
            self.add_url_from_entry()
        if not self.op:
            messagebox.showerror("Error", "Please select an output folder.")
            return
        am_txt = self.am.get().strip()
        if self.dt.get() and (not am_txt or am_txt.startswith("|") or am_txt.startswith("⚠️")):
            messagebox.showerror("Error", "Please select a valid installed AI model.")
            return
        if not self.groups and not self.jobs:
            messagebox.showerror("Error", "Please add at least 1 video/audio file or URL.")
            return

        self.abort_flag = False
        self.running = True
        self.sb_btn.config(text=xui(self.ui, "stop"), bg="#b71c1c", state="normal")
        threading.Thread(target=self._prepare_and_run_pipeline, daemon=True).start()

    def _reset_start_button(self):
        if not self.w.winfo_exists():
            return
        vloc = VID_UI_LOC.get(self.ui, VID_UI_LOC["English"])
        self.sb_btn.config(text=vloc["start"], bg="#D83B01", state="normal")

    def _prepare_and_run_pipeline(self):
        # DÜZELTME: Erken durdurma veya beklenmeyen bir hata olduğunda "çalışıyor" durumu takılı kalıyordu.
        try:
            wait_loops = 0
            while any(g.get("resolving", False) for g in self.groups) and wait_loops < 60:
                if self.abort_flag:
                    return
                time.sleep(0.25)
                wait_loops += 1

            ev = threading.Event()
            def _sync_rebuild():
                self.running = False
                self._rebuild_cards_grid()
                self.running = True
                ev.set()
            self.w.after(0, _sync_rebuild)
            ev.wait(timeout=5.0)
            self._run_async_pipeline()
        except Exception as e:
            try:
                if self.w.winfo_exists():
                    self.w.after(0, lambda m=str(e)[:400]: messagebox.showerror("Error", m, parent=self.w))
            except Exception:
                pass
        finally:
            self.running = False
            try:
                if self.w.winfo_exists():
                    self.w.after(0, self._reset_start_button)
                    self.w.after(0, lambda: TASKBAR.clear(self.w))
            except Exception:
                pass

    def _run_async_pipeline(self):
        c_loc = CARD_LOC.get(self.ui, CARD_LOC["English"])
        do_ai = self.dt.get()
        whisper_model_size = self.wv.get()
        is_gpu_selected = (self.whisper_dev_mode.get() == "GPU")
        installed = get_installed_ollama_models(force_refresh=True)
        ai_model_name = resolve_installed_model_name(self.am.get().split(" ")[0], installed)
        sl_name = L[self.as_.get()][0]
        sl_code = L[self.as_.get()][1]
        tl_name = L[self.at.get()][0]
        tl_code = L[self.at.get()][1]
        custom_p = self.vp.get().strip()
        chunk_sz = int(self.cz.get())
        filler_re = re.compile(r'^(m+|a+|ı+|e+|hı+|hm+|hmm+|he+|uff+|uf+|uh+|um+|er+|ah+|oh+)[\.\,\!\?\-]*$', flags=re.IGNORECASE)

        # Uzun başlıklı videolarda MAX_PATH çökmesini önleyen ve gerçek dosya yolunu yakalayan indirici.
        def download_job_media(job):
            if self.abort_flag:
                return False
            if job["type"] == "local":
                if not os.path.exists(job["src"]):
                    raise Exception("File not found")
                self._set_job_state(job, 25.0, c_loc["wh_Wait"], force=True)
                return True
            url = job["src"]
            self._set_job_state(job, 2.0, c_loc["dl"].format(p=0, sp="..."), force=True)
            last_hook_t = [0.0]
            hook_files = []

            def dl_hook(d):
                if self.abort_flag:
                    raise Exception("Aborted")
                st = d.get("status")
                if st == "downloading":
                    now = time.time()
                    if now - last_hook_t[0] < 0.15:
                        return
                    last_hook_t[0] = now
                    tot_b = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                    dl_b = d.get("downloaded_bytes", 0)
                    raw_p = int((dl_b / tot_b) * 100) if tot_b > 0 else 50
                    sp_str = re.sub(r'\x1b\[[0-9;]*m', '', str(d.get('_speed_str', '')).strip())
                    self._set_job_state(job, raw_p * 0.30, c_loc["dl"].format(p=raw_p, sp=sp_str))
                elif st == "finished" and d.get("filename"):
                    hook_files.append(d.get("filename"))

            ydl_opts = {
                # DÜZELTME: FFmpeg yoksa ayrı video+ses birleştirilemez; tek dosyalı (sesli) format seçilir.
                'format': ('bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best' if FFMPEG_EXE
                           else 'best[ext=mp4][acodec!=none]/best[acodec!=none]/best'),
                'outtmpl': os.path.join(self.op, '%(title).75s [%(id)s].%(ext)s'),
                'trim_file_name': 80,
                'progress_hooks': [dl_hook],
                'concurrent_fragment_downloads': 4,
                'http_chunk_size': 5242880,
                'buffersize': 524288,
                'windowsfilenames': True,
                'noplaylist': True,
                'retries': 10,
                'fragment_retries': 10,
                'quiet': True
            }
            if FFMPEG_EXE:
                ydl_opts['ffmpeg_location'] = FFMPEG_EXE
                ydl_opts['merge_output_format'] = 'mp4'
            js_rt = get_js_runtime_for_ytdlp()
            if js_rt:
                ydl_opts['js_runtimes'] = js_rt

            last_dl_err = None
            for dl_try in range(3):
                if self.abort_flag:
                    return False
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info_dict = ydl.extract_info(url, download=True)
                        if not job.get("thumb_url") and info_dict.get("thumbnail"):
                            im = self._download_thumb_pil(info_dict["thumbnail"], job["w"], job["h"])
                            if im:
                                self._set_job_base_image(job, im)

                        # yt-dlp'nin diske yazdığı gerçek dosya yolunu tespit eder.
                        media_p = ""
                        req_dls = info_dict.get("requested_downloads") or []
                        for rd in req_dls:
                            fp = rd.get("filepath") or rd.get("filename") or ""
                            if fp and os.path.exists(fp):
                                media_p = fp
                                break
                        if not media_p and info_dict.get("_filename") and os.path.exists(info_dict["_filename"]):
                            media_p = info_dict["_filename"]
                        if not media_p:
                            dl_file = ydl.prepare_filename(info_dict)
                            base_no_ext = os.path.splitext(dl_file)[0]
                            for ext_cand in (".mp4", ".mkv", ".webm", ".m4a", ".mp3", ".wav"):
                                if os.path.exists(base_no_ext + ext_cand):
                                    media_p = base_no_ext + ext_cand
                                    break
                            if not media_p and os.path.exists(dl_file):
                                media_p = dl_file
                        if not media_p and hook_files:
                            for hf in reversed(hook_files):
                                b_hf = os.path.splitext(hf)[0]
                                b_hf = re.sub(r"\.f\d+$", "", b_hf)
                                if os.path.exists(b_hf + ".mp4"):
                                    media_p = b_hf + ".mp4"
                                    break
                                if os.path.exists(hf):
                                    media_p = hf
                                    break
                        if not media_p:
                            vid_id = str(info_dict.get("id") or "")
                            if vid_id and os.path.isdir(self.op):
                                for fn in os.listdir(self.op):
                                    if vid_id in fn and not fn.endswith((".part", ".ytdl", ".srt")):
                                        media_p = os.path.join(self.op, fn)
                                        break
                        if not media_p or not os.path.exists(media_p):
                            raise Exception("Downloaded media file could not be located on disk.")

                        job["media_path"] = media_p
                        job["title"] = info_dict.get("title") or os.path.splitext(os.path.basename(media_p))[0]
                    last_dl_err = None
                    break
                except Exception as e:
                    last_dl_err = e
                    if self.abort_flag or "Aborted" in str(e):
                        return False
                    err_l = str(e).lower()
                    if ("certificate_verify_failed" in err_l or "certificate verify failed" in err_l) and not ydl_opts.get('nocheckcertificate'):
                        # Bozuk sertifika deposu olan sistemlerde yalnızca bu durumda doğrulamasız tekrar dene.
                        ydl_opts['nocheckcertificate'] = True
                        continue
                    if "ffmpeg" in err_l and FFMPEG_EXE is None:
                        ydl_opts['format'] = 'best[acodec!=none]/best'
                    time.sleep(2.0)
            if last_dl_err:
                raise last_dl_err

            self._set_job_state(job, 30.0, c_loc["wh_Wait"], force=True)
            return True

        # Bozuk ses kodeklerinde veya uzun dosya yollarında FFmpeg ile 16kHz WAV çıkararak Whisper çökmesini önler.
        def transcribe_job_audio(job, wm_inst, dev_tag):
            if self.abort_flag:
                return
            media_p = job["media_path"]
            base_name = sanitize_filename(os.path.splitext(os.path.basename(media_p))[0], max_len=80)
            job["base_name"] = base_name

            lang_param = None if sl_code == "auto" else sl_code
            transcribe_kwargs = dict(
                language=lang_param,
                beam_size=5,
                word_timestamps=True,
                condition_on_previous_text=False,
                no_speech_threshold=0.75,
                log_prob_threshold=-1.2,
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.35,
                    min_silence_duration_ms=650,
                    speech_pad_ms=400
                )
            )

            def _open_whisper_stream(target_p):
                try:
                    if sl_code == "auto":
                        return wm_inst.transcribe(target_p, multilingual=True, **transcribe_kwargs)
                    return wm_inst.transcribe(target_p, **transcribe_kwargs)
                except TypeError:
                    return wm_inst.transcribe(target_p, **transcribe_kwargs)

            def _run_stream(target_p):
                sg, inf = _open_whisper_stream(target_p)
                out_tuples = []
                seq_preview = 0
                for s in sg:
                    if self.abort_flag:
                        return None
                    for st_s, en_s, tx_s in build_tight_whisper_segments(s, filler_re):
                        out_tuples.append([st_s, en_s, tx_s])
                        seq_preview += 1
                    if inf.duration and inf.duration > 0:
                        w_pct = min(100, int((s.end / inf.duration) * 100))
                        span = 35.0 if do_ai else 70.0
                        self._set_job_state(job, 30.0 + (w_pct / 100.0) * span, c_loc["wh"].format(dev=dev_tag, p=w_pct, ln=seq_preview))
                return out_tuples

            # LİSANS + SAĞLAMLIK DÜZELTMESİ: Ses, harici FFmpeg ile 16 kHz mono'ya çözülüp Whisper'a numpy dizisi
            # olarak verilir. Böylece GPL'li PyAV/x264 kütüphanelerine gerek kalmaz; bozuk kodekler ve çok uzun
            # dosya yolları da FFmpeg tarafından sorunsuz işlenir (eski WAV yedek yolunun yerini aldı).
            if FFMPEG_EXE:
                self._set_job_state(job, max(30.0, float(job.get("pct", 30.0))), c_loc["wh"].format(dev=dev_tag, p=0, ln=0))
                audio_np = decode_audio_ffmpeg(media_p, 16000, abort_check=lambda: self.abort_flag)
                if audio_np is None:
                    return
                raw_tuples = _run_stream(audio_np)
                del audio_np
            elif not AV_STUBBED:
                raw_tuples = _run_stream(media_p)
            else:
                raise Exception("FFmpeg (ffmpeg.exe) not found")
            if raw_tuples is None:
                return

            segs_out = []
            for idx_t, item in enumerate(raw_tuples):
                st_s, en_s, tx_s = item
                if idx_t + 1 < len(raw_tuples):
                    next_st = raw_tuples[idx_t + 1][0]
                    if en_s > next_st - 0.02:
                        en_s = max(st_s + 0.35, next_st - 0.02)
                segs_out.append((idx_t + 1, f"{format_srt_time(st_s)} --> {format_srt_time(en_s)}", tx_s))

            if not segs_out:
                raise Exception("No speech detected")

            orig_srt_path = os.path.join(self.op, f"{base_name}_Orijinal.srt")
            with open(orig_srt_path, "w", encoding="utf-8") as fl:
                fl.write("\n\n".join(f"{idx}\n{ts}\n{tx}" for idx, ts, tx in segs_out) + "\n")
            job["orig_srt_path"] = orig_srt_path
            job["segments"] = segs_out

        def translate_job_srt(job, sess):
            if self.abort_flag:
                return
            segs = job.get("segments", [])
            base_name = sanitize_filename(job.get("base_name", f"Video_{job.get('id', 1)}"), max_len=80)
            chunks = [segs[i:i + chunk_sz] for i in range(0, len(segs), chunk_sz)]
            tot_c = max(1, len(chunks))
            translated_blocks = []

            wait_until_ollama_ready(
                abort_check=lambda: self.abort_flag,
                status_cb=lambda _m: self._set_job_state(job, 65.0, c_loc["retry"].format(c=0, t=tot_c))
            )

            for i, ch_items in enumerate(chunks):
                if self.abort_flag:
                    return
                ai_pct = int(((i + 1) / tot_c) * 100)
                cur_prog = 65.0 + (ai_pct / 100.0) * 35.0
                self._set_job_state(job, cur_prog, c_loc["ai"].format(p=ai_pct, c=i + 1, t=tot_c))
                block_res = translate_srt_chunk_locked(
                    ch_items, ai_model_name, sl_name, tl_name, custom_p,
                    session=sess,
                    abort_check=lambda: self.abort_flag,
                    status_cb=lambda _m, ci=i+1: self._set_job_state(job, cur_prog, c_loc["retry"].format(c=ci, t=tot_c))
                )
                translated_blocks.append(block_res)

            trans_srt_path = os.path.join(self.op, f"{base_name}_{tl_code}.srt")
            if os.path.abspath(trans_srt_path) == os.path.abspath(job.get("orig_srt_path", "")):
                trans_srt_path = os.path.join(self.op, f"{base_name}_{tl_code}_Translated.srt")
            with open(trans_srt_path, "w", encoding="utf-8") as fl:
                fl.write("\n\n".join(translated_blocks) + "\n")
            self._set_job_state(job, 100.0, c_loc["done"], force=True)

        if is_gpu_selected:
            worker_sess = create_http_session()
            try:
                for job in self.jobs:
                    if self.abort_flag:
                        break
                    try:
                        if not download_job_media(job) or self.abort_flag:
                            break
                        unload_all_ollama_models(ACTIVE_HOST)
                        wm, active_dev_label = load_whisper_auto(whisper_model_size, force_cpu=False)
                        try:
                            try:
                                transcribe_job_audio(job, wm, active_dev_label)
                            except Exception:
                                if "GPU" in active_dev_label:
                                    del wm
                                    gc.collect()
                                    wm, active_dev_label = load_whisper_auto(whisper_model_size, force_cpu=True)
                                    transcribe_job_audio(job, wm, active_dev_label)
                                else:
                                    raise
                        finally:
                            try:
                                del wm
                            except:
                                pass
                            gc.collect()

                        if self.abort_flag:
                            break
                        if do_ai:
                            self._set_job_state(job, 65.0, c_loc["ai_wait"], force=True)
                            translate_job_srt(job, worker_sess)
                            unload_all_ollama_models(ACTIVE_HOST)
                        else:
                            self._set_job_state(job, 100.0, c_loc["done"], force=True)
                    except Exception as e:
                        if not self.abort_flag:
                            self._set_job_state(job, job["pct"], c_loc["err"].format(e=str(e)[:38]), force=True)
            finally:
                worker_sess.close()
        else:
            unload_all_ollama_models(ACTIVE_HOST)
            whisper_q = queue.Queue()
            ready_for_ai = []

            def dl_and_enqueue(job):
                if self.abort_flag:
                    return
                try:
                    if download_job_media(job) and not self.abort_flag:
                        whisper_q.put(job)
                except Exception as e:
                    if not self.abort_flag:
                        self._set_job_state(job, job["pct"], c_loc["err"].format(e=str(e)[:38]), force=True)

            def cpu_whisper_worker():
                # DÜZELTME: Whisper modeli yüklenemezse (internet yok, disk dolu...) iş parçacığı sessizce ölüyor,
                # kartlar sonsuza kadar "bekliyor" görünüyordu. Artık hata her karta yazılır.
                try:
                    wm, active_dev_label = load_whisper_auto(whisper_model_size, force_cpu=True)
                except Exception as e:
                    err_txt = c_loc["err"].format(e=("Whisper: " + str(e))[:60])
                    while True:
                        job = whisper_q.get()
                        if job is None:
                            break
                        self._set_job_state(job, job["pct"], err_txt, force=True)
                    return
                try:
                    while not self.abort_flag:
                        job = whisper_q.get()
                        if job is None or self.abort_flag:
                            break
                        try:
                            transcribe_job_audio(job, wm, active_dev_label)
                            if not self.abort_flag:
                                if do_ai:
                                    self._set_job_state(job, 65.0, c_loc["ai_wait"], force=True)
                                    ready_for_ai.append(job)
                                else:
                                    self._set_job_state(job, 100.0, c_loc["done"], force=True)
                        except Exception as e:
                            if not self.abort_flag:
                                self._set_job_state(job, job["pct"], c_loc["err"].format(e=str(e)[:38]), force=True)
                        finally:
                            whisper_q.task_done()
                finally:
                    del wm
                    gc.collect()

            t_wh = threading.Thread(target=cpu_whisper_worker, daemon=True)
            t_wh.start()
            with ThreadPoolExecutor(max_workers=2) as dl_pool:
                dl_pool.map(dl_and_enqueue, self.jobs)
            whisper_q.put(None)
            t_wh.join()

            if do_ai and ready_for_ai and not self.abort_flag:
                worker_sess = create_http_session()
                try:
                    for job in ready_for_ai:
                        if self.abort_flag:
                            break
                        try:
                            translate_job_srt(job, worker_sess)
                        except Exception as e:
                            if not self.abort_flag:
                                self._set_job_state(job, job["pct"], c_loc["err"].format(e=str(e)[:38]), force=True)
                    unload_all_ollama_models(ACTIVE_HOST)
                finally:
                    worker_sess.close()

        self.running = False
        if self.w.winfo_exists():
            self.w.after(0, lambda: TASKBAR.clear(self.w))
            self.w.after(0, self._reset_start_button)
            if not self.abort_flag and self.var_shutdown_vid.get():
                trigger_pc_shutdown_with_cancel(self.w, self.var_shutdown_vid, self.ui)

# Ana SRT çeviri ve kontrol merkezi.
class AILocalTranslator:
    def __init__(self, root):
        self.r = root
        self.r.title(f"{APP_NAME} v{APP_VERSION}")
        self.running_main = False
        self.r.geometry("960x960")
        self.r.minsize(820, 760)
        set_dark(self.r)
        set_icon(self.r)
        self.sf_list = []
        self.of = ""
        self.abort_main = False
        self.var_shutdown_main = tk.BooleanVar(value=False)
        self.var_shutdown_main.trace_add("write", self._on_main_shutdown_toggle)
        self.lang_var = tk.StringVar(value="English")
        self.lang_var.trace_add("write", self.cl)
        self.hdr_icon_img = None
        self.r.protocol("WM_DELETE_WINDOW", self.on_app_close)
        self.bui()
        threading.Thread(target=self.rm, daemon=True).start()

    def _on_main_shutdown_toggle(self, *args):
        if not self.var_shutdown_main.get():
            cancel_pc_shutdown()

    def on_app_close(self):
        self.abort_main = True
        self.var_shutdown_main.set(False)
        cancel_pc_shutdown()
        TASKBAR.clear(self.r)
        cleanup_on_exit()
        self.r.destroy()
        os._exit(0)

    def wl(self, txt):
        if not self.r.winfo_exists():
            return
        self.log.insert(tk.END, txt + "\n")
        self.log.see(tk.END)
        self.r.update_idletasks()

    def bui(self):
        self.r.grid_columnconfigure(0, weight=1)
        self.r.grid_rowconfigure(7, weight=2, minsize=95)
        self.r.grid_rowconfigure(12, weight=5, minsize=180)

        hdr_frame = tk.Frame(self.r, bg=BG)
        hdr_frame.grid(row=0, column=0, pady=6)
        self.hdr_icon_img = load_header_icon_photo((28, 28))
        if self.hdr_icon_img:
            self.hdr_lbl = tk.Label(
                hdr_frame, text=f" {APP_NAME.upper()} v{APP_VERSION}", image=self.hdr_icon_img, compound="left",
                font=("Segoe UI", 18, "bold"), bg=BG, fg="#00aaff"
            )
        else:
            self.hdr_lbl = tk.Label(
                hdr_frame, text=f" {APP_NAME.upper()} v{APP_VERSION}",
                font=("Segoe UI", 18, "bold"), bg=BG, fg="#00aaff"
            )
        self.hdr_lbl.pack(side="left")
        # YENİ: Üçüncü taraf lisans bildirimlerini açan buton (LGPL/Apache uyumluluğu için kullanıcıya erişilebilir olmalı).
        self.btn_lic = tk.Button(hdr_frame, text="📜", bg=BG, fg="#aaaaaa", bd=0, font=("Segoe UI", 11), cursor="hand2", command=self.show_licenses)
        self.btn_lic.pack(side="left", padx=(10, 0))

        self.bv = tk.Button(self.r, bg="#D83B01", fg="white", font=("Segoe UI", 12, "bold"), command=self.ov)
        self.bv.grid(row=2, column=0, sticky="ew", padx=15, pady=(0, 6), ipady=8)

        self.l1 = ttk.LabelFrame(self.r)
        self.l1.grid(row=3, column=0, sticky="ew", padx=15, pady=3)
        self.fl = ttk.Label(self.l1, text="No file selected...")
        self.fl.pack(side="left", padx=10, pady=5)
        ttk.Button(self.l1, text="🗑️", width=3, command=self.c_f).pack(side="right", padx=(2, 10), pady=4)
        self.b1 = ttk.Button(self.l1, command=self.s_f)
        self.b1.pack(side="right", padx=4, pady=4)

        self.l2 = ttk.LabelFrame(self.r)
        self.l2.grid(row=4, column=0, sticky="ew", padx=15, pady=3)
        self.ol = ttk.Label(self.l2, text="No folder selected...")
        self.ol.pack(side="left", padx=10, pady=5)
        self.b2 = ttk.Button(self.l2, command=self.s_o)
        self.b2.pack(side="right", padx=10, pady=4)

        self.l3 = ttk.LabelFrame(self.r)
        self.l3.grid(row=5, column=0, sticky="ew", padx=15, pady=3)
        self.ll1 = tk.Label(self.l3, bg=BG, fg=TXT)
        self.ll1.pack(side="left", padx=10, pady=5)
        self.sl = ttk.Combobox(self.l3, values=list(L.keys()), state="readonly", width=18)
        self.sl.set("Auto")
        self.sl.pack(side="left", padx=5)
        self.ll2 = tk.Label(self.l3, bg=BG, fg=TXT)
        self.ll2.pack(side="left", padx=(20, 10))
        self.tl = ttk.Combobox(self.l3, values=list(L.keys()), state="readonly", width=18)
        self.tl.set("Turkish")
        self.tl.pack(side="left", padx=5)

        self.l4 = ttk.LabelFrame(self.r)
        self.l4.grid(row=6, column=0, sticky="ew", padx=15, pady=3)
        self.lm = tk.Label(self.l4, bg=BG, fg=TXT)
        self.lm.pack(side="left", padx=10, pady=5)
        self.cz_main = ttk.Combobox(self.l4, values=[5, 10, 20, 50, 100], state="readonly", width=6)
        self.cz_main.set(20)
        self.cz_main.pack(side="right", padx=10)
        self.l_chk = tk.Label(self.l4, text="Chunk:", bg=BG, fg=TXT)
        self.l_chk.pack(side="right", padx=4)
        self.mb = ttk.Combobox(self.l4, state="readonly")
        self.mb.pack(side="left", padx=5, fill="x", expand=True)

        self.l5 = ttk.LabelFrame(self.r)
        self.l5.grid(row=7, column=0, sticky="nsew", padx=15, pady=3)
        self.cp = tk.Text(self.l5, height=3, font=("Segoe UI", 10), bg=FRM, fg=TXT)
        self.cp.pack(fill="both", expand=True, padx=10, pady=5)

        self.l6 = ttk.LabelFrame(self.r)
        self.l6.grid(row=8, column=0, sticky="ew", padx=15, pady=3)
        ttk.Button(self.l6, text="Sil / Delete", command=self.dlm).pack(side="right", padx=(4, 10), pady=5)
        ttk.Button(self.l6, text="İndir / Download", command=self.dm).pack(side="right", padx=4, pady=5)
        self.mn = ttk.Combobox(self.l6, state="readonly")
        self.mn.pack(side="left", padx=(10, 5), pady=5, fill="x", expand=True)

        self.l7 = ttk.LabelFrame(self.r)
        self.l7.grid(row=9, column=0, sticky="ew", padx=15, pady=3)
        self.uic = ttk.Combobox(self.l7, textvariable=self.lang_var, values=list(UIT.keys()), state="readonly", width=22)
        self.uic.pack(side="left", padx=10, pady=5)

        st = ttk.Frame(self.r)
        st.grid(row=10, column=0, sticky="ew", padx=15, pady=3)
        self.pg = ttk.Progressbar(st, length=100, mode="determinate")
        self.pg.pack(fill="x", pady=2)
        info_row = ttk.Frame(st)
        info_row.pack(fill="x")
        self.pl = ttk.Label(info_row, text="Progress: 0%", font=("Segoe UI", 10, "bold"))
        self.pl.pack(side="left")
        self.part_lbl = ttk.Label(info_row, text="File: 0/0 | Chunk: 0/0", font=("Segoe UI", 10, "bold"))
        self.part_lbl.pack(side="right")
        self.st_var = tk.StringVar(value="Ready.")
        ttk.Label(st, textvariable=self.st_var, foreground="#00aaff", font=("Segoe UI", 10, "bold")).pack(pady=2)

        act_frame = tk.Frame(self.r, bg=BG)
        act_frame.grid(row=11, column=0, sticky="ew", padx=15, pady=4)
        self.bs = tk.Button(act_frame, bg="#333333", fg="white", font=("Segoe UI", 12, "bold"), command=self.s_t)
        self.bs.pack(side="left", fill="x", expand=True, ipadx=20, ipady=7)
        self.chk_shutdown_main = tk.Checkbutton(
            act_frame, text="🔌 Shutdown PC When Done", variable=self.var_shutdown_main,
            bg=BG, fg="#ffb74d", selectcolor=FRM, activebackground=BG, activeforeground="#ffb74d",
            font=("Segoe UI", 10, "bold")
        )
        self.chk_shutdown_main.pack(side="right", padx=10)

        self.log_fr = ttk.LabelFrame(self.r)
        self.log_fr.grid(row=12, column=0, sticky="nsew", padx=15, pady=(3, 10))
        self.log_scroll = ttk.Scrollbar(self.log_fr, orient="vertical")
        self.log = tk.Text(self.log_fr, height=6, bg="#111111", fg="#00ff00", font=("Consolas", 10), yscrollcommand=self.log_scroll.set)
        self.log_scroll.config(command=self.log.yview)
        self.log_scroll.pack(side="right", fill="y", pady=5)
        self.log.pack(side="left", fill="both", expand=True, padx=(5, 0), pady=5)

        dev_name = "GPU (CUDA)" if get_best_whisper_device()[0] == "cuda" else "CPU"
        self.wl(f"System started. Whisper Audio Engine Available: {dev_name}")
        self.cl()

    def cl(self, *a):
        l = self.lang_var.get()
        t = UIT.get(l, UIT["English"])
        srt_list = get_loc_list(MODELS_SRT, l, include_custom=True, allow_vision=False)
        mng_list = get_loc_list(MODELS_MANAGE, l, include_custom=True, allow_vision=True)
        installed = get_cached_models()

        self.bv.config(text=t[1])
        self.l1.config(text=t[2]); self.l2.config(text=t[3])
        self.l3.config(text=t[4]); self.l4.config(text=t[5])
        self.l5.config(text=t[6]); self.l6.config(text=t[7])
        self.l7.config(text=t[8]); self.bs.config(text=(xui(l, "stop") if self.running_main else t[9]))
        self.btn_lic.config(text=xui(l, "licenses"))
        self.lm.config(text=t[10]); self.ll1.config(text=t[11])
        self.ll2.config(text=t[12]); self.b1.config(text=t[13])
        self.b2.config(text=t[14]); self.log_fr.config(text=t[15])
        self.l_chk.config(text=t[16])
        self.chk_shutdown_main.config(text=t[17])

        self.mb.config(values=srt_list)
        self.mn.config(values=mng_list)
        if self.mb.get() not in srt_list:
            self.mb.set(pick_best_installed_option(srt_list, installed))
        if self.mn.get() not in mng_list:
            self.mn.set(pick_best_installed_option(mng_list, installed))
        if hasattr(self, "_refresh_sub_ui"):
            self._refresh_sub_ui()


    def ov(self):
        VideoTranslator(self.r, self.lang_var.get())

    def rm(self):
        installed = get_installed_ollama_models(force_refresh=True)
        if installed:
            self.r.after(0, self.wl, f"✅ Ollama active ({ACTIVE_HOST}). Installed models: {', '.join(installed)}")
        else:
            try:
                if HTTP.get(T_URL, timeout=1.5).status_code == 200:
                    self.r.after(0, self.wl, "⚠️ Ollama is running but no models are installed yet. Download a model from section 6.")
                else:
                    self.r.after(0, self.wl, "⚠️ No active Ollama found! Please start the Ollama application.")
            except:
                self.r.after(0, self.wl, "⚠️ No active Ollama found! Please start the Ollama application.")
        self.r.after(0, self.cl)

    def dm(self):
        s = self.mn.get().strip()
        if not s or s.startswith("|") or s.startswith("⚠️"):
            return
        m_name = s.split(" ")[0]
        lic = model_license_warning(m_name)
        if lic and not messagebox.askyesno(xui(self.lang_var.get(), "lic_title"), xui(self.lang_var.get(), "lic_warn", m=m_name, lic=lic)):
            return
        self.wl(f"📥 Downloading model ({ACTIVE_HOST}): {m_name}")
        threading.Thread(target=self._dl, args=(m_name,), daemon=True).start()

    def show_licenses(self):
        p = find_notices_file()
        if not p or not open_path_with_os(p):
            messagebox.showinfo(xui(self.lang_var.get(), "licenses"), xui(self.lang_var.get(), "notices_missing"))

    def _dl(self, m):
        try:
            # DÜZELTME: Ollama indirme hatalarını akış içinde {"error": ...} olarak bildirir; eskiden bu
            # görmezden gelinip "İndirildi!" yazılıyordu (ör. var olmayan model adı).
            final_status = ""
            with HTTP.post(P_URL, json={"name": m, "model": m, "stream": True}, stream=True, timeout=7200) as r:
                if r.status_code != 200:
                    try:
                        raise Exception(r.json().get("error") or f"HTTP {r.status_code}")
                    except ValueError:
                        raise Exception(f"HTTP {r.status_code}")
                for line in r.iter_lines():
                    if line:
                        data = json.loads(line.decode("utf-8"))
                        if data.get("error"):
                            raise Exception(data["error"])
                        final_status = data.get("status", final_status)
                        tot = data.get("total", 0)
                        comp = data.get("completed", 0)
                        if tot > 0:
                            pct = int((comp / tot) * 100)
                            self.r.after(0, self.st_var.set, f"📥 Downloading {m}: {pct}%")
                            self.r.after(0, lambda v=pct: TASKBAR.set_progress(self.r, v, 100))
            if final_status and final_status != "success":
                raise Exception(f"unexpected status: {final_status}")
            self.r.after(0, lambda: TASKBAR.clear(self.r))
            self.r.after(0, self.st_var.set, f"✅ {m} Downloaded!")
            self.r.after(0, self.wl, f"✅ Download completed: {m}")
            self.rm()
        except Exception as e:
            self.r.after(0, lambda: TASKBAR.clear(self.r))
            self.r.after(0, self.wl, f"❌ Download error: {e}")

    def dlm(self):
        s = self.mn.get().strip()
        if not s or s.startswith("|") or s.startswith("⚠️"):
            return
        m_name = s.split(" ")[0]
        self.wl(f"🗑️ Deleting: {m_name}")
        threading.Thread(target=self._del, args=(m_name,), daemon=True).start()

    def _del(self, m):
        try:
            installed = get_installed_ollama_models(force_refresh=True)
            target_m = resolve_installed_model_name(m, installed)
            r = HTTP.delete(D_URL, json={"name": target_m, "model": target_m}, timeout=15)
            if r.status_code not in (200, 204) and target_m != m:
                r = HTTP.delete(D_URL, json={"name": m, "model": m}, timeout=15)
            if r.status_code in (200, 204):
                self.r.after(0, self.wl, f"🗑️ Model deleted: {target_m}")
                self.r.after(0, self.st_var.set, f"🗑️ {target_m} Deleted!")
            else:
                self.r.after(0, self.wl, f"⚠️ Model could not be deleted ({r.status_code}): {target_m}")
            self.r.after(0, lambda: self.mb.set(""))
            self.r.after(0, lambda: self.mn.set(""))
            self.rm()
        except Exception as e:
            self.r.after(0, self.wl, f"❌ Delete error: {e}")

    def c_f(self):
        self.sf_list.clear()
        self.fl.config(text="No file selected...")
        self.wl("🗑️ SRT queue cleared.")

    def s_f(self):
        files = filedialog.askopenfilenames(title="Select SRT Subtitle Files", filetypes=[("SRT Subtitle", "*.srt")])
        if files:
            for f in files:
                if f not in self.sf_list:
                    self.sf_list.append(f)
            self.fl.config(text=os.path.basename(self.sf_list[0]) if len(self.sf_list) == 1 else f"{len(self.sf_list)} SRT files in queue")
            self.wl(f"Total SRT files in queue: {len(self.sf_list)}")

    def s_o(self):
        d = filedialog.askdirectory()
        if d:
            self.of = d
            self.ol.config(text=d)
            self.wl(f"Output folder: {d}")

    def s_t(self):
        # YENİ: Çalışırken buton "DURDUR" olur (eskiden SRT kuyruğunu durdurmanın yolu yoktu).
        if self.running_main:
            self.abort_main = True
            self.bs.config(text=xui(self.lang_var.get(), "stopping"), state="disabled")
            return
        if not self.sf_list or not self.of:
            messagebox.showerror("Error", "Please select at least one SRT file and an output folder.")
            return
        mb_txt = self.mb.get().strip()
        if not mb_txt or mb_txt.startswith("|") or mb_txt.startswith("⚠️"):
            messagebox.showerror("Error", xui(self.lang_var.get(), "no_model"))
            return
        self.abort_main = False
        self.running_main = True
        self.bs.config(text=xui(self.lang_var.get(), "stop"), bg="#b71c1c", state="normal")
        threading.Thread(target=self.tr_batch, daemon=True).start()

    def _reset_main_button(self):
        if not self.r.winfo_exists():
            return
        t = UIT.get(self.lang_var.get(), UIT["English"])
        self.bs.config(text=t[9], bg="#333333", state="normal")

    def tr_batch(self):
        batch_sess = create_http_session()
        try:
            wait_until_ollama_ready(
                abort_check=lambda: self.abort_main,
                status_cb=lambda msg: self.r.after(0, self.st_var.set, msg)
            )
            if self.abort_main:
                return
            installed = get_installed_ollama_models(force_refresh=True)
            if not installed:
                raise Exception("No running Ollama server / no installed models found. Please start Ollama.")
            m = resolve_installed_model_name(self.mb.get().strip().split(" ")[0], installed)
            if m not in installed:
                raise OllamaModelMissing(xui(self.lang_var.get(), "model_missing", m=m))
            self.r.after(0, self.st_var.set, f"⏳ Model loading: {m}")
            ollama_preload(m, batch_sess)
            sl = L[self.sl.get()][0]
            tl = L[self.tl.get()][0]
            tc = L[self.tl.get()][1]
            cp = self.cp.get("1.0", tk.END).strip()
            chunk_len = int(self.cz_main.get())

            queue_snapshot = list(self.sf_list)
            total_files = len(queue_snapshot)
            for f_idx, srt_path in enumerate(queue_snapshot):
                if self.abort_main:
                    break
                f_name = os.path.basename(srt_path)
                self.r.after(0, self.wl, f"▶ [{f_idx + 1}/{total_files}] Processing ({ACTIVE_HOST}): {f_name}")
                raw_content = read_srt_file_smart(srt_path)

                parsed_items = parse_srt_blocks(raw_content)
                if not parsed_items:
                    self.r.after(0, self.wl, f"⚠️ Skipped (No valid SRT blocks): {f_name}")
                    continue

                chunks = [parsed_items[i:i + chunk_len] for i in range(0, len(parsed_items), chunk_len)]
                tot_chunks = max(1, len(chunks))
                tp = []

                for c_idx, ch_items in enumerate(chunks):
                    if self.abort_main:
                        break
                    file_fraction = (c_idx + 1) / float(tot_chunks)
                    overall_pct = int(((f_idx + file_fraction) / float(total_files)) * 100)
                    self.r.after(0, self.st_var.set, f"Translating ({f_idx + 1}/{total_files}): {f_name} ({int(file_fraction * 100)}%)")
                    self.r.after(0, self.pl.config, {"text": f"Progress: {overall_pct}%"})
                    self.r.after(0, self.part_lbl.config, {"text": f"File: {f_idx + 1}/{total_files} | Chunk: {c_idx + 1}/{tot_chunks}"})
                    self.r.after(0, lambda v=overall_pct: self.pg.configure(value=v))
                    self.r.after(0, lambda v=overall_pct: TASKBAR.set_progress(self.r, v, 100))
                    tp.append(translate_srt_chunk_locked(
                        ch_items, m, sl, tl, cp,
                        session=batch_sess,
                        abort_check=lambda: self.abort_main,
                        status_cb=lambda msg: self.r.after(0, self.st_var.set, msg)
                    ))

                if self.abort_main:
                    break
                safe_stem = sanitize_filename(os.path.splitext(f_name)[0], max_len=80)
                out_file = os.path.join(self.of, f"{safe_stem}_{tc}.srt")
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write("\n\n".join(tp) + "\n")
                self.r.after(0, self.wl, f"✅ [{f_idx + 1}/{total_files}] Saved -> {out_file}")

            unload_all_ollama_models(ACTIVE_HOST)
            if not self.abort_main:
                self.r.after(0, lambda: self.pg.configure(value=100))
                self.r.after(0, self.pl.config, {"text": "Progress: 100%"})
                self.r.after(0, self.st_var.set, f"✅ Queue Completed! ({total_files}/{total_files} Files)")
                self.r.after(0, self.wl, "🎉 ALL SRT QUEUE COMPLETED SUCCESSFULLY!")
                if self.var_shutdown_main.get():
                    trigger_pc_shutdown_with_cancel(self.r, self.var_shutdown_main, self.lang_var.get())
        except Exception as e:
            if not self.abort_main:
                self.r.after(0, self.st_var.set, "❌ Error!")
                self.r.after(0, self.wl, f"ERROR: {e}")
        finally:
            batch_sess.close()
            self.running_main = False
            if self.r.winfo_exists():
                self.r.after(0, lambda: TASKBAR.clear(self.r))
                self.r.after(0, self._reset_main_button)
                if self.abort_main:
                    self.r.after(0, self.st_var.set, "🛑 Stopped.")

if __name__ == '__main__':
    rt = tk.Tk()
    ap = AILocalTranslator(rt)
    rt.mainloop()
