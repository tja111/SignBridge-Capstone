import os
import sys
import time
import threading
from pathlib import Path
import subprocess
import json
import logging
import traceback

import cv2
import torch
import albumentations as A
try:
    from albumentations.pytorch import ToTensorV2 as A_ToTensorV2
except Exception:
    A_ToTensorV2 = getattr(A, "ToTensorV2")

from PIL import Image, ImageTk

from model import DETR
from utils.setup import get_colors
from utils.boxes import rescale_bboxes
from core.smoothing import SmoothingBuffer
from core.playback import text_to_sequence
from services.tts import TTSWorker
from word_classes import WORD_CLASSES

import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk  # pip install customtkinter

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resource_path(relative_path: str) -> Path:
    """Find bundled resources in both source and PyInstaller --onedir runs."""
    base = Path(getattr(sys, "_MEIPASS", PROJECT_ROOT))
    return base / relative_path


def writable_path(relative_path: str) -> Path:
    """Use the portable app folder when writable, otherwise LOCALAPPDATA."""
    app_root = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else PROJECT_ROOT
    target = app_root / relative_path
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target.parent / ".write_test", "a", encoding="utf-8"):
            pass
        (target.parent / ".write_test").unlink(missing_ok=True)
        return target
    except OSError:
        return Path(os.environ.get("LOCALAPPDATA", PROJECT_ROOT)) / "SignBridge" / relative_path


LOG_FILE = writable_path("logs/signbridge.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
APP_LOGGER = logging.getLogger("signbridge")

ASSETS_DIR = resource_path("assets")
ASSETS_ICON = ASSETS_DIR / "signbridge.ico"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
VENV_PYTHONW = PROJECT_ROOT / ".venv" / "Scripts" / "pythonw.exe"

ALPHABET_CKPT = "checkpoints/alphabet_model.pt"
WORDS_CKPT = "checkpoints/words/words_model.pt"
SIGNS_DIR = resource_path("assets/signs")
PREVIEW_W = 520
PREVIEW_H = 340
HIDE_CONSOLE = True
MAX_HISTORY_CHIPS = 30
PREFERENCES_FILE = writable_path("settings.json")


def load_preferences() -> dict:
    try:
        with open(PREFERENCES_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


class _ProgressCompat(ctk.CTkProgressBar):
    """Shim so existing configure(value=N) calls work with CTkProgressBar.

    The recognition loop sets progress via self.hold_progress.configure(value=pct)
    where pct is 0-100.  CTkProgressBar uses set(0.0-1.0), so we intercept
    configure() and normalise.
    """
    def configure(self, **kwargs):
        if "value" in kwargs:
            self.set(float(kwargs.pop("value")) / 100.0)
        if kwargs:
            super().configure(**kwargs)


class SignBridgeApp:
    def __init__(self, root: tk.Tk, cam_index: int = 0):
        self.root = root
        self._preferences = load_preferences()
        try:
            self.cam_index = int(self._preferences.get("camera_index", cam_index))
        except (TypeError, ValueError):
            self.cam_index = cam_index
        self._fullscreen_preview_window = None
        self._fullscreen_preview_canvas = None
        self._fullscreen_preview_image = None
        self._fullscreen_preview_image_id = None
        saved_theme = self._preferences.get("theme", "Light")
        if saved_theme not in ("Light", "Dark", "System"):
            saved_theme = "Light"
        self._preferences["theme"] = saved_theme
        ctk.set_appearance_mode(saved_theme)
        scale_text = self._preferences.get("font_scale", "100%")
        scale_value = {"100%": 1.0, "115%": 1.15, "130%": 1.30}.get(scale_text, 1.0)
        ctk.set_widget_scaling(scale_value)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        APP_LOGGER.info("Using device: %s", self.device.type.upper())
        self.root.title("SignBridge")
        try:
            if ASSETS_ICON.exists():
                self.root.iconbitmap(default=str(ASSETS_ICON))
        except Exception:
            pass

        # State
        self.running = False
        self.thread = None
        self.cap = None
        self.secondary_thread = None
        self.cap_secondary = None
        self.secondary_window_name = "SignBridge - Camera 2 Recognition"
        self.photo = None
        self.preview_target_w = PREVIEW_W
        self.preview_target_h = PREVIEW_H
        self.output_text = tk.StringVar(value="")
        self.detected_text = tk.StringVar(value="Detected: —")
        self.history_chips = []
        self.supported_signs_win = None
        self.render_interval = 1.0 / 30.0
        self._last_render_ts = 0.0
        self._fps_ts = time.time()
        self._fps_frames = 0
        self._fps_value = 0.0
        self._session_started_at = None
        self._session_saved_count = 0
        self.eval_mode = tk.BooleanVar(value=False)
        self.cv_preview = tk.BooleanVar(value=False)
        self.cv_window_name = "SignBridge Preview"
        self._photo_cache = []
        self.mode_banner = None

        self._model_loading = False
        self._model_lock = threading.Lock()
        self.active_mode = self._preferences.get("mode", "alphabet")
        if self.active_mode not in ("alphabet", "words"):
            self.active_mode = "alphabet"
        self.active_model = None
        self.active_classes = None
        self.active_colors = None
        self._alphabet_model = None
        self._words_model = None
        self._alphabet_classes = None
        self._words_classes = None
        self._alphabet_colors = None
        self._words_colors = None

        self.final_output = ""
        self.current_word = ""
        self.last_commit_letter = None
        self.last_commit_ts = 0.0
        self.last_accept_ts = None
        self.repeat_gap = 0.45
        self.repeat_cooldown = 1.0
        self.no_det_gap = 0.20
        self.min_streak = 6
        # Alphabet must be quick enough to spell a word, while full words need
        # a longer hold to prevent accidental transcript entries.
        self.alphabet_hold_seconds = 0.8
        self.word_hold_seconds = 2.5
        self.min_stable_ms = int(self.alphabet_hold_seconds * 1000)
        # A short release rearms the same letter; a longer pause finishes the
        # current word. This leaves room to reposition between letters.
        self.idle_timeout = 1.2
        self.last_seen_label = None
        self.streak = 0
        self.stable_since = 0.0
        self.last_none_ts = None
        self.rearm_ready = False
        self.auto_speak = tk.BooleanVar(value=False)
        self._last_auto_speak_ts = 0.0
        self._auto_speak_idle = 2.5  # seconds of silence before auto-speaking

        # Words-mode confidence accumulator
        # Each frame the model's best prediction is added to its score bucket;
        # all buckets decay each frame so stale evidence fades away.  When a
        # bucket crosses _word_commit_thresh the word is committed.
        self._word_scores: dict = {}      # label -> accumulated score
        self._word_commit_ts: dict = {}   # label -> last commit timestamp
        self._word_commit_thresh = 3.5    # total accumulated score to commit
        self._word_decay = 0.82           # per-frame score decay (~0.82^15 ≈ 0.05)
        self._word_cooldown = 2.0         # seconds before same word can commit again
        self._word_min_conf = 0.55        # reject no-sign false positives
        self._word_candidate = None
        self._word_candidate_since = 0.0
        self._word_last_committed = None
        self._word_armed = True

        # Temporal smoothing and TTS
        self.smoother = SmoothingBuffer(window=9, switch_votes=3, min_conf=0.5)
        self._speech_rate_var = tk.IntVar(value=160)
        self._speech_volume_var = tk.IntVar(value=90)
        self.voice_var = tk.StringVar(value=self._preferences.get("voice", "Default voice"))
        self.tts = TTSWorker(rate=self._speech_rate_var.get(),
                             volume=self._speech_volume_var.get() / 100)
        self.tts.start()

        # Model & transforms
        self.transforms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A_ToTensorV2(),
        ])
        # Words Mode can use a higher input resolution than Alphabet Mode.
        # Keep a separate transform so the alphabet model is never affected.
        self._words_image_size = 224
        self.word_transforms = A.Compose([
            A.Resize(self._words_image_size, self._words_image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A_ToTensorV2(),
        ])
        self.model = None
        self.CLASSES = []
        self.COLORS = []

        self._build_ui()
        self.set_mode(self.active_mode)
        if not self._preferences.get("onboarding_seen", False):
            self.root.after(700, lambda: self.show_help(first_run=True))

        self.root.bind_all("<space>", self._on_space)
        self.root.bind_all("<Return>", self._on_enter)
        self.root.bind_all("<BackSpace>", self._on_backspace)
        self.root.bind_all("<Control-BackSpace>", self._on_ctrl_backspace)
        self.root.bind_all("<Escape>", self._exit_fullscreen)
        self.root.bind_all("<c>", self._on_clear_key)
        self.root.bind_all("<C>", self._on_clear_key)

    # ── Styles ────────────────────────────────────────────────────────────────

    def _setup_styles(self):
        self._bg       = "#eef0f4"   # window background
        self._card_bg  = "white"     # card surface
        self._accent   = "#2563eb"   # primary blue
        self._border   = "#e2e8f0"   # subtle card border
        self._muted    = "#94a3b8"   # secondary text
        try:
            self.root.configure(fg_color=self._bg)
        except Exception:
            pass
        # Remaining ttk widgets (canvas chip frames) use white to match cards
        _s = ttk.Style()
        _s.configure("TFrame", background="white")
        # TLabel padding improves history/sequence chip height and spacing
        _s.configure("TLabel", background="white", padding=(10, 5))

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self._setup_styles()

        # ── Header ───────────────────────────────────────────────────────────
        self._header_frame = ctk.CTkFrame(
            self.root, fg_color="#1e2d45", corner_radius=0, height=60)
        self._header_frame.pack(fill="x")
        self._header_frame.pack_propagate(False)
        self.mode_banner = ctk.CTkLabel(
            self._header_frame, text="ALPHABET MODE",
            text_color="white", font=ctk.CTkFont(size=19, weight="bold"),
            fg_color="transparent")
        self.mode_banner.place(relx=0.5, rely=0.5, anchor="center")

        # ── Toolbar ──────────────────────────────────────────────────────────
        # FIX: tk.Frame instead of ctk.CTkFrame — CTkFrame defaults to height=200
        # and in some CTk versions disables pack propagation, inflating the row
        # to ~200px and creating the large visible gap above/below the buttons.
        # tk.Frame always resizes to fit its children (no minimum size issues).
        toolbar = tk.Frame(self.root, bg=self._bg)
        toolbar.pack(fill="x", padx=16, pady=(4, 4))

        ctk.CTkLabel(toolbar, text="Camera:", font=ctk.CTkFont(size=12),
                      text_color="#4b5563").pack(side="left")
        self.cam_var = tk.StringVar(value=str(self.cam_index))
        ctk.CTkComboBox(toolbar, variable=self.cam_var, values=["0", "1", "2", "3"],
                         width=75, height=36, font=ctk.CTkFont(size=12)).pack(
                             side="left", padx=(4, 10))
        self.mode_var = tk.BooleanVar(value=False)
        self.mode_toggle_btn = ctk.CTkButton(
            toolbar, text="Mode: Alphabet", command=self._on_mode_toggle,
            width=140, height=36, fg_color="#e8edf5", hover_color="#d5dce8",
            text_color="#1e2d45", border_width=1, border_color="#c5cfe0",
            font=ctk.CTkFont(size=12))
        self.mode_toggle_btn.pack(side="left", padx=(0, 20))
        # FIX: tk.Frame for the 1px separator — avoids another CTkFrame inflation
        tk.Frame(toolbar, bg="#d1d5db", width=1).pack(
            side="left", fill="y", pady=5, padx=(0, 20))
        self.start_btn = ctk.CTkButton(
            toolbar, text="▶  Start", command=self.start,
            width=110, height=36, corner_radius=8,
            fg_color="#2563eb", hover_color="#1d4ed8", text_color="white",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.start_btn.pack(side="left", padx=(0, 8))
        self.stop_btn = ctk.CTkButton(
            toolbar, text="⏹  Stop", command=self.stop,
            width=110, height=36, corner_radius=8,
            fg_color="#ef4444", hover_color="#dc2626", text_color="white",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.stop_btn.pack(side="left")
        self.stop_btn.configure(state="disabled")
        ctk.CTkButton(toolbar, text="Fullscreen", command=self.toggle_fullscreen,
                       width=100, height=30, corner_radius=7,
                       fg_color="transparent", hover_color="#e8edf5",
                       text_color="#475569", border_width=1, border_color=self._border,
                       font=ctk.CTkFont(size=10)).pack(side="left", padx=(12, 0))
        ctk.CTkButton(toolbar, text="Help", command=self.show_help,
                       width=70, height=30, corner_radius=7,
                       fg_color="transparent", hover_color="#e8edf5",
                       text_color="#475569", border_width=1, border_color=self._border,
                       font=ctk.CTkFont(size=10)).pack(side="left", padx=(6, 0))
        self.status_lbl = ctk.CTkLabel(toolbar, text="● Camera: Ready",
                                        font=ctk.CTkFont(size=11),
                                        text_color="#64748b")
        self.status_lbl.pack(side="right")
        ctk.CTkButton(toolbar, text="About", command=self.show_about,
                       width=70, height=30, corner_radius=7,
                       fg_color="transparent", hover_color="#e8edf5",
                       text_color="#475569", border_width=1, border_color=self._border,
                       font=ctk.CTkFont(size=10)).pack(side="right", padx=(0, 10))

        # ── Body: single grid coordinator for main + below ───────────────────
        # Use plain tk.Frame for layout containers — CTkFrame adds internal
        # canvas padding that creates a visible gap even with corner_radius=0.
        # row 0 (weight=5) — main preview area is DOMINANT
        # row 1 (weight=1) — lower sections get minimal space, user can scroll
        self.root.minsize(900, 750)
        body = tk.Frame(self.root, bg=self._bg)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=5)  # FIX: 5:1 ratio gives preview ~83% of body height
        body.rowconfigure(1, weight=1)

        # ── Main 2-column area ────────────────────────────────────────────────
        # tk.Frame eliminates CTkFrame internal padding; pady=0 closes the gap
        main = tk.Frame(body, bg=self._bg)
        main.grid(row=0, column=0, sticky="nsew", padx=16, pady=(0, 4))
        main.columnconfigure(0, weight=62, uniform="col")
        main.columnconfigure(1, weight=38, uniform="col")
        main.rowconfigure(0, weight=1)

        # ── Left card: Sign Recognition ───────────────────────────────────────
        left = ctk.CTkFrame(main, fg_color=self._card_bg, corner_radius=12,
                             border_width=1, border_color=self._border)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        # Section title with blue accent bar
        ltitle = ctk.CTkFrame(left, fg_color="transparent")
        ltitle.grid(row=0, column=0, sticky="ew", padx=14, pady=(8, 6))
        ctk.CTkFrame(ltitle, fg_color=self._accent, width=3, height=18,
                      corner_radius=2).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(ltitle, text="Sign Recognition",
                      font=ctk.CTkFont(size=13, weight="bold"),
                      text_color="#1e2d45").pack(side="left")

        # Camera preview — FIX: Ensure canvas fills available space and resizes dynamically
        # tk.Canvas keeps existing video-frame drawing logic untouched
        preview_wrap = ctk.CTkFrame(left, fg_color="#0c0c0c", corner_radius=8,
                                     border_width=1, border_color="#2a2a2a")
        preview_wrap.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 10))
        preview_wrap.rowconfigure(0, weight=1)
        preview_wrap.columnconfigure(0, weight=1)

        # FIX: Canvas with proper sizing — it will expand to fill preview_wrap
        self.preview = tk.Canvas(preview_wrap, width=PREVIEW_W, height=PREVIEW_H,
                                  bg="#0c0c0c", highlightthickness=0)
        self.preview.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.preview_image_id = None

        # FIX: Configure event fires when canvas size changes; update target render dimensions
        def _on_preview_resize_handler(event):
            if event.width > 20 and event.height > 20:  # Ignore spurious tiny sizes
                self.preview_target_w = max(100, event.width - 4)  # Account for padding
                self.preview_target_h = max(80, event.height - 4)
        self.preview.bind("<Configure>", _on_preview_resize_handler)

        # Camera placeholder — centered text shown before first frame is drawn.
        # Video frames naturally stack on top; delete("all") in stop() clears it too.
        self.preview.create_text(
            PREVIEW_W // 2, PREVIEW_H // 2,
            text="Camera Preview\n\nPress Start to begin recognition",
            fill="#cbd5e1", font=("Segoe UI", 10), anchor="center",
            justify="center", tags="preview_hint")

        def _reposition_hint(event):
            items = self.preview.find_withtag("preview_hint")
            for item in items:
                self.preview.coords(item, event.width // 2, event.height // 2)

        self.preview.bind("<Configure>", _reposition_hint, add="+")

        # ── Detection status row (row 2) — live indicator panel ──────────────
        # Bordered sub-card inside the left column.
        # Row 0 : detected sign (bold) | mode pill badge
        # 1px rule
        # Row 2 : confidence score      | colour-coded status dot
        detect_row = ctk.CTkFrame(left, fg_color="#f8fafc", corner_radius=8,
                                   border_width=1, border_color=self._border)
        detect_row.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 8))
        detect_row.columnconfigure(0, weight=1)
        detect_row.columnconfigure(1, weight=0)

        # Row 0 left: detected sign — detected_text StringVar keeps this current
        self.detected_lbl = ctk.CTkLabel(detect_row, text="Detected: —",
                                          font=ctk.CTkFont(size=14, weight="bold"),
                                          text_color="#374151", anchor="w")
        self.detected_lbl.grid(row=0, column=0, sticky="w", padx=(10, 0), pady=(8, 4))
        self.detected_text.trace_add(
            "write",
            lambda *_: self.detected_lbl.configure(text=self.detected_text.get()))

        # Row 0 right: mode pill badge — _update_mode_indicator() swaps text + colors
        self._mode_badge = ctk.CTkFrame(detect_row, fg_color="#eff6ff", corner_radius=4,
                                         border_width=1, border_color="#bfdbfe")
        self._mode_badge.grid(row=0, column=1, sticky="e", padx=(4, 10), pady=(8, 4))
        self.mode_indicator_lbl = ctk.CTkLabel(self._mode_badge, text="Mode: Alphabet",
                                                font=ctk.CTkFont(size=9, weight="bold"),
                                                text_color="#1d4ed8", fg_color="transparent",
                                                padx=8, pady=3)
        self.mode_indicator_lbl.pack()

        # 1px rule separating the sign label from the metrics row
        tk.Frame(detect_row, bg=self._border, height=1).grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=8)

        # Row 2 left: confidence score — updated by _set_confidence() from _loop()
        self.conf_lbl = ctk.CTkLabel(detect_row, text="Confidence: —",
                                      font=ctk.CTkFont(size=11), text_color="#6b7280",
                                      anchor="w")
        self.conf_lbl.grid(row=2, column=0, sticky="w", padx=(10, 0), pady=(4, 8))

        # Row 2 right: status dot — updated by _set_status() from _loop()
        # Waiting (gray) · Detecting (blue) · Stable (green)
        # Low Confidence (orange) · No Sign Detected (red)
        self.status_dot_lbl = ctk.CTkLabel(detect_row, text="● Waiting",
                                            font=ctk.CTkFont(size=11), text_color="#9ca3af",
                                            anchor="e")
        self.status_dot_lbl.grid(row=2, column=1, sticky="e", padx=(4, 10), pady=(4, 8))

        self.last_saved_lbl = ctk.CTkLabel(
            detect_row, text="Last saved: —", font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#0f766e", anchor="w")
        self.last_saved_lbl.grid(row=3, column=0, columnspan=2,
                                 sticky="w", padx=10, pady=(0, 2))

        self.instruction_lbl = ctk.CTkLabel(
            detect_row,
            text="Hold a letter for 0.8s • briefly release between letters • pause 1.2s to finish",
            font=ctk.CTkFont(size=9), text_color="#64748b", anchor="w")
        self.instruction_lbl.grid(row=4, column=0, columnspan=2,
                                  sticky="w", padx=10, pady=(0, 8))

        # ── Action buttons row (row 3) ────────────────────────────────────────
        ctrl = ctk.CTkFrame(left, fg_color="transparent")
        ctrl.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 4))
        ctk.CTkLabel(ctrl, text="Speech Controls", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#1e2d45").pack(side="left", padx=(0, 12))
        ctk.CTkButton(ctrl, text="🔊  Speak", command=self.on_speak,
                       width=90, height=32, corner_radius=8,
                       fg_color=self._accent, hover_color="#1d4ed8",
                       font=ctk.CTkFont(size=11, weight="bold")).pack(
                           side="left", padx=(0, 4))
        ctk.CTkButton(ctrl, text="✕  Clear", command=self.on_clear,
                       width=80, height=32, corner_radius=8,
                       fg_color="transparent", hover_color="#fee2e2",
                       text_color="#ef4444", border_width=1, border_color="#fca5a5",
                       font=ctk.CTkFont(size=11)).pack(side="left", padx=(0, 12))
        self.undo_btn = ctk.CTkButton(ctrl, text="Undo Letter", command=self._on_backspace,
                       width=105, height=32, corner_radius=8,
                       fg_color="transparent", hover_color="#fef3c7",
                       text_color="#92400e", border_width=1, border_color="#fcd34d",
                       font=ctk.CTkFont(size=11))
        self.undo_btn.pack(side="left", padx=(0, 12))
        self.undo_btn.configure(state="disabled")
        self.finish_word_btn = ctk.CTkButton(ctrl, text="Space / Finish Word", command=self._finalize_word,
                       width=125, height=32, corner_radius=8,
                       fg_color="#eff6ff", hover_color="#dbeafe",
                       text_color="#1d4ed8", border_width=1, border_color="#bfdbfe",
                       font=ctk.CTkFont(size=11))
        self.finish_word_btn.pack(side="left", padx=(0, 12))
        self.finish_word_btn.configure(state="disabled")
        ctk.CTkSwitch(ctrl, text="Auto-Speak", variable=self.auto_speak,
                       onvalue=True, offvalue=False,
                       font=ctk.CTkFont(size=11)).pack(side="left")
        speech = ctk.CTkFrame(left, fg_color="transparent")
        speech.grid(row=4, column=0, sticky="ew", padx=14, pady=(2, 2))
        ctk.CTkLabel(speech, text="Voice:", font=ctk.CTkFont(size=10, weight="bold"),
                     text_color="#4b5563").pack(side="left")
        self.voice_selector = ctk.CTkComboBox(
            speech, variable=self.voice_var, values=["Default voice"], width=170,
            height=28, command=self._on_voice_selected, font=ctk.CTkFont(size=10))
        self.voice_selector.pack(side="left", padx=(6, 14))
        ctk.CTkLabel(speech, text="Speech speed:", font=ctk.CTkFont(size=10),
                     text_color="#4b5563").pack(side="left", padx=(0, 4))
        ctk.CTkSlider(speech, from_=80, to=260, number_of_steps=18,
                      variable=self._speech_rate_var, width=100,
                      command=self._on_speech_rate_change).pack(side="left", padx=(0, 5))
        self.speech_rate_lbl = ctk.CTkLabel(speech, text="160 WPM", width=55,
                                             font=ctk.CTkFont(size=10), text_color=self._muted)
        self.speech_rate_lbl.pack(side="left")
        ctk.CTkLabel(speech, text="Volume:", font=ctk.CTkFont(size=10),
                     text_color="#4b5563").pack(side="left", padx=(14, 4))
        ctk.CTkSlider(speech, from_=0, to=100, number_of_steps=20,
                      variable=self._speech_volume_var, width=80,
                      command=self._on_volume_change).pack(side="left", padx=(0, 5))
        self.volume_lbl = ctk.CTkLabel(speech, text="90%", width=36,
                                       font=ctk.CTkFont(size=10), text_color=self._muted)
        self.volume_lbl.pack(side="left")
        self.root.after(500, self._refresh_voice_options)

        # ── Sensitivity row (row 4) ───────────────────────────────────────────
        sens = ctk.CTkFrame(left, fg_color="transparent")
        sens.grid(row=5, column=0, sticky="ew", padx=14, pady=(2, 2))
        self.sensitivity_row = sens
        sens.grid_remove()  # only relevant to Words Mode
        ctk.CTkLabel(sens, text="Word Sensitivity:",
                      font=ctk.CTkFont(size=10, weight="bold"),
                      text_color="#4b5563").pack(side="left")
        # _thresh_var DoubleVar — slider syncs it; command pushes to _word_commit_thresh
        self._thresh_var = tk.DoubleVar(value=self._word_commit_thresh)
        ctk.CTkSlider(sens, from_=1.0, to=8.0, variable=self._thresh_var, width=110,
                       command=lambda v: setattr(self, "_word_commit_thresh",
                                                  float(v))).pack(side="left", padx=(8, 6))
        ctk.CTkLabel(sens, text="lower = easier trigger",
                      font=ctk.CTkFont(size=9), text_color="#9ca3af").pack(side="left")

        self.hold_progress_label = ctk.CTkLabel(
            left, text="Letter hold progress (0.8s)", anchor="w",
            font=ctk.CTkFont(size=9), text_color=self._muted)
        self.hold_progress_label.grid(row=6, column=0, sticky="ew", padx=14, pady=(2, 0))
        # Shared hold-progress bar for both Alphabet and Words Mode.
        self.hold_progress = _ProgressCompat(
            left, orientation="horizontal", height=6,
            fg_color="#e5e7eb", progress_color=self._accent)
        self.hold_progress.set(0)
        self.hold_progress.grid(row=7, column=0, sticky="ew", padx=14, pady=(0, 12))

        # ── Right card: Recognized Text ───────────────────────────────────────
        right = ctk.CTkFrame(main, fg_color=self._card_bg, corner_radius=12,
                              border_width=1, border_color=self._border)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        rtitle = ctk.CTkFrame(right, fg_color="transparent")
        rtitle.grid(row=0, column=0, sticky="ew", padx=14, pady=(8, 6))
        ctk.CTkFrame(rtitle, fg_color=self._accent, width=3, height=18,
                      corner_radius=2).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(rtitle, text="Recognized Sentence",
                      font=ctk.CTkFont(size=13, weight="bold"),
                      text_color="#1e2d45").pack(side="left")

        # Inner output panel — off-white with border, feels like a dedicated text display
        output_area = ctk.CTkFrame(right, fg_color="#f8fafc", corner_radius=8,
                                    border_width=1, border_color=self._border)
        output_area.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 4))
        output_area.rowconfigure(0, weight=1)
        output_area.rowconfigure(1, weight=0)   # footer row
        output_area.columnconfigure(0, weight=1)

        # output_text StringVar traced; placeholder until text arrives
        self.output_lbl = ctk.CTkLabel(
            output_area,
            text="Start signing to see recognized text here...",
            anchor="nw", wraplength=380,
            font=ctk.CTkFont(size=21), text_color="#6b7280",
            fg_color="transparent", padx=14, pady=16)
        self.output_lbl.grid(row=0, column=0, sticky="nsew")

        # Word/char count footer — gives the panel a sense of activity even when short
        self._char_count_lbl = ctk.CTkLabel(
            output_area, text="",
            font=ctk.CTkFont(size=9), text_color=self._muted,
            anchor="e", padx=10, pady=4, fg_color="transparent")
        self._char_count_lbl.grid(row=1, column=0, sticky="ew")
        self.session_stats_lbl = ctk.CTkLabel(
            output_area, text="Session: not started", anchor="w",
            font=ctk.CTkFont(size=9), text_color="#64748b", padx=10, pady=3)
        self.session_stats_lbl.grid(row=2, column=0, sticky="ew")
        ctk.CTkLabel(output_area, text="Camera frames are processed locally and are not saved.",
                     anchor="w", font=ctk.CTkFont(size=8), text_color="#94a3b8",
                     padx=10, pady=3).grid(row=3, column=0, sticky="ew")

        def _on_output_change(*_):
            val = self.output_text.get()
            if val:
                self.output_lbl.configure(text=val, text_color="#111827")
                nw = len(val.split())
                nc = len(val.replace(" ", ""))
                self._char_count_lbl.configure(
                    text=f"{nw} word{'s' if nw != 1 else ''}  ·  {nc} char{'s' if nc != 1 else ''}")
            else:
                self.output_lbl.configure(
                    text="Start signing to see recognized text here...",
                    text_color="#6b7280")
                self._char_count_lbl.configure(text="")

        self.output_text.trace_add("write", _on_output_change)
        output_area.bind("<Configure>",
                          lambda e: self.output_lbl.configure(
                              wraplength=max(160, e.width - 32)))

        # Save Transcript — right-aligned at card bottom
        transcript_actions = ctk.CTkFrame(right, fg_color="transparent")
        transcript_actions.grid(row=2, column=0, sticky="e", padx=14, pady=(4, 12))
        ctk.CTkButton(
            transcript_actions, text="Copy Text", command=self.copy_text,
            width=100, height=30, corner_radius=6,
            fg_color="transparent", text_color="#374151",
            hover_color="#f1f5f9", border_width=1, border_color=self._border,
            font=ctk.CTkFont(size=11)).pack(side="left", padx=(0, 6))
        ctk.CTkButton(
            transcript_actions, text="💾  Save Transcript", command=self.on_save,
            width=140, height=30, corner_radius=6,
            fg_color="transparent", text_color="#374151",
            hover_color="#f1f5f9", border_width=1, border_color=self._border,
            font=ctk.CTkFont(size=11)).pack(side="left")

        # ── Below sections — scrollable container in grid row 1 ────────────────
        # FIX: Use a scrollable frame so lower sections don't push preview up.
        # When Advanced Settings expand, user scrolls instead of layout breaking.
        # FIX: tk.Frame so no CTkFrame internal padding steals height from below sections
        below_outer = tk.Frame(body, bg=self._bg)
        below_outer.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 4))
        below_outer.columnconfigure(0, weight=1)
        below_outer.rowconfigure(0, weight=1)

        # Scrollable canvas for lower sections
        below_canvas = tk.Canvas(
            below_outer, bg=self._bg, highlightthickness=0,
            relief="flat", borderwidth=0
        )
        below_canvas.grid(row=0, column=0, sticky="nsew")
        below_scrollbar = ttk.Scrollbar(
            below_outer, orient="vertical", command=below_canvas.yview
        )
        below_scrollbar.grid(row=0, column=1, sticky="ns")
        below_canvas.configure(yscrollcommand=below_scrollbar.set)

        # Frame inside scrollable canvas — holds all lower section widgets.
        # tk.Frame avoids CTkFrame internal padding so cards fill the full width.
        below = tk.Frame(below_canvas, bg=self._bg)
        _below_win_id = below_canvas.create_window((0, 0), window=below, anchor="nw")

        # FIX: Bind to the CANVAS resize so below frame fills the full canvas width.
        # Without this the inner frame only gets its natural/minimum width, making
        # sections appear narrow and Advanced Settings get pushed off the right edge.
        def _on_canvas_resize(event):
            if event.width > 1:
                below_canvas.itemconfig(_below_win_id, width=event.width)
        below_canvas.bind("<Configure>", _on_canvas_resize)

        # FIX: Separate binding on below frame to update the scrollregion only
        # (updating scrollregion from below's own Configure event is the correct pattern)
        def _on_below_frame_configure(event):
            below_canvas.configure(scrollregion=below_canvas.bbox("all"))
        below.bind("<Configure>", _on_below_frame_configure)

        # Mouse wheel scrolling — bound to canvas so wheel works over lower sections
        def _on_mousewheel(event):
            below_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        below_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Sign History card
        hist_card = ctk.CTkFrame(below, fg_color=self._card_bg, corner_radius=12,
                                  border_width=1, border_color=self._border)
        hist_card.pack(fill="x", pady=(0, 8))
        htitle = ctk.CTkFrame(hist_card, fg_color="transparent")
        htitle.pack(anchor="w", padx=14, pady=(12, 6))
        ctk.CTkFrame(htitle, fg_color=self._accent, width=3, height=14,
                      corner_radius=2).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(htitle, text="Sign History",
                      font=ctk.CTkFont(size=11, weight="bold"),
                      text_color="#1e2d45").pack(side="left")
        self.history_canvas = tk.Canvas(hist_card, height=32, highlightthickness=0,
                                        background=self._card_bg)
        self.history_canvas.pack(side="top", fill="x", expand=True, padx=14, pady=(0, 2))
        # Empty-state hint — sits behind chip windows; covered naturally as chips accumulate
        self.history_canvas.create_text(
            14, 16,
            text="Signed characters appear here as you sign",
            fill="#adb5bd", font=("Segoe UI", 9, "italic"), anchor="w",
            tags="hist_hint")
        self.history_scroll = ttk.Scrollbar(hist_card, orient="horizontal",
                                             command=self.history_canvas.xview)
        self.history_scroll.pack(side="bottom", fill="x", padx=14, pady=(0, 6))
        self.history_canvas.configure(xscrollcommand=self.history_scroll.set)
        self.history_inner = ttk.Frame(self.history_canvas)
        self.history_canvas.create_window((0, 0), window=self.history_inner, anchor="nw")
        self.history_inner.bind(
            "<Configure>",
            lambda e: self.history_canvas.configure(
                scrollregion=self.history_canvas.bbox("all")))

        # "Text to Signs" divider
        self._build_divider(below, "Text to Signs", pady=(0, 8))

        # Text input and sign playback card
        comms_card = ctk.CTkFrame(below, fg_color=self._card_bg, corner_radius=12,
                                   border_width=1, border_color=self._border)
        comms_card.pack(fill="x", pady=(0, 8))
        ctitle = ctk.CTkFrame(comms_card, fg_color="transparent")
        ctitle.pack(anchor="w", padx=14, pady=(12, 6))
        ctk.CTkFrame(ctitle, fg_color=self._accent, width=3, height=14,
                      corner_radius=2).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(ctitle, text="Text Input and Playback",
                      font=ctk.CTkFont(size=11, weight="bold"),
                      text_color="#1e2d45").pack(side="left")
        comms_row = ctk.CTkFrame(comms_card, fg_color="transparent")
        comms_row.pack(fill="x", padx=14, pady=(0, 12))
        self._input_placeholder = "Type text to convert into signs…"
        self._input_placeholder_active = True
        self.input_text = tk.StringVar(value=self._input_placeholder)
        self.text_entry = ctk.CTkEntry(comms_row, textvariable=self.input_text,
                                       font=ctk.CTkFont(size=12), height=36,
                                       text_color="#94a3b8")
        self.text_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.text_entry.bind("<FocusIn>", self._clear_text_placeholder, add="+")
        self.text_entry.bind("<FocusOut>", self._restore_text_placeholder, add="+")
        # Show Signs — primary action for this section
        ctk.CTkButton(comms_row, text="👁  Show Signs", command=self.on_show_signs,
                       width=110, height=36, corner_radius=8,
                       fg_color=self._accent, hover_color="#1d4ed8", text_color="white",
                       font=ctk.CTkFont(size=11, weight="bold")).pack(
                           side="right", padx=(4, 0))
        # Voice — secondary; outlined so it reads as a supporting option
        ctk.CTkButton(comms_row, text="🎙  Voice", command=self.on_voice,
                       width=90, height=36, corner_radius=8,
                       fg_color="transparent", hover_color="#e8edf5",
                       text_color="#374151", border_width=1, border_color=self._border,
                       font=ctk.CTkFont(size=11)).pack(side="right", padx=(0, 4))
        ctk.CTkButton(comms_row, text="Supported Signs", command=self.show_supported_signs,
                       width=120, height=36, corner_radius=8,
                       fg_color="transparent", hover_color="#e8edf5",
                       text_color="#374151", border_width=1, border_color=self._border,
                       font=ctk.CTkFont(size=11, weight="bold")).pack(side="right", padx=(0, 4))
        ctk.CTkLabel(comms_card,
                     text="Alphabet A–Z • Words: 8 supported signs • Camera required",
                     font=ctk.CTkFont(size=9), text_color="#64748b").pack(
                         anchor="w", padx=14, pady=(0, 10))

        # ── Advanced & Developer Settings ─────────────────────────────────────
        # FIX: Uses grid_remove/grid to collapse/expand WITHOUT affecting layout order.
        # When expanded, it stays INSIDE the scrollable 'below' frame, so user scrolls
        # instead of the layout breaking. Advanced section is deliberately understated.
        adv_outer = ctk.CTkFrame(below, fg_color=self._bg, corner_radius=8,
                                  border_width=1, border_color=self._border)
        adv_outer.pack(fill="x", pady=(4, 8))
        adv_outer.columnconfigure(0, weight=1)

        # Row 0: always-visible toggle button (built first so _toggle_adv can ref it)
        adv_btn = ctk.CTkButton(
            adv_outer, text="⚙  Advanced & Developer Settings    ›",
            anchor="w", fg_color="transparent", text_color="#9ca3af",
            hover_color="#ebebeb", font=ctk.CTkFont(size=10),
            corner_radius=8, height=28)
        adv_btn.grid(row=0, column=0, sticky="ew", padx=4, pady=2)

        # Row 1: collapsible content frame — starts hidden via grid_remove()
        # FIX: Expanding Advanced Settings adds content to scrollable 'below',
        # not to the main area, so preview stays fixed at top with scroll access to settings.
        adv_content = ctk.CTkFrame(adv_outer, fg_color="transparent")
        adv_content.columnconfigure(0, weight=1)
        adv_content.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))
        adv_content.grid_remove()          # collapsed by default

        # Camera Settings (packed inside adv_content)
        cam_lf = ctk.CTkFrame(adv_content, fg_color="#efefef", corner_radius=6)
        cam_lf.pack(fill="x", pady=(4, 4))
        ctk.CTkLabel(cam_lf, text="Camera Settings",
                      font=ctk.CTkFont(size=10, weight="bold"),
                      text_color="#6b7280").pack(anchor="w", padx=10, pady=(6, 3))
        theme_row = ctk.CTkFrame(cam_lf, fg_color="transparent")
        theme_row.pack(fill="x", padx=10, pady=(0, 4))
        ctk.CTkLabel(theme_row, text="Appearance:",
                      font=ctk.CTkFont(size=10), text_color="#6b7280").pack(side="left")
        self.theme_var = tk.StringVar(value=self._preferences.get("theme", "Light"))
        ctk.CTkComboBox(theme_row, variable=self.theme_var, values=["Light", "Dark", "System"],
                         width=110, height=26, command=self._on_theme_selected,
                         font=ctk.CTkFont(size=10)).pack(side="left", padx=(6, 0))
        ctk.CTkLabel(theme_row, text="Text size:",
                      font=ctk.CTkFont(size=10), text_color="#6b7280").pack(side="left", padx=(18, 0))
        self.font_scale_var = tk.StringVar(value=self._preferences.get("font_scale", "100%"))
        ctk.CTkComboBox(theme_row, variable=self.font_scale_var, values=["100%", "115%", "130%"],
                         width=85, height=26, command=self._on_font_scale_selected,
                         font=ctk.CTkFont(size=10)).pack(side="left", padx=(6, 0))
        cam_row = ctk.CTkFrame(cam_lf, fg_color="transparent")
        cam_row.pack(fill="x", padx=10, pady=(0, 6))
        ctk.CTkLabel(cam_row, text="Backend:",
                      font=ctk.CTkFont(size=10), text_color="#6b7280").pack(side="left")
        self.backend_var = tk.StringVar(value=self._preferences.get("backend", "DirectShow"))
        ctk.CTkComboBox(cam_row, variable=self.backend_var,
                         values=["Default", "DirectShow", "MSMF"],
                         width=120, height=26, font=ctk.CTkFont(size=10)).pack(
                             side="left", padx=(4, 12))
        ctk.CTkLabel(cam_row, text="Resolution:",
                      font=ctk.CTkFont(size=10), text_color="#6b7280").pack(side="left")
        self.res_var = tk.StringVar(value=self._preferences.get("resolution", "640x480"))
        ctk.CTkComboBox(cam_row, variable=self.res_var,
                         values=["640x480", "1280x720"],
                         width=110, height=26, font=ctk.CTkFont(size=10)).pack(
                             side="left", padx=(4, 12))
        self.mjpg_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(cam_row, text="MJPG", variable=self.mjpg_var,
                         onvalue=True, offvalue=False,
                         font=ctk.CTkFont(size=10)).pack(side="left", padx=(0, 8))
        self.lowlat_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(cam_row, text="Low Latency", variable=self.lowlat_var,
                         onvalue=True, offvalue=False,
                         font=ctk.CTkFont(size=10)).pack(side="left", padx=(0, 8))
        ctk.CTkCheckBox(cam_row, text="OpenCV Preview", variable=self.cv_preview,
                         onvalue=True, offvalue=False,
                         font=ctk.CTkFont(size=10)).pack(side="left", padx=(0, 8))
        ctk.CTkCheckBox(cam_row, text="Eval Mode (FPS)", variable=self.eval_mode,
                         onvalue=True, offvalue=False,
                         font=ctk.CTkFont(size=10)).pack(side="left")
        ctk.CTkButton(cam_lf, text="Reset Camera", command=self.reset_camera,
                       width=100, height=26, corner_radius=6,
                       fg_color="transparent", hover_color="#dbeafe",
                       text_color="#1d4ed8", border_width=1, border_color="#bfdbfe",
                       font=ctk.CTkFont(size=10)).pack(anchor="w", padx=10, pady=(0, 6))
        secondary_row = ctk.CTkFrame(cam_lf, fg_color="transparent")
        secondary_row.pack(fill="x", padx=8, pady=(0, 7))
        self.dual_camera_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(secondary_row, text="Enable Camera 2 recognition window",
                         variable=self.dual_camera_var, onvalue=True, offvalue=False,
                         font=ctk.CTkFont(size=10)).pack(side="left")
        ctk.CTkLabel(secondary_row, text="Camera 2:", font=ctk.CTkFont(size=10),
                     text_color="#6b7280").pack(side="left", padx=(14, 4))
        self.secondary_cam_var = tk.StringVar(value="1")
        ctk.CTkComboBox(secondary_row, variable=self.secondary_cam_var,
                         values=["0", "1", "2", "3"], width=70, height=26,
                         font=ctk.CTkFont(size=10)).pack(side="left")

        # Developer Tools (packed inside adv_content)
        dev_lf = ctk.CTkFrame(adv_content, fg_color="#efefef", corner_radius=6)
        dev_lf.pack(fill="x", pady=(0, 0))
        ctk.CTkLabel(dev_lf, text="Developer Tools",
                      font=ctk.CTkFont(size=10, weight="bold"),
                      text_color="#6b7280").pack(anchor="w", padx=10, pady=(6, 3))
        dev_row = ttk.Frame(dev_lf)
        dev_row.pack(fill="x", padx=10, pady=(0, 6))
        self._add_command_buttons(dev_row)
        diag_row = ctk.CTkFrame(dev_lf, fg_color="transparent")
        diag_row.pack(fill="x", padx=10, pady=(0, 7))
        ctk.CTkButton(diag_row, text="Open Logs", command=self.open_logs,
                       width=90, height=26, corner_radius=6,
                       font=ctk.CTkFont(size=10)).pack(side="left", padx=(0, 6))
        ctk.CTkButton(diag_row, text="Copy Diagnostics", command=self.copy_diagnostics,
                       width=120, height=26, corner_radius=6,
                       font=ctk.CTkFont(size=10)).pack(side="left")

        # ── Toggle: grid_remove/grid so position is remembered, no re-ordering ─
        _adv_visible = tk.BooleanVar(value=False)

        def _toggle_adv():
            if _adv_visible.get():
                adv_content.grid_remove()
                _adv_visible.set(False)
                adv_btn.configure(text="⚙  Advanced & Developer Settings    ›")
            else:
                adv_content.grid()   # restores row=1 position remembered by grid_remove
                _adv_visible.set(True)
                adv_btn.configure(text="⚙  Advanced & Developer Settings    ⌄")

        adv_btn.configure(command=_toggle_adv)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_divider(self, parent, text, **pack_kwargs):
        frame = ctk.CTkFrame(parent, fg_color="transparent", corner_radius=0)
        frame.pack(fill="x", **pack_kwargs)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(2, weight=1)
        ctk.CTkFrame(frame, height=1, fg_color=self._border,
                      corner_radius=0).grid(
                          row=0, column=0, sticky="ew", padx=(0, 10), pady=6)
        ctk.CTkLabel(frame, text=text, text_color="#60a5fa",
                      font=ctk.CTkFont(size=9, weight="bold")).grid(row=0, column=1)
        ctk.CTkFrame(frame, height=1, fg_color=self._border,
                      corner_radius=0).grid(
                          row=0, column=2, sticky="ew", padx=(10, 0), pady=6)

    def _on_close(self):
        self._save_preferences()
        self.stop()
        try:
            self.tts.stop()
        except Exception:
            pass
        self.root.destroy()

    def _save_preferences(self):
        values = {
            "camera_index": getattr(self, "cam_var", tk.StringVar(value=str(self.cam_index))).get(),
            "mode": self.active_mode,
            "voice": getattr(self, "voice_var", tk.StringVar(value="Default voice")).get(),
            "theme": getattr(self, "theme_var", tk.StringVar(value="Light")).get(),
            "font_scale": getattr(self, "font_scale_var", tk.StringVar(value="100%")).get(),
            "backend": getattr(self, "backend_var", tk.StringVar(value="DirectShow")).get(),
            "resolution": getattr(self, "res_var", tk.StringVar(value="640x480")).get(),
        }
        try:
            PREFERENCES_FILE.write_text(json.dumps(values, indent=2), encoding="utf-8")
        except OSError:
            APP_LOGGER.warning("Could not save preferences", exc_info=True)

    def _on_theme_selected(self, theme):
        ctk.set_appearance_mode(theme)
        self._save_preferences()

    def _on_font_scale_selected(self, scale_text):
        ctk.set_widget_scaling({"100%": 1.0, "115%": 1.15, "130%": 1.30}[scale_text])
        self._save_preferences()

    def show_help(self, first_run=False):
        messagebox.showinfo(
            "How to use SignBridge",
            "1. Choose Alphabet or Words Mode.\n"
            "2. Select a camera, then press Start.\n"
            "3. Hold an alphabet letter for 0.8 seconds.\n"
            "4. Hold a word sign for 2.5 seconds.\n\n"
            "Alphabet shortcuts:\n"
            "Space: finish the current word\n"
            "Backspace: undo the last letter\n"
            "Esc: leave the full-screen camera view"
        )
        if first_run:
            self._preferences["onboarding_seen"] = True
            self._save_preferences()

    def _update_session_stats(self):
        if self._session_started_at is None:
            self.session_stats_lbl.configure(text="Session: not started")
            return
        elapsed = int(time.time() - self._session_started_at)
        minutes, seconds = divmod(elapsed, 60)
        self.session_stats_lbl.configure(
            text=f"Session: {self._session_saved_count} saved • {minutes}:{seconds:02d} • {self._mode_label(self.active_mode)}")
        if self.running:
            self.root.after(1000, self._update_session_stats)

    def _record_saved_sign(self):
        self._session_saved_count += 1
        self._run_on_ui(self._update_session_stats)

    def open_logs(self):
        try:
            os.startfile(str(LOG_FILE))
        except OSError:
            messagebox.showinfo("Logs", f"Logs are saved at:\n{LOG_FILE}")

    def copy_diagnostics(self):
        device_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU mode"
        details = (f"SignBridge diagnostics\nDevice: {device_name}\n"
                   f"Mode: {self._mode_label(self.active_mode)}\n"
                   f"Camera: {self.cam_var.get()}\nLog: {LOG_FILE}")
        self.root.clipboard_clear()
        self.root.clipboard_append(details)
        self.root.update()
        messagebox.showinfo("Diagnostics", "Diagnostics copied to the clipboard.")

    def toggle_fullscreen(self):
        if self._fullscreen_preview_window is not None:
            self._exit_fullscreen()
            return
        window = tk.Toplevel(self.root)
        window.title("SignBridge Camera Preview")
        window.configure(bg="#000000")
        window.attributes("-fullscreen", True)
        window.bind("<Escape>", self._exit_fullscreen)
        window.protocol("WM_DELETE_WINDOW", self._exit_fullscreen)
        canvas = tk.Canvas(window, bg="#000000", highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        canvas.create_text(
            0.5, 0.5, text="Camera preview will appear here\n\nPress Esc to exit",
            fill="#cbd5e1", font=("Segoe UI", 16), justify="center", anchor="center",
            tags="fullscreen_hint")
        canvas.bind("<Configure>", lambda event: canvas.coords(
            "fullscreen_hint", event.width // 2, event.height // 2))
        self._fullscreen_preview_window = window
        self._fullscreen_preview_canvas = canvas
        self._fullscreen_preview_image = None
        self._fullscreen_preview_image_id = None

    def _exit_fullscreen(self, _event=None):
        window = self._fullscreen_preview_window
        self._fullscreen_preview_window = None
        self._fullscreen_preview_canvas = None
        self._fullscreen_preview_image = None
        self._fullscreen_preview_image_id = None
        if window is not None:
            try:
                window.destroy()
            except tk.TclError:
                pass
        return "break"

    def _update_fullscreen_preview(self, image):
        canvas = self._fullscreen_preview_canvas
        window = self._fullscreen_preview_window
        if canvas is None or window is None:
            return
        try:
            if not window.winfo_exists():
                self._exit_fullscreen()
                return
            width, height = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
            scale = min(width / image.width, height / image.height)
            resized = image.resize((max(1, int(image.width * scale)),
                                    max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
            self._fullscreen_preview_image = ImageTk.PhotoImage(resized)
            if self._fullscreen_preview_image_id is None:
                self._fullscreen_preview_image_id = canvas.create_image(
                    width // 2, height // 2, image=self._fullscreen_preview_image)
            else:
                canvas.coords(self._fullscreen_preview_image_id, width // 2, height // 2)
                canvas.itemconfig(self._fullscreen_preview_image_id,
                                  image=self._fullscreen_preview_image)
            canvas.delete("fullscreen_hint")
        except tk.TclError:
            self._exit_fullscreen()

    def _open_camera_with_fallbacks(self, camera_index, preferred_backend=None):
        """Open a camera using Windows backends in a safe fallback order."""
        candidates = [preferred_backend, None]
        if os.name == "nt":
            candidates.extend([cv2.CAP_DSHOW, cv2.CAP_MSMF])

        attempted = set()
        for backend in candidates:
            # Backend constants are integers; None represents OpenCV's default.
            key = "default" if backend is None else int(backend)
            if key in attempted:
                continue
            attempted.add(key)
            try:
                capture = (cv2.VideoCapture(camera_index, backend)
                           if backend is not None else cv2.VideoCapture(camera_index))
                if capture.isOpened():
                    backend_name = "Default" if backend is None else str(backend)
                    APP_LOGGER.info("Camera %s opened with backend %s", camera_index, backend_name)
                    return capture
                capture.release()
            except Exception as exc:
                APP_LOGGER.debug("Camera %s backend %s failed: %s", camera_index, backend, exc)
        return None

    def start(self):
        if self.running:
            self.stop()
        try:
            self.cam_index = int(self.cam_var.get())
        except Exception:
            self.cam_index = 0
        self._save_preferences()

        backend = None
        if os.name == "nt":
            bsel = (self.backend_var.get() or "DirectShow").lower()
            if bsel == "directshow":
                backend = cv2.CAP_DSHOW
            elif bsel == "msmf":
                backend = cv2.CAP_MSMF

        self.cap = self._open_camera_with_fallbacks(self.cam_index, backend)

        try:
            rw, rh = (self.res_var.get() or "640x480").split("x")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(rw))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(rh))
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            if self.lowlat_var.get() and hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if self.mjpg_var.get():
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

        if not self.cap or not self.cap.isOpened():
            APP_LOGGER.error("Camera %s could not be opened", self.cam_index)
            messagebox.showerror("Camera not available",
                                 "SignBridge could not access the webcam. Please connect or "
                                 "enable a camera and try again.")
            return

        self.smoother.clear()
        self.last_seen_label = None
        self.streak = 0
        self.stable_since = 0.0
        self.last_none_ts = None
        self.rearm_ready = False
        self._word_scores.clear()
        self._word_candidate = None
        self._word_candidate_since = 0.0
        self._word_last_committed = None
        self._word_armed = True
        self.preview_image_id = None
        self.running = True
        self._session_started_at = time.time()
        self._session_saved_count = 0
        self._update_session_stats()
        self._set_recognition_controls(running=True)
        self.status_lbl.configure(text="● Camera: Connected", text_color="#16a34a")
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self._start_secondary_camera(backend)

    def reset_camera(self):
        """Release and reopen the selected camera using the current settings."""
        self.stop()
        self.root.after(150, self.start)

    def _start_secondary_camera(self, backend):
        """Optional second-person recognition in an independent OpenCV window."""
        if not self.dual_camera_var.get():
            return
        try:
            second_index = int(self.secondary_cam_var.get())
        except Exception:
            second_index = 1
        if second_index == self.cam_index:
            messagebox.showwarning("Camera 2", "Choose a different camera number for Camera 2.")
            return
        try:
            self.cap_secondary = self._open_camera_with_fallbacks(second_index, backend)
            rw, rh = (self.res_var.get() or "640x480").split("x")
            self.cap_secondary.set(cv2.CAP_PROP_FRAME_WIDTH, int(rw))
            self.cap_secondary.set(cv2.CAP_PROP_FRAME_HEIGHT, int(rh))
            if self.lowlat_var.get() and hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                self.cap_secondary.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            self.cap_secondary = None
        if not self.cap_secondary or not self.cap_secondary.isOpened():
            APP_LOGGER.warning("Camera 2 could not be opened")
            messagebox.showwarning("Camera 2 not available",
                                   "SignBridge could not open the second camera. The main camera is still running.")
            return
        self.secondary_thread = threading.Thread(target=self._secondary_loop, daemon=True)
        self.secondary_thread.start()

    def _secondary_loop(self):
        """Draw Camera 2 detections only; it never writes into Camera 1's transcript."""
        last_inference = 0.0
        label_text = "Waiting for sign"
        while self.running and self.cap_secondary and self.cap_secondary.isOpened():
            ok, frame = self.cap_secondary.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            now = time.time()
            if now - last_inference >= 0.08:  # cap Camera 2 at ~12 FPS
                last_inference = now
                try:
                    with self._model_lock:
                        model = self.active_model
                        classes = self.active_classes or []
                        mode = self.active_mode
                    if model is not None:
                        device = next(model.parameters()).device
                        inference_transforms = (self.word_transforms if mode == "words"
                                                else self.transforms)
                        transformed = inference_transforms(image=frame)
                        inp = torch.unsqueeze(transformed["image"], 0).to(device)
                        with torch.no_grad():
                            result = model(inp)
                        probs = result["pred_logits"].softmax(-1)[:, :, :-1]
                        max_probs, max_classes = probs.max(-1)
                        score, query = max_probs[0].max(0)
                        threshold = 0.55 if mode == "words" else 0.50
                        if float(score.detach().cpu()) >= threshold:
                            index = int(max_classes[0, query].detach().cpu())
                            label = classes[index] if index < len(classes) else "Unknown"
                            h, w = frame.shape[:2]
                            box = rescale_bboxes(result["pred_boxes"][0, query:query + 1], (w, h))[0]
                            x1, y1, x2, y2 = map(int, box.detach().cpu().numpy())
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (37, 99, 235), 3)
                            label_text = f"Detected: {label} ({float(score.detach().cpu()) * 100:.0f}%)"
                        else:
                            label_text = "Waiting for sign"
                except Exception:
                    APP_LOGGER.exception("Camera 2 inference failed")
                    label_text = "Recognition unavailable"
            cv2.putText(frame, f"Camera 2 | {label_text}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.imshow(self.secondary_window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            try:
                if cv2.getWindowProperty(self.secondary_window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

    def stop(self):
        self.running = False
        self._exit_fullscreen()
        self._set_recognition_controls(running=False)
        self.status_lbl.configure(text="● Camera: Stopped", text_color="#64748b")
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        if self.cap_secondary:
            try:
                self.cap_secondary.release()
            except Exception:
                pass
            self.cap_secondary = None
        try:
            cv2.destroyWindow(self.secondary_window_name)
        except Exception:
            pass
        if self.cv_preview.get():
            try:
                cv2.destroyWindow(self.cv_window_name)
            except Exception:
                pass
        if self.thread and self.thread.is_alive():
            try:
                self.thread.join(timeout=0.5)
            except Exception:
                pass
        if self.secondary_thread and self.secondary_thread.is_alive():
            try:
                self.secondary_thread.join(timeout=0.5)
            except Exception:
                pass
        try:
            self.preview.delete("all")
        except Exception:
            pass
        self.preview_image_id = None
        self.smoother.clear()
        # [NEW] reset confidence and status indicators to idle state on stop
        try:
            self.conf_lbl.configure(text="Confidence: —", text_color="#6b7280")
            self.status_dot_lbl.configure(text="● Waiting", text_color="#9ca3af")
        except Exception:
            pass

    def _update_preview(self, photo, cx, cy):
        if not self.running:
            return
        try:
            if self.preview_image_id is None:
                self.preview_image_id = self.preview.create_image(cx, cy, image=photo)
            else:
                self.preview.coords(self.preview_image_id, cx, cy)
                self.preview.itemconfig(self.preview_image_id, image=photo)
        except Exception:
            pass

    def _run_on_ui(self, fn, *args, **kwargs):
        try:
            self.root.after(0, lambda: fn(*args, **kwargs))
        except Exception:
            pass

    def _set_recognition_controls(self, running: bool):
        """Keep controls that need an active camera unavailable at idle."""
        state = "normal" if running else "disabled"
        for widget_name in ("stop_btn", "undo_btn", "finish_word_btn"):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.configure(state=state)

    def _set_last_saved(self, text: str):
        self._run_on_ui(self.last_saved_lbl.configure, text=f"Last saved: {text}")

    def _set_output_text(self, text: str):
        self._run_on_ui(self.output_text.set, text)

    def _set_detected_text(self, text: str):
        self._run_on_ui(self.detected_text.set, text)

    def _mode_label(self, mode: str) -> str:
        return "Words Mode" if mode == "words" else "Alphabet Mode"

    def _update_mode_banner(self, mode: str):
        if not self.mode_banner:
            return
        if mode == "words":
            new_bg = "#0f766e"
            self.mode_banner.configure(text="WORDS MODE", text_color="#ecfeff")
            if hasattr(self, "instruction_lbl"):
                self.instruction_lbl.configure(
                    text="Hold the same word for 2.5s • release/change to read another word")
        else:
            new_bg = "#1e2d45"
            self.mode_banner.configure(text="ALPHABET MODE", text_color="white")
            if hasattr(self, "instruction_lbl"):
                self.instruction_lbl.configure(
                    text="Hold a letter for 0.8s • briefly release between letters • pause 1.2s to finish")
        if hasattr(self, "_header_frame"):
            self._header_frame.configure(fg_color=new_bg)

    def _format_detected_text(self, label, _mode: str, accepted: bool = False) -> str:
        # _mode param kept for call-site compatibility; mode is now shown via mode_indicator_lbl
        if label:
            suffix = " ✓" if accepted else ""
            return f"Detected Sign: {label}{suffix}"
        return "Detected: —"

    # [NEW] ── Confidence, status, and mode indicator helpers ─────────────────
    # These are the only three methods needed to drive the new detect-row labels.
    # Each is called from _loop() via _run_on_ui so UI updates stay on the main thread.

    def _set_confidence(self, conf):
        """Update the confidence label. conf: float 0–1, or None when no detection."""
        def _update():
            if conf is None or conf < 0:
                self.conf_lbl.configure(text="Confidence: —", text_color="#6b7280")
            else:
                pct = int(conf * 100)
                # Color-code the score so low confidence is immediately obvious
                color = "#22c55e" if pct >= 70 else ("#f59e0b" if pct >= 50 else "#ef4444")
                self.conf_lbl.configure(text=f"Confidence: {pct}%", text_color=color)
        self._run_on_ui(_update)

    def _set_status(self, status: str):
        """Update the status dot with color coding. Called from _loop()."""
        _STATUS_MAP = {
            "Waiting":          ("#9ca3af", "● Waiting"),
            "Detecting":        ("#3b82f6", "● Detecting"),
            "Stable":           ("#22c55e", "● Stable"),
            "Low Confidence":   ("#f59e0b", "● Low Confidence"),
            "No Sign Detected": ("#ef4444", "● No Sign Detected"),
        }
        color, text = _STATUS_MAP.get(status, ("#9ca3af", f"● {status}"))
        self._run_on_ui(self.status_dot_lbl.configure, text=text, text_color=color)

    def _update_mode_indicator(self, mode: str):
        """Sync the mode badge in the detect row. Mirrors the header banner colors."""
        if mode == "words":
            label, bg, border, text = "Mode: Words",    "#f0fdfa", "#99f6e4", "#0f766e"
        else:
            label, bg, border, text = "Mode: Alphabet", "#eff6ff", "#bfdbfe", "#1d4ed8"
        try:
            self.mode_indicator_lbl.configure(text=label, text_color=text)
            self._mode_badge.configure(fg_color=bg, border_color=border)
        except Exception:
            pass

    # ── end new helpers ───────────────────────────────────────────────────────

    def _update_output_display(self):
        text = self.final_output
        if self.current_word:
            text = f"{text} {self.current_word}" if text else self.current_word
        self._set_output_text(text)

    def _on_space(self, event=None):
        self._finalize_word()
        return "break"

    def _on_enter(self, event=None):
        self.on_speak()
        return "break"

    def _on_backspace(self, event=None):
        if self.current_word:
            self.current_word = self.current_word[:-1]
            self._update_output_display()
        elif self.active_mode == "alphabet" and self.final_output.strip():
            # The normal pause may already have finalized the word. Reopen its
            # last token so Backspace still corrects the final letter instead
            # of forcing the user to clear the entire transcript.
            completed_words = self.final_output.strip().split()
            last_word = completed_words.pop()
            self.final_output = " ".join(completed_words)
            self.current_word = last_word[:-1]
            if self.history_chips:
                self.history_chips.pop().destroy()
            self._update_output_display()
        return "break"

    def _on_ctrl_backspace(self, event=None):
        self.current_word = ""
        self._update_output_display()
        return "break"

    def _on_clear_key(self, event=None):
        self.on_clear()
        return "break"

    def _on_mode_toggle(self):
        self.set_mode("alphabet" if self.active_mode == "words" else "words")

    def set_mode(self, mode: str):
        if mode not in ("alphabet", "words"):
            return
        if self._model_loading:
            return

        self._model_loading = True
        self.status_lbl.configure(text="Loading model...")
        self.mode_toggle_btn.configure(state="disabled")
        self._update_mode_banner(mode)

        def _load():
            try:
                if mode == "alphabet":
                    model, classes, colors = self.load_alphabet_model()
                else:
                    model, classes, colors = self.load_words_model()

                with self._model_lock:
                    self.active_mode = mode
                    self.active_model = model
                    self.active_classes = classes
                    self.active_colors = colors
                    self.model = model
                    self.CLASSES = classes
                    self.COLORS = colors

                # Word sensitivity is not used by Alphabet Mode; hiding it
                # keeps the main controls focused and uncluttered.
                if mode == "words":
                    self.sensitivity_row.grid()
                    self.hold_progress_label.configure(text="Word hold progress (2.5s)")
                else:
                    self.sensitivity_row.grid_remove()
                    self.hold_progress_label.configure(text="Letter hold progress (0.8s)")
                self.hold_progress.set(0)

                self.smoother.clear()
                self.current_word = ""
                self.last_commit_letter = None
                self.last_commit_ts = 0.0
                self.last_accept_ts = None
                self.last_seen_label = None
                self.streak = 0
                self.stable_since = 0.0
                self.last_none_ts = None
                self.rearm_ready = False
                self._word_scores.clear()
                self._word_commit_ts.clear()
                self._word_candidate = None
                self._word_candidate_since = 0.0
                self._word_last_committed = None
                self._word_armed = True
                self._update_output_display()
                self._set_detected_text(self._format_detected_text(None, mode))
                self._run_on_ui(self._save_preferences)
            except FileNotFoundError as exc:
                APP_LOGGER.exception("Required model is missing")
                title = "Alphabet model missing" if mode == "alphabet" else "Words model missing"
                self._run_on_ui(messagebox.showerror, title,
                                "The required model file could not be found.\n\n"
                                f"Expected: {exc}\n\nSee {LOG_FILE} for details.")
            except Exception:
                APP_LOGGER.error("Model loading failed:\n%s", traceback.format_exc())
                self._run_on_ui(messagebox.showerror, "Model loading failed",
                                "SignBridge could not load this model. Please verify the "
                                "distribution is complete.\n\nTechnical details were saved to:\n"
                                f"{LOG_FILE}")
            finally:
                self._model_loading = False
                label = "Mode: Words" if mode == "words" else "Mode: Alphabet"
                self._run_on_ui(self.mode_var.set, mode == "words")
                self._run_on_ui(self.mode_toggle_btn.configure, text=label, state="normal")
                self._run_on_ui(self._update_mode_banner, mode)
                self._run_on_ui(self._update_mode_indicator, mode)  # [NEW] sync detect-row mode label
                if self.running:
                    self._run_on_ui(self.status_lbl.configure,
                                    text="● Camera: Connected", text_color="#16a34a")
                else:
                    self._run_on_ui(self.status_lbl.configure,
                                    text="● Camera: Ready", text_color="#64748b")

        threading.Thread(target=_load, daemon=True).start()

    def _loop(self):
        while self.running and self.cap and self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            with self._model_lock:
                model = self.active_model
                classes_ref = self.active_classes or []
                colors_ref = self.active_colors or []
                mode = self.active_mode

            if model is None:
                time.sleep(0.01)
                continue

            # Inference
            stable_label = None
            try:
                device = next(model.parameters()).device
                inference_transforms = (self.word_transforms if mode == "words"
                                        else self.transforms)
                transformed = inference_transforms(image=frame)
                inp = torch.unsqueeze(transformed['image'], dim=0).to(device)
                with torch.no_grad():
                    result = model(inp)
                probs = result['pred_logits'].softmax(-1)[:, :, :-1]
                max_probs, max_classes = probs.max(-1)
                # Use a lower confidence floor for bounding-box drawing in
                # words mode so the user sees the box even at medium confidence.
                draw_thresh = 0.55 if mode == "words" else 0.5
                keep = max_probs > draw_thresh
                bi, qi = torch.where(keep)
                if len(qi) > 0:
                    top = max_probs[bi, qi].argmax()
                    bi, qi = bi[top:top+1], qi[top:top+1]
                h, w = frame.shape[:2]
                bboxes = rescale_bboxes(result['pred_boxes'][bi, qi, :], (w, h))
                classes = max_classes[bi, qi]
                probas = max_probs[bi, qi]

                top_raw_conf = None  # [NEW] raw model confidence for the top detection this frame
                for bc, bp, bb in zip(classes, probas, bboxes):
                    idx = int(bc.detach().cpu().numpy())
                    color = colors_ref[idx] if idx < len(colors_ref) else (0, 255, 0)
                    x1, y1, x2, y2 = map(int, bb.detach().cpu().numpy())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    label = classes_ref[idx] if idx < len(classes_ref) else ""
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
                    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
                    cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    top_raw_conf = float(bp.detach().cpu().numpy())  # [NEW] capture once per top detection
                    if mode == "alphabet":
                        stable_label, _ = self.smoother.update(label, top_raw_conf)

                now_ts = time.time()

                # ── Alphabet mode: streak-based commit ───────────────────────
                if mode == "alphabet":
                    valid_label = stable_label if (stable_label and len(stable_label) == 1 and stable_label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ") else None

                    if valid_label is None:
                        self._run_on_ui(self.hold_progress.configure, value=0)
                        if self.last_none_ts is None:
                            self.last_none_ts = now_ts
                        if (now_ts - self.last_none_ts) >= self.no_det_gap:
                            self.rearm_ready = True
                            self.last_seen_label = None
                            self.streak = 0
                            self.stable_since = now_ts
                        self._set_detected_text(self._format_detected_text(None, mode))
                        self._set_confidence(None)       # [NEW] no detection → blank confidence
                        self._set_status("Waiting")      # [NEW] nothing visible → Waiting
                    else:
                        if self.last_commit_letter and valid_label != self.last_commit_letter:
                            self.rearm_ready = True
                        if valid_label == self.last_seen_label:
                            self.streak += 1
                        else:
                            self.last_seen_label = valid_label
                            self.streak = 1
                            self.stable_since = now_ts
                            self.last_none_ts = None
                        stable_ok = (now_ts - self.stable_since) >= self.alphabet_hold_seconds
                        hold_pct = int(min(100, (now_ts - self.stable_since) /
                                           self.alphabet_hold_seconds * 100))
                        self._run_on_ui(self.hold_progress.configure, value=hold_pct)
                        self._set_detected_text(self._format_detected_text(valid_label, mode, accepted=stable_ok))
                        self._set_confidence(top_raw_conf)                               # [NEW] live confidence score
                        self._set_status("Stable" if stable_ok else "Detecting")        # [NEW] streak done or building
                        if stable_ok:
                            self.last_accept_ts = now_ts
                            can_commit = False
                            if valid_label != self.last_commit_letter:
                                can_commit = True
                            elif self.rearm_ready and (now_ts - self.last_commit_ts) >= self.repeat_gap:
                                can_commit = True
                            if can_commit:
                                self.current_word += valid_label
                                self.last_commit_letter = valid_label
                                self.last_commit_ts = now_ts
                                self.rearm_ready = False
                                self._update_output_display()
                                self._set_last_saved(f"Letter {valid_label}")
                                self._record_saved_sign()

                    if self.current_word and self.last_accept_ts and (now_ts - self.last_accept_ts) >= self.idle_timeout:
                        self._finalize_word()
                        self.last_accept_ts = None

                # ── Words mode: decaying confidence accumulator ───────────────
                else:
                    # Take the best prediction across ALL queries (not just > draw_thresh)
                    # so we accumulate evidence even at lower confidence.
                    raw_val, raw_qi = max_probs[0].max(0)
                    raw_conf = float(raw_val.detach())
                    raw_idx = int(max_classes[0, raw_qi])
                    raw_label = (
                        classes_ref[raw_idx]
                        if raw_conf >= self._word_min_conf and raw_idx < len(classes_ref)
                        else None
                    )

                    # The same word must stay visible for the full hold time.
                    # A different prediction or no usable prediction restarts
                    # the timer, so partial/accidental poses are never saved.
                    if raw_label != self._word_candidate:
                        self._word_candidate = raw_label
                        self._word_candidate_since = now_ts if raw_label else 0.0
                    if raw_label is None:
                        self._word_armed = True
                    elif raw_label != self._word_last_committed:
                        self._word_armed = True
                    hold_elapsed = (now_ts - self._word_candidate_since) if raw_label else 0.0

                    # Decay all buckets each frame so stale evidence fades away.
                    for k in list(self._word_scores.keys()):
                        self._word_scores[k] *= self._word_decay
                        if self._word_scores[k] < 0.05:
                            del self._word_scores[k]

                    # Accumulate evidence from each eligible frame. This keeps
                    # Words Mode responsive; its score decay and commit
                    # threshold still prevent one-frame predictions committing.
                    if raw_label:
                        self._word_scores[raw_label] = self._word_scores.get(raw_label, 0.0) + raw_conf

                    # Find the leading candidate.
                    if self._word_scores:
                        best_word = max(self._word_scores, key=lambda k: self._word_scores[k])
                        best_score = self._word_scores[best_word]
                    else:
                        best_word, best_score = None, 0.0

                    # Show live feedback with a progress percentage.
                    if raw_label:
                        pct = int(min(100, hold_elapsed / self.word_hold_seconds * 100))
                        # [NEW] detected label simplified \u2014 confidence and status shown in detect row
                        self._set_detected_text(f"Detected: {raw_label}")
                        self._set_confidence(raw_conf)   # [NEW] raw per-frame model confidence
                        self._set_status("Stable" if hold_elapsed >= self.word_hold_seconds else "Detecting")
                        self._run_on_ui(self.hold_progress.configure, value=pct)
                    else:
                        self._set_detected_text("Detected: \u2014")   # [NEW] simplified (no mode suffix)
                        self._set_confidence(None)                     # [NEW] no usable detection
                        # [NEW] distinguish a weak visible sign from total absence
                        if top_raw_conf is not None and raw_conf < self._word_min_conf:
                            self._set_status("Low Confidence")
                        else:
                            self._set_status("Waiting")
                        self._run_on_ui(self.hold_progress.configure, value=0)

                    # Commit only once per continuous hold. To enter the same
                    # word again, briefly remove the sign and hold it again.
                    if (raw_label and self._word_armed and
                            hold_elapsed >= self.word_hold_seconds):
                        self._word_commit_ts[raw_label] = now_ts
                        self._word_scores[raw_label] = 0.0
                        self._word_last_committed = raw_label
                        self._word_armed = False
                        self._record_saved_sign()
                        self._append_output(raw_label)
                        self._run_on_ui(self._append_history, raw_label)
            except Exception as e:
                APP_LOGGER.error("Inference error in %s mode:\n%s", mode, traceback.format_exc())
                # Keep technical details out of the recognition panel. The
                # status remains useful to a normal user and details go to log.
                self._set_detected_text("Detected: —")
                self._set_confidence(None)
                self._set_status("Recognition paused — check log")
                time.sleep(0.1)

            # Render
            now = time.time()
            try:
                if self.cv_preview.get():
                    cv2.imshow(self.cv_window_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    try:
                        if cv2.getWindowProperty(self.cv_window_name, cv2.WND_PROP_VISIBLE) < 1:
                            break
                    except cv2.error:
                        break
                else:
                    if (now - self._last_render_ts) >= self.render_interval:
                        # FIX: Always use current canvas dimensions for proper preview sizing
                        # This ensures the preview scales to fill available space when window resizes
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(rgb)
                        if self._fullscreen_preview_window is not None:
                            self.root.after(0, self._update_fullscreen_preview, img.copy())
                        # Use preview_target_w/h (set by Configure event) for rendering
                        tw, th = self.preview_target_w, self.preview_target_h
                        fh, fw = frame.shape[:2]
                        # Scale frame to fit in target area while preserving aspect ratio
                        scale = min(tw / fw, th / fh)
                        nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
                        resample = getattr(Image, "Resampling", Image).LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
                        img = img.resize((nw, nh), resample)
                        self.photo = ImageTk.PhotoImage(img)
                        self._photo_cache.append(self.photo)
                        if len(self._photo_cache) > 3:
                            self._photo_cache.pop(0)
                        # Center image in canvas — gets current canvas center coordinates
                        cx = max(1, self.preview.winfo_width() // 2)
                        cy = max(1, self.preview.winfo_height() // 2)
                        self.root.after(0, self._update_preview, self.photo, cx, cy)
                        self._last_render_ts = now

                self._fps_frames += 1
                if (now - self._fps_ts) >= 1.0:
                    self._fps_value = self._fps_frames / (now - self._fps_ts)
                    self._fps_ts = now
                    self._fps_frames = 0
                    if self.eval_mode.get():
                        self.status_lbl.configure(text=f"FPS: {self._fps_value:.1f}")
            except Exception:
                pass

            time.sleep(0.005)

        # A read failure means the camera was unplugged or stopped responding.
        # Normal Stop already sets running to False, so only notify on an
        # unexpected disconnect.
        if self.running:
            self.running = False
            try:
                self.cap.release()
            except Exception:
                pass
            self._run_on_ui(self._set_recognition_controls, running=False)
            self._run_on_ui(self.status_lbl.configure,
                            text="● Camera disconnected — use Reset Camera",
                            text_color="#dc2626")
            self._set_status("Camera disconnected")

    def _append_output(self, token):
        txt = self.final_output.strip()
        self.final_output = f"{txt} {token}" if txt else token
        self._update_output_display()
        self._set_last_saved(f"Word {token}")
        # Auto-speak: fire TTS after a pause when auto-speak is enabled
        self._last_auto_speak_ts = time.time()
        if self.auto_speak.get():
            def _delayed_speak(expected_ts):
                if self._last_auto_speak_ts == expected_ts:
                    self.tts.speak(self.final_output)
            ts = self._last_auto_speak_ts
            self.root.after(int(self._auto_speak_idle * 1000), _delayed_speak, ts)

    def _finalize_word(self):
        if not self.current_word:
            return
        word = self.current_word
        self.current_word = ""
        self._append_output(word)
        self._run_on_ui(self._append_history, word)

    def _append_history(self, token):
        chip = ttk.Label(self.history_inner, text=token, relief="groove", padding=(8, 4))
        chip.pack(side="left", padx=4)
        self.history_chips.append(chip)
        while len(self.history_chips) > MAX_HISTORY_CHIPS:
            self.history_chips.pop(0).destroy()
        self.history_canvas.xview_moveto(1.0)

    def on_clear(self):
        if (self.final_output.strip() or self.current_word) and not messagebox.askyesno(
                "Clear recognized text", "Clear all recognized text and sign history?"):
            return
        self.final_output = ""
        self.current_word = ""
        self.last_commit_letter = None
        self.last_commit_ts = 0.0
        self.last_accept_ts = None
        self.last_seen_label = None
        self.streak = 0
        self.stable_since = 0.0
        self.last_none_ts = None
        self.rearm_ready = False
        self._word_scores.clear()
        self._word_commit_ts.clear()
        self._word_candidate = None
        self._word_candidate_since = 0.0
        self._word_last_committed = None
        self._word_armed = True
        self._set_last_saved("—")
        self.hold_progress.set(0)
        self.output_text.set("")
        self.detected_text.set(self._format_detected_text(None, self.active_mode))
        # [NEW] reset confidence and status indicators when the user clears output
        try:
            self.conf_lbl.configure(text="Confidence: —", text_color="#6b7280")
            self.status_dot_lbl.configure(text="● Waiting", text_color="#9ca3af")
        except Exception:
            pass
        for chip in self.history_chips:
            chip.destroy()
        self.history_chips.clear()
        self.history_canvas.delete("all")
        self.history_inner = ttk.Frame(self.history_canvas)
        self.history_canvas.create_window((0, 0), window=self.history_inner, anchor="nw")
        self.history_inner.bind("<Configure>", lambda e: self.history_canvas.configure(scrollregion=self.history_canvas.bbox("all")))
        self.smoother.clear()

    def on_save(self):
        import datetime
        text = self.final_output.strip()
        if not text:
            messagebox.showinfo("Save", "Nothing to save yet.")
            return
        saved_at = datetime.datetime.now()
        ts = saved_at.strftime("%Y%m%d_%H%M%S")
        save_path = writable_path(f"transcript_{ts}.txt")
        try:
            transcript = f"SignBridge Transcript\nSaved: {saved_at:%Y-%m-%d %H:%M:%S}\n\n{text}\n"
            save_path.write_text(transcript, encoding="utf-8")
            messagebox.showinfo("Saved", f"Transcript saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Save", f"Failed to save: {e}")

    def copy_text(self):
        text = self.output_text.get().strip()
        if not text:
            messagebox.showinfo("Copy", "Nothing to copy yet.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()
        self.status_lbl.configure(text="Text copied to clipboard", text_color="#16a34a")

    def show_about(self):
        device_name = (torch.cuda.get_device_name(0) if self.device.type == "cuda"
                       else "CPU mode")
        messagebox.showinfo(
            "About SignBridge",
            "SignBridge\n\n"
            "Real-time sign recognition and text-to-sign playback.\n"
            "Alphabet Mode: A–Z\n"
            "Words Mode: 8 supported signs\n\n"
            f"Active device: {device_name}\n"
            "Version 1.0"
        )

    def on_speak(self):
        try:
            text = self.current_word if self.current_word else self.final_output
            self.tts.speak(text)
        except Exception as e:
            messagebox.showerror("Speak", f"TTS failed: {e}")

    def _on_speech_rate_change(self, value):
        rate = int(float(value))
        self._speech_rate_var.set(rate)
        self.tts.set_rate(rate)
        self.speech_rate_lbl.configure(text=f"{rate} WPM")

    def _on_volume_change(self, value):
        volume = int(float(value))
        self._speech_volume_var.set(volume)
        self.tts.set_volume(volume / 100)
        self.volume_lbl.configure(text=f"{volume}%")

    def _on_voice_selected(self, voice_name):
        if voice_name != "Default voice":
            self.tts.set_voice(voice_name)
        self._save_preferences()

    def _refresh_voice_options(self, attempt=0):
        voices = self.tts.voice_names()
        if voices:
            self.voice_selector.configure(values=["Default voice", *voices])
        elif attempt < 8:
            self.root.after(500, self._refresh_voice_options, attempt + 1)

    def on_voice(self):
        text = self._typed_input_text()
        if not text:
            return
        try:
            self.tts.speak(text)
        except Exception as e:
            messagebox.showerror("Voice", f"TTS failed: {e}")
        seq = text_to_sequence(text)
        if seq:
            self._show_signs_window(seq)

    def on_show_signs(self):
        text = self._typed_input_text()
        seq = text_to_sequence(text)
        if seq:
            self._show_signs_window(seq)

    def _clear_text_placeholder(self, _event=None):
        if self._input_placeholder_active:
            self.input_text.set("")
            self._input_placeholder_active = False
            self.text_entry.configure(text_color="#1e293b")

    def _restore_text_placeholder(self, _event=None):
        if not self.input_text.get().strip():
            self.input_text.set(self._input_placeholder)
            self._input_placeholder_active = True
            self.text_entry.configure(text_color="#94a3b8")

    def _typed_input_text(self):
        if self._input_placeholder_active:
            return ""
        return self.input_text.get().strip()

    def show_supported_signs(self):
        """Open the supported-sign reference window without interrupting recognition."""
        if self.supported_signs_win is not None:
            try:
                if self.supported_signs_win.winfo_exists():
                    self.supported_signs_win.deiconify()
                    self.supported_signs_win.lift()
                    self.supported_signs_win.focus_force()
                    return
            except Exception:
                pass
        win = ctk.CTkToplevel(self.root)
        self.supported_signs_win = win
        win.title("Supported Signs")
        win.configure(fg_color="#f5f7fb")
        win.resizable(False, False)
        win.geometry("920x620")
        try:
            if ASSETS_ICON.exists():
                win.iconbitmap(default=str(ASSETS_ICON))
        except Exception:
            pass

        def close():
            self.supported_signs_win = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", close)
        win._image_refs = []
        self.root.update_idletasks()
        x = self.root.winfo_rootx() + max(0, (self.root.winfo_width() - 920) // 2)
        y = self.root.winfo_rooty() + max(0, (self.root.winfo_height() - 620) // 2)
        win.geometry(f"920x620+{x}+{y}")

        header = ctk.CTkFrame(win, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(16, 8))
        ctk.CTkFrame(header, fg_color=self._accent, width=4, height=22, corner_radius=2).pack(
            side="left", padx=(0, 9))
        ctk.CTkLabel(header, text="Supported Signs", font=ctk.CTkFont(size=20, weight="bold"),
                     text_color="#13233a").pack(side="left")
        ctk.CTkLabel(win, text="Reference guide for gestures currently recognized by SignBridge",
                     font=ctk.CTkFont(size=11), text_color="#64748b").pack(
                         anchor="w", padx=33, pady=(0, 10))

        content = ctk.CTkFrame(win, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=0)
        content.grid_rowconfigure(0, weight=1)
        tabs = ctk.CTkTabview(content, fg_color="#edf1f7",
                              segmented_button_fg_color="#e3e9f2",
                              segmented_button_selected_color=self._accent,
                              segmented_button_selected_hover_color="#1d4ed8",
                              text_color="#172b4d")
        tabs.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        alphabet_tab = tabs.add("Alphabet Mode (A-Z)")
        words_tab = tabs.add(f"Words Mode ({len(WORD_CLASSES)} Signs)")

        notes = ctk.CTkFrame(content, width=230, fg_color="#edf5ff", corner_radius=10,
                              border_width=1, border_color="#bfdbfe")
        notes.grid(row=0, column=1, sticky="ns")
        notes.grid_propagate(False)
        ctk.CTkLabel(notes, text="Important Notes", font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="#1d4ed8").pack(anchor="w", padx=14, pady=(16, 10))
        for note in (
            "SignBridge currently recognizes supported static hand gestures only.",
            "Keep the hand clearly visible inside the camera view.",
            "Recognition works best with sufficient lighting.",
            "Avoid visually cluttered backgrounds when possible.",
            "Use proper hand placement and camera distance.",
        ):
            ctk.CTkLabel(notes, text=f"✓  {note}", justify="left", anchor="w", wraplength=172,
                         font=ctk.CTkFont(size=10), text_color="#334155").pack(
                             anchor="w", padx=14, pady=(0, 12))

        self._build_supported_tab(alphabet_tab, "Alphabet Mode (A-Z)",
            "Use this mode for fingerspelling letters from A to Z.",
            [(letter, letter) for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"], "alphabet", 6, win)
        self._build_supported_tab(words_tab, f"Words Mode ({len(WORD_CLASSES)} Signs)",
            "Use this mode for these supported word-level signs.",
            [(label.replace("_", " "), label) for label in WORD_CLASSES],
            "words", 5, win)
        ctk.CTkButton(win, text="Close", command=close, width=120, height=36,
                       fg_color=self._accent, hover_color="#1d4ed8", text_color="white",
                       corner_radius=8,
                       font=ctk.CTkFont(size=11, weight="bold")).pack(pady=(0, 16))

    def _build_supported_tab(self, tab, heading, description, signs, kind, columns, win):
        ctk.CTkLabel(tab, text=heading, font=ctk.CTkFont(size=15, weight="bold"),
                     text_color="#13233a").pack(anchor="w", padx=14, pady=(14, 1))
        ctk.CTkLabel(tab, text=description, font=ctk.CTkFont(size=10),
                     text_color="#64748b").pack(anchor="w", padx=14, pady=(0, 10))
        # Keep every supported sign reachable on smaller displays without
        # stretching the reference window beyond the main application.
        grid = ctk.CTkScrollableFrame(tab, fg_color="transparent", corner_radius=0)
        grid.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        for col in range(columns):
            grid.grid_columnconfigure(col, weight=1)
        for index, (label, asset_name) in enumerate(signs):
            card = ctk.CTkFrame(grid, fg_color="#ffffff", corner_radius=8,
                                border_width=1, border_color="#e2e8f0")
            card.grid(row=index // columns, column=index % columns, padx=4, pady=4, sticky="nsew")
            image = self._load_supported_sign_image(kind, asset_name, size=(72, 72))
            if image is not None:
                win._image_refs.append(image)
                ctk.CTkLabel(card, text="", image=image).pack(pady=(10, 3))
            else:
                ctk.CTkLabel(card, text="No image", width=72, height=72, corner_radius=6,
                             fg_color="#f1f5f9", text_color="#94a3b8",
                             font=ctk.CTkFont(size=9)).pack(pady=(10, 3))
            ctk.CTkLabel(card, text=label, wraplength=82, justify="center",
                         font=ctk.CTkFont(size=10, weight="bold"), text_color="#1e293b").pack(
                             padx=4, pady=(2, 10))

    def _load_supported_sign_image(self, kind, token, size):
        """Load a gallery reference safely; missing user images show a placeholder."""
        directory = SIGNS_DIR / kind
        variants = {token, token.lower(), token.upper(), token.capitalize(),
                    token.replace(" ", "_"), token.replace(" ", "")}
        for name in variants:
            for extension in ("png", "jpg", "jpeg", "webp", "gif"):
                path = directory / f"{name}.{extension}"
                if path.is_file():
                    try:
                        image = Image.open(path).convert("RGB")
                        return ctk.CTkImage(light_image=image, dark_image=image, size=size)
                    except Exception:
                        APP_LOGGER.warning("Could not load sign reference image: %s", path)
                        return None
        return None

    def _load_sign_image(self, kind: str, token: str, size: int = 320):
        """Return an ImageTk.PhotoImage for the given sign, or None if not found."""
        sub = "words" if kind == "WORD" else "alphabet"
        exts = ("png", "jpg", "jpeg", "gif", "webp")
        candidates = [token, token.upper(), token.lower(), token.capitalize()]
        search_dirs = [SIGNS_DIR / sub, SIGNS_DIR]
        for d in search_dirs:
            for name in candidates:
                for ext in exts:
                    p = d / f"{name}.{ext}"
                    if p.exists():
                        try:
                            pil = Image.open(p).convert("RGB")
                            resample = getattr(Image, "Resampling", Image).LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
                            pil = pil.resize((size, size), resample)
                            return ImageTk.PhotoImage(pil)
                        except Exception:
                            return None
        return None

    def _show_signs_window(self, seq):
        """Open a sign viewer window that cycles through sign images for the sequence."""
        WIN_SIZE = 320
        win = tk.Toplevel(self.root)
        win.title("Sign Viewer")
        win.resizable(False, False)
        try:
            if ASSETS_ICON.exists():
                win.iconbitmap(default=str(ASSETS_ICON))
        except Exception:
            pass

        # Pre-load images (None means no image available)
        entries = []
        for kind, token in seq:
            img = self._load_sign_image(kind, token, WIN_SIZE)
            entries.append((kind, token, img))

        idx_var = tk.IntVar(value=0)
        photo_ref = [None]  # keep a strong reference to the current PhotoImage

        # ── Image canvas ────────────────────────────────────────────────────
        canvas = tk.Canvas(win, width=WIN_SIZE, height=WIN_SIZE, bg="#111111",
                           highlightthickness=0)
        canvas.pack(padx=12, pady=(12, 4))
        img_item = canvas.create_image(WIN_SIZE // 2, WIN_SIZE // 2, anchor="center")

        # ── Labels ──────────────────────────────────────────────────────────
        token_lbl = ttk.Label(win, text="", font=("Segoe UI", 20, "bold"))
        token_lbl.pack()
        counter_lbl = ttk.Label(win, text="", foreground="gray")
        counter_lbl.pack()

        def show(i):
            i = max(0, min(len(entries) - 1, i))
            idx_var.set(i)
            kind, token, img = entries[i]
            canvas.delete("placeholder")
            if img is not None:
                photo_ref[0] = img
                canvas.itemconfig(img_item, image=img)
            else:
                photo_ref[0] = None
                canvas.itemconfig(img_item, image="")
                canvas.create_text(
                    WIN_SIZE // 2, WIN_SIZE // 2,
                    text=token,
                    font=("Segoe UI", 64, "bold"),
                    fill="white",
                    tags="placeholder",
                )
            token_lbl.config(text=token)
            counter_lbl.config(text=f"{i + 1} / {len(entries)}")
            prev_btn.config(state="normal" if i > 0 else "disabled")
            next_btn.config(state="normal" if i < len(entries) - 1 else "disabled")

        # ── Navigation ──────────────────────────────────────────────────────
        nav = ttk.Frame(win)
        nav.pack(pady=8)
        prev_btn = ttk.Button(nav, text="◀ Prev",
                              command=lambda: show(idx_var.get() - 1))
        prev_btn.pack(side="left", padx=8)
        next_btn = ttk.Button(nav, text="Next ▶",
                              command=lambda: show(idx_var.get() + 1))
        next_btn.pack(side="left", padx=8)

        # Keyboard navigation
        win.bind("<Left>", lambda e: show(idx_var.get() - 1))
        win.bind("<Right>", lambda e: show(idx_var.get() + 1))
        win.bind("<Escape>", lambda e: win.destroy())

        show(0)

    def _python_for_mode(self):
        return VENV_PYTHONW if HIDE_CONSOLE and VENV_PYTHONW.exists() else VENV_PYTHON

    def _run_command(self, args):
        try:
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" and HIDE_CONSOLE else 0
            subprocess.Popen(args, cwd=PROJECT_ROOT, creationflags=flags)
        except Exception as e:
            messagebox.showerror("Command", f"Failed: {e}")

    def _add_command_buttons(self, parent):
        py = self._python_for_mode()
        row = ttk.Frame(parent)
        row.pack(fill="x")
        ttk.Button(row, text="📥 Collect", command=lambda: self._run_command([str(py), str(PROJECT_ROOT / 'src' / 'utils' / 'collect_images.py')])).pack(side="left", padx=2, pady=2)
        ttk.Button(row, text="🛠️ Train", command=lambda: self._run_command([str(py), str(PROJECT_ROOT / 'src' / 'train.py')])).pack(side="left", padx=2, pady=2)
        ttk.Button(row, text="🧪 Test", command=lambda: self._run_command([str(py), str(PROJECT_ROOT / 'src' / 'test.py')])).pack(side="left", padx=2, pady=2)
        ttk.Button(row, text="⚡ Realtime", command=lambda: self._run_command([str(py), str(PROJECT_ROOT / 'src' / 'realtime.py')])).pack(side="left", padx=2, pady=2)
        ttk.Button(row, text="🗂 Data", command=lambda: self._run_command([str(py), str(PROJECT_ROOT / 'src' / 'data.py')])).pack(side="left", padx=2, pady=2)
        lblstudio = PROJECT_ROOT / '.venv' / 'Scripts' / 'label-studio.exe'
        ttk.Button(row, text="🏷️ Label Studio", command=lambda: self._run_command([str(lblstudio)])).pack(side="left", padx=2, pady=2)

    def _load_classes_from_meta(self, ckpt_rel, fallback):
        ckpt_path = resource_path(ckpt_rel)
        meta_path = ckpt_path.with_name("meta.json")
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                classes = meta.get("classes")
                if isinstance(classes, list) and classes:
                    return classes
            except Exception:
                pass
        return fallback

    def load_alphabet_model(self):
        if self._alphabet_model is not None and self._alphabet_classes is not None and self._alphabet_colors is not None:
            return self._alphabet_model, self._alphabet_classes, self._alphabet_colors

        fallback_classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        classes = self._load_classes_from_meta(ALPHABET_CKPT, fallback_classes)
        ckpt_path = resource_path(ALPHABET_CKPT)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Alphabet model missing: {ckpt_path}")
        model = DETR(num_classes=len(classes), pretrained_backbone=False, verbose=False)
        model.eval()
        try:
            model.load_pretrained(str(ckpt_path), device=self.device)
        except Exception as e:
            APP_LOGGER.exception("Alphabet model loading failed")
            raise RuntimeError("The Alphabet Mode model could not be loaded.") from e

        colors = self._resolve_colors(classes)
        self._alphabet_model = model
        self._alphabet_classes = classes
        self._alphabet_colors = colors
        return model, classes, colors

    def load_words_model(self):
        if self._words_model is not None and self._words_classes is not None and self._words_colors is not None:
            return self._words_model, self._words_classes, self._words_colors

        classes = self._load_classes_from_meta(WORDS_CKPT, WORD_CLASSES)
        ckpt_path = resource_path(WORDS_CKPT)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Words model missing: {ckpt_path}")
        # Older checkpoints used a one-layer box head. New Words Mode training
        # stores its three-layer head in metadata, so both formats remain
        # loadable while Alphabet Mode remains unchanged.
        box_head_layers = 1
        words_image_size = 224
        meta_path = ckpt_path.with_name("meta.json")
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            box_head_layers = int(metadata.get("box_head_layers", 1))
            words_image_size = int(metadata.get("words_image_size", 224))
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        model = DETR(num_classes=len(classes), pretrained_backbone=False,
                     verbose=False, box_head_layers=box_head_layers)
        model.eval()
        try:
            model.load_pretrained(str(ckpt_path), device=self.device)
        except Exception as e:
            APP_LOGGER.exception("Words model loading failed")
            raise RuntimeError("The Words Mode model could not be loaded.") from e

        colors = self._resolve_colors(classes)
        self._words_image_size = words_image_size
        self.word_transforms = A.Compose([
            A.Resize(words_image_size, words_image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A_ToTensorV2(),
        ])
        self._words_model = model
        self._words_classes = classes
        self._words_colors = colors
        return model, classes, colors

    def _resolve_colors(self, classes):
        colors = get_colors()
        if isinstance(colors, list) and len(colors) >= len(classes):
            return colors[:len(classes)]

        palette = []
        for i in range(len(classes)):
            r = int((37 * i) % 255)
            g = int((83 * i) % 255)
            b = int((149 * i) % 255)
            palette.append((r, g, b))
        return palette


def main():
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    APP_LOGGER.info("Using device: %s", device.type.upper())
    if device.type == "cuda":
        APP_LOGGER.info("CUDA device: %s", torch.cuda.get_device_name(0))
    SignBridgeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
