import threading
import queue
import time
import re


# Model class labels may omit spaces for training compatibility. These rules are
# used only for speech, so recognition labels and saved output are unchanged.
PRONUNCIATION_OVERRIDES = {
    "iloveyou": "I love you",
    "thankyou": "Thank you",
    "my name is": "My name is",
}


def format_for_speech(text: str) -> str:
    """Convert compact sign labels to natural phrases before pyttsx3 speaks."""
    formatted = text.replace("_", " ")
    for compact, spoken in PRONUNCIATION_OVERRIDES.items():
        pattern = rf"\b{re.escape(compact)}\b"
        formatted = re.sub(pattern, spoken, formatted, flags=re.IGNORECASE)
    return " ".join(formatted.split())

class TTSWorker:
    """
    Background TTS queue using pyttsx3 non-blocking loop to avoid
    'speak can only be used once at a time' errors when multiple
    speak requests are queued quickly.
    """
    def __init__(self, rate: int = 160, volume: float = 0.9):
        self.q: queue.Queue[str] = queue.Queue()
        self.thread: threading.Thread | None = None
        self.running = False
        self._engine = None
        self._rate = int(rate)
        self._volume = float(volume)
        self._voice_id = None
        self._voices: dict[str, str] = {}
        self._rate_lock = threading.Lock()
        self._settings_dirty = True

    def set_rate(self, rate: int):
        """Set words-per-minute; applied by the TTS thread before speaking."""
        with self._rate_lock:
            self._rate = max(80, min(300, int(rate)))
            self._settings_dirty = True

    def set_volume(self, volume: float):
        with self._rate_lock:
            self._volume = max(0.0, min(1.0, float(volume)))
            self._settings_dirty = True

    def set_voice(self, voice_name: str):
        with self._rate_lock:
            self._voice_id = self._voices.get(voice_name)
            self._settings_dirty = True

    def voice_names(self) -> list[str]:
        with self._rate_lock:
            return list(self._voices.keys())

    def _apply_settings(self):
        """Called only from the worker thread because SAPI is not thread-safe."""
        with self._rate_lock:
            if not self._settings_dirty:
                return
            self._engine.setProperty("rate", self._rate)
            self._engine.setProperty("volume", self._volume)
            if self._voice_id:
                self._engine.setProperty("voice", self._voice_id)
            self._settings_dirty = False

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        # signal loop to exit
        self.running = False
        try:
            self.q.put_nowait("")
        except Exception:
            pass
        # attempt to stop engine loop
        try:
            if self._engine:
                self._engine.stop()
                # endLoop if started
                try:
                    self._engine.endLoop()
                except Exception:
                    pass
        except Exception:
            pass

    def speak(self, text: str):
        text = format_for_speech(text)
        if not text:
            return
        try:
            self.q.put_nowait(text)
        except Exception:
            pass

    def _loop(self):
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
        except Exception:
            self._engine = None
        # Start non-blocking event loop for pyttsx3
        try:
            if self._engine:
                with self._rate_lock:
                    voices = self._engine.getProperty("voices") or []
                    self._voices = {
                        (getattr(voice, "name", "") or getattr(voice, "id", "")): voice.id
                        for voice in voices
                    }
                self._apply_settings()
                # False => non-blocking
                self._engine.startLoop(False)
        except Exception:
            # fallback: no loop, will run synchronously (still guarded)
            pass

        while self.running:
            if self._engine:
                try:
                    self._apply_settings()
                except Exception:
                    pass
            # drain queue: add sayings to engine
            try:
                msg = self.q.get(timeout=0.2)
            except queue.Empty:
                msg = None
            if not self.running:
                break
            if self._engine and msg:
                try:
                    self._engine.say(msg)
                except Exception:
                    # swallow TTS errors to avoid crashing the worker
                    pass
            # iterate the engine loop
            try:
                if self._engine:
                    self._engine.iterate()
            except Exception:
                # if iterate fails, try to re-init engine once
                try:
                    import pyttsx3
                    self._engine = pyttsx3.init()
                    self._engine.startLoop(False)
                except Exception:
                    pass
            time.sleep(0.02)
