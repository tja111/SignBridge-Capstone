"""Set bundled Tcl/Tk locations before tkinter or Pillow ImageTk imports."""
import os
import sys
from pathlib import Path

bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
os.environ.setdefault("TCL_LIBRARY", str(bundle_root / "_tcl_data"))
os.environ.setdefault("TK_LIBRARY", str(bundle_root / "_tk_data"))
