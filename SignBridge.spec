# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller definition for the portable SignBridge Windows application."""
from pathlib import Path
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH)
ICON = ROOT / "assets" / "signbridge.ico"
PYTHON_ROOT = Path(sys.base_prefix)

datas = [
    (str(ROOT / "checkpoints" / "alphabet_model.pt"), "checkpoints"),
    (str(ROOT / "checkpoints" / "words" / "words_model.pt"), "checkpoints/words"),
    # Stores the Words Mode class order, box-head architecture, and input size.
    (str(ROOT / "checkpoints" / "words" / "meta.json"), "checkpoints/words"),
    (str(ROOT / "assets"), "assets"),
    (str(ROOT / "src" / "config.json"), "."),
]
datas += collect_data_files("customtkinter")
# This Python installation's Tcl/Tk auto-discovery is unreliable, so bundle
# the complete runtime explicitly for tkinter, Pillow ImageTk, and CustomTkinter.
datas += [
    # These names are required by PyInstaller's bundled pyi_rth_tkinter hook.
    (str(PYTHON_ROOT / "tcl" / "tcl8.6"), "_tcl_data"),
    (str(PYTHON_ROOT / "tcl" / "tk8.6"), "_tk_data"),
    # PyInstaller's analyzer suppresses tkinter when Tcl auto-detection fails.
    # Ship the stdlib package as runtime files so imports resolve regardless.
    (str(PYTHON_ROOT / "Lib" / "tkinter"), "tkinter"),
]

binaries = collect_dynamic_libs("torch") + collect_dynamic_libs("torchvision") + [
    (str(PYTHON_ROOT / "DLLs" / "_tkinter.pyd"), "."),
    (str(PYTHON_ROOT / "DLLs" / "tcl86t.dll"), "."),
    (str(PYTHON_ROOT / "DLLs" / "tk86t.dll"), "."),
]
hiddenimports = [
    "tkinter", "tkinter.ttk", "tkinter.messagebox", "_tkinter",
    "albumentations.pytorch", "albumentations.pytorch.transforms",
    "pyttsx3.drivers", "pyttsx3.drivers.sapi5",
    "cv2", "PIL.ImageTk", "torchvision.models", "torchvision.ops",
] + collect_submodules("tkinter")

a = Analysis(
    [str(ROOT / "src" / "signbridge_app.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "src" / "runtime_hooks" / "tkinter_runtime.py")],
    excludes=["jupyter", "label_studio", "matplotlib"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="SignBridge", debug=False, bootloader_ignore_signals=False,
    strip=False, upx=False, console=False,
    icon=str(ICON) if ICON.exists() else None,
)
coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=False, upx_exclude=[],
    name="SignBridge",
)
