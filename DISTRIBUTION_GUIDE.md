# SignBridge distribution guide

## Build

1. Install the dependencies from `requirements.txt` into the project's Python environment. PyTorch must match the intended target: the current environment uses a CUDA build, but a CPU PyTorch build is suitable for groupmates without NVIDIA GPUs.
2. Optional: add `assets/signbridge.ico` before building.
3. Double-click `build_signbridge.bat`. It checks that Tcl/Tk is installed before packaging; repair/reinstall Python with Tcl/Tk support if this check fails.
4. The portable application is `SignBridge_App\dist\SignBridge\SignBridge.exe`.

The build is deliberately **one-folder**. Give groupmates a ZIP of the complete `SignBridge_App\dist\SignBridge` folder; they extract it and double-click `SignBridge.exe`. Do not share only the executable.

## Included resources

The build includes `checkpoints\alphabet_model.pt`, `checkpoints\words\words_model.pt`, and all `assets\signs` images. To update a model later, replace the corresponding checkpoint in the source project and rebuild. Keep the filename and the model's matching class architecture.

## GPU and CPU

SignBridge selects CUDA when it is available; otherwise it uses CPU. CPU inference is slower but does not change the trained models or require an NVIDIA GPU. A CPU-only PyTorch environment can make the package substantially smaller; build it only after testing recognition with that environment.

## Troubleshooting

- **Model missing/loading failed:** confirm the full distribution folder was extracted. Technical details are in `logs\signbridge.log`; if installed under Program Files, the log is under `%LOCALAPPDATA%\SignBridge\logs`.
- **Camera not available:** connect/enable a webcam, select the correct camera number, and retry.
- **Windows warning:** distribute the full ZIP or installer from a trusted source; unsigned local executables can trigger SmartScreen.
- **No icon:** add `assets\signbridge.ico` and rebuild.

## Optional installer

After confirming the portable build, open `installer\SignBridgeInstaller.iss` with Inno Setup 6 and compile it. The resulting `installer_output\SignBridge_Setup.exe` installs the complete portable distribution, Start Menu shortcut, optional Desktop shortcut, and uninstaller.
