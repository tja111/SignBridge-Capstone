; Compile this only after a successful portable build, using Inno Setup 6.
#define AppName "SignBridge"
#define AppVersion "1.0.0"
#define AppPublisher "SignBridge Team"
#define AppExeName "SignBridge.exe"

[Setup]
AppId={{C7E684D0-1E50-4DA2-8DF7-6289F0F8AA10}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
OutputDir=..\installer_output
OutputBaseFilename=SignBridge_Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#AppExeName}

[Files]
Source: "..\SignBridge_App\dist\SignBridge\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"
