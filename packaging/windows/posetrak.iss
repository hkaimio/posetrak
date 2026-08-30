; SPDX-FileCopyrightText: 2026 Harri Kaimio
;
; SPDX-License-Identifier: Apache-2.0

; Posetrak Windows installer (Inno Setup script)
;
; installer-prototype-plan.md Phase 2: wraps the exact same
; thin-bootstrap folder layout hand-validated in Phase 1 (uv.exe, the
; pre-built C++ tracker + its runtime DLLs, a pinned Python source
; snapshot, and the launch.bat/launch.ps1 entry point) into a proper
; installer -- Start Menu shortcut, uninstaller, and an honest notice
; about the unsigned-binary SmartScreen warning (see code-signing-plan.md
; for why this ships unsigned for now).
;
; Per-user install, no admin/UAC prompt: this mirrors the app's own
; %USERPROFILE%\.posetrak tracker-binary convention and keeps the
; unsigned-prototype experience as low-friction as possible.
;
; Build (from the repo root):
;   "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" ^
;       /DSourceDir="D:\mocap\posetrak-bootstrap-proto" ^
;       packaging\windows\posetrak.iss
;
; SourceDir must contain exactly Phase 1's bootstrap-folder layout:
; uv.exe, tracker\, app\, launch.bat, launch.ps1, README.txt. Defaults
; to a sibling "dist" directory if not given explicitly -- a future CI
; release workflow would assemble that directory itself and pass its
; own path the same way.
;
; Branding assets (assets\*.ico/*.bmp) are generated from the master
; artwork in branding\ (posetrak-logo-abstract.png for the icon/small
; wizard image, posetrak-logo-aikido.png for the wizard banner) --
; regenerate them from those masters rather than hand-editing the
; derived files if the logo ever changes.

#ifndef SourceDir
  #define SourceDir "dist"
#endif

#define MyAppName "Posetrak"
#define MyAppVersion "0.1.0-proto2"
#define MyAppPublisher "Harri Kaimio"
#define MyAppURL "https://github.com/hkaimio/posetrak"

[Setup]
AppId={{E33049C2-BAE1-4CE2-AAC8-3B8155EE3400}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
; No code signing yet (see code-signing-plan.md) -- install per-user so
; no admin/UAC prompt is needed, and be upfront about the resulting
; SmartScreen warning via InfoBeforeFile below rather than hiding it.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
LicenseFile=..\..\LICENSE
InfoBeforeFile=smartscreen-notice.txt
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
OutputDir=output
OutputBaseFilename=posetrak-setup-{#MyAppVersion}
; Branding (logo by Nelli Kaimio -- see branding/ and REUSE.toml).
SetupIconFile=assets\posetrak-icon.ico
WizardImageFile=assets\wizard-banner.bmp
WizardSmallImageFile=assets\wizard-small.bmp
UninstallDisplayIcon={app}\posetrak-icon.ico

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "{#SourceDir}\uv.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "assets\posetrak-icon.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\launch.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\launch.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\README.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\tracker\*"; DestDir: "{app}\tracker"; Flags: ignoreversion recursesubdirs createallsubdirs
; Excludes guard against whatever happens to be sitting in SourceDir\app at
; build time -- a .venv, __pycache__, egg-info, or log files left behind by
; a prior `uv sync`/app run against that same folder (including one done
; *inside* a Windows Sandbox session using this as its live-mapped folder;
; a non-portable .venv or a stray log file must never end up bundled into
; the installer even if the staging folder isn't perfectly clean).
Source: "{#SourceDir}\app\*"; DestDir: "{app}\app"; Excludes: ".venv,__pycache__,*.egg-info,logs,*.pyc"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\launch.bat"; WorkingDir: "{app}"; IconFilename: "{app}\posetrak-icon.ico"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\launch.bat"; WorkingDir: "{app}"; IconFilename: "{app}\posetrak-icon.ico"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
; Quick starting point per Harri, 2026-08-23: an install-time way to pull
; in the segmentation extras (torch/Cutie) is being added now; a more
; discoverable in-app "install additional components later" action is
; still wanted too (see docs/roadmap/features/packaging/status.md's
; TODO) but is real design work, not a five-minute addition.
Name: "cutie"; Description: "Install GPU segmentation support (PyTorch/Cutie) -- several GB extra download, requires an NVIDIA GPU"; GroupDescription: "Optional components:"; Flags: unchecked

[Run]
; Runs uv's own console output visibly (no runhidden) -- this can be a
; multi-GB download, and a silently "stuck" wizard page would look
; broken. Placed before the launch entry so first launch doesn't race a
; second, redundant sync.
Filename: "{app}\uv.exe"; Parameters: "sync --group segmentation --project ""{app}\app"""; WorkingDir: "{app}"; StatusMsg: "Installing GPU segmentation support (this can take several minutes and needs internet access)..."; Tasks: cutie; Flags: waituntilterminated
Filename: "{app}\launch.bat"; Description: "Launch {#MyAppName} now"; Flags: postinstall nowait skipifsilent

[UninstallDelete]
; uv sync creates app\.venv (and .python-version-provisioned interpreters
; live under %USERPROFILE%\AppData\Roaming\uv, untouched here on purpose --
; shared across any other uv-based tool, not this installer's to remove).
; Clean up what this installer's own launch actually creates under {app}.
Type: filesandordirs; Name: "{app}\app\.venv"
Type: filesandordirs; Name: "{app}\app\python\logs"
