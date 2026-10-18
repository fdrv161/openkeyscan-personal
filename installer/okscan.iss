; Установщик личной сборки OpenKeyScan (Inno Setup 6).
; Файл в UTF-8 с BOM — иначе Inno Setup прочитает русские строки в неверной кодировке.
; Ставится в профиль пользователя, права администратора не нужны.
; Сборка: iscc /DAppVersion=1.0.0 installer\okscan.iss  (после pyinstaller okscan.spec)

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif

[Setup]
AppId={{6E0B7C3A-4F1D-4C55-9A8E-2D3B5F7A1C90}
AppName=OpenKeyScan (личная сборка)
AppVersion={#AppVersion}
AppPublisher=Личная сборка
DefaultDirName={localappdata}\Programs\OpenKeyScan-Personal
DefaultGroupName=OpenKeyScan (личная сборка)
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=OpenKeyScan-Personal-Setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
LZMANumBlockThreads=4
UninstallDisplayIcon={app}\OpenKeyScan.exe
WizardStyle=modern

[Languages]
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "Ярлык на рабочем столе"; Flags: unchecked
Name: "sendto"; Description: "Пункт «Отправить → OpenKeyScan» в меню файлов и папок"

[Files]
Source: "..\dist\OpenKeyScan\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\OpenKeyScan (личная сборка)"; Filename: "{app}\OpenKeyScan.exe"
Name: "{autodesktop}\OpenKeyScan (личная сборка)"; Filename: "{app}\OpenKeyScan.exe"; Tasks: desktopicon
Name: "{usersendto}\OpenKeyScan"; Filename: "{app}\OpenKeyScan.exe"; Tasks: sendto

[Run]
Filename: "{app}\OpenKeyScan.exe"; Description: "Запустить OpenKeyScan"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Кэш ускорения (numba) создаётся программой; настройки пользователя не трогаем.
Type: filesandordirs; Name: "{userappdata}\OpenKeyScan-Personal\numba-cache"
