# OpenKeyScan for REAPER

[Справка на русском](README.ru.md)

Detects the key (regular name, Camelot and Open Key) and tempo (BPM) of the selected items right in REAPER. The scripts have no window of their own: select an item, press a shortcut, get the answer.

Audio files are only read, nothing is ever written to them.

## Requirements

- Windows 10 or 11, 64-bit.
- REAPER 6 or newer.
- About 200 MB of free disk space.

## Installation

1. In REAPER open Options, then Show REAPER resource path in explorer/finder, and go to the Scripts folder.
2. Unpack the whole OpenKeyScan folder from the archive there. It should contain the .lua files, this README and the engine folder.
3. Open Actions, then Show action list, press New action, then Load ReaScript, and choose "OpenKeyScan - Install actions.lua".
4. Run the "OpenKeyScan - Install actions" action once. It adds the other three actions to the action list. OpenKeyScan_core.lua is shared code and does not need loading.
5. Optionally assign shortcuts to the actions.

Updating from version 2.0.0: action file names have changed. Delete the old OpenKeyScan folder, remove the old actions from the action list, install as above and assign your shortcuts again.

## Actions

- "OpenKeyScan - Key and BPM of selected items" reports the result for each selected item, for example "beat: 88 BPM, 7A, D minor".
- "OpenKeyScan - Key and BPM, set project tempo" does the same and also:
  - sets the project tempo from the first selected item (if the project has no tempo markers);
  - appends the key to the track name: "instr" becomes "instr [D minor, 7A]";
  - moves all project items so that the first downbeat of the beat lands on a bar line.

  All of this is undone with a single Ctrl+Z.
- "OpenKeyScan - Cycle BPM range" cycles the range the tempo is folded into: 50 to 100, 70 to 140, or no folding. The choice is remembered.

The whole source file of the item is analyzed. MIDI and empty items are skipped.

## Good to know

- By default the tempo is folded into the 50 to 100 BPM range: 133 BPM is shown as 66.5. Turn on the full tempo with the "Cycle BPM range" action.
- If the network is not sure about the tempo, the answer ends with "tempo uncertain, check by ear".
- If the file's tags already contain both BPM and key, the first action answers from the tags without analysis and says "From file tags". Tags can be inaccurate: to check by analysis, use the second action, it does not read tags.
- Analyzing a 3-minute track takes a couple of seconds. The very first run after unpacking can take longer, because Windows checks the new files.
- Results are remembered in the engine\cache folder. Analyzing the same file again takes a fraction of a second. If the file has been overwritten, it is analyzed again.
- Without OSARA the result is shown in a message window.

## Using a screen reader

If OSARA is installed in REAPER, all messages (start of analysis, result, errors) are spoken by your screen reader: NVDA, JAWS or Narrator. No windows appear and the focus stays where it was.

## Uninstalling

Delete the OpenKeyScan folder from Scripts and remove the actions from the action list. Temporary analysis files are in the %TEMP%\OpenKeyScan-REAPER folder and are deleted automatically.

## Licenses

- OpenKeyScan key model: MIT (rekordcloud, based on MusicalKeyCNN).
- DeepRhythm 0.7 tempo detection: AGPL-3.0 (bleugreen). That is why the whole package is distributed under AGPL-3.0.

The license texts are in the engine\_internal\licenses folder.
