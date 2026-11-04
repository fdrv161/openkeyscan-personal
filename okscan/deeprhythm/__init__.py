"""
Trimmed copy of DeepRhythm 0.7: only what is needed to detect tempo.

Source: https://github.com/bleugreen/deeprhythm (commit b1f6116e, July 2026).
The original's license is AGPL-3.0 (see LICENSE-deeprhythm next to this file). So a build
with this module falls under AGPL-3.0 too: if you distribute it, you must also provide
the source code.

Differences from the original:
- no torch, torchaudio or nnAudio: HCQM features are computed in numpy with the same
  result (see features.py), and the network is converted to models/deeprhythm.onnx
  (tools/export_onnx.py) and runs through onnxruntime;
- no downloading of weights at run time: the network ships in the build;
- no batch folder processing or CLI: our program does that.
"""
