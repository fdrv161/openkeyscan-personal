# DeepRhythm 0.7 network classes (AGPL-3.0, see __init__.py): 256 tempo values from 30 to 286 BPM.
# Separate from model.py so that the engine does not need torch.
MIN_BPM = 30
MAX_BPM = 286
NUM_CLASSES = 256
CLASS_WIDTH = (MAX_BPM - MIN_BPM) / NUM_CLASSES


def class_to_bpm(class_index):
    """As in the original: returns the lower bound of the class (an integer BPM)."""
    return MIN_BPM + CLASS_WIDTH * class_index
