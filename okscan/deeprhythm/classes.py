# Классы сети DeepRhythm 0.7 (AGPL-3.0, см. __init__.py): 256 значений темпа от 30 до 286 BPM.
# Отдельно от model.py, чтобы движку не нужен был torch.
MIN_BPM = 30
MAX_BPM = 286
NUM_CLASSES = 256
CLASS_WIDTH = (MAX_BPM - MIN_BPM) / NUM_CLASSES


def class_to_bpm(class_index):
    """Как в оригинале: возвращает нижнюю границу класса (то есть целое число BPM)."""
    return MIN_BPM + CLASS_WIDTH * class_index
