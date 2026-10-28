# Архитектура сети DeepRhythm 0.7 без изменений (AGPL-3.0, см. __init__.py).
# Нужна только tools/export_onnx.py: движок запускает уже готовый models/deeprhythm.onnx.
import torch.nn as nn
import torch.nn.functional as F

from okscan.deeprhythm.classes import NUM_CLASSES


class DeepRhythmModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        # Вход: (6 гармоник, 240 бинов, 8 полос)
        self.conv1 = nn.Conv2d(6, 128, kernel_size=(4, 6), padding='same')
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(4, 6), padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(4, 6), padding='same')
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=(4, 6), padding='same')
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 8, kernel_size=(120, 6))
        self.bn5 = nn.BatchNorm2d(8)
        self.fc1 = nn.Linear(2904, 256)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = x.reshape(x.size(0), -1)
        x = self.dropout(self.elu(self.fc1(x)))
        return self.fc2(x)

