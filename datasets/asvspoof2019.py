"""ASVspoof 2019 LA Dataset Loader for HCN-TA."""

import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from .preprocessing import AudioPreprocessor, MelSpectrogramGenerator


class ASVspoof2019Dataset(Dataset):
    def __init__(self, file_list, mode="audio", transform=None, sr=16000, duration=4.0):
        self.file_list = file_list
        self.mode = mode
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        if mode == "audio":
            self.preprocessor = AudioPreprocessor(sample_rate=sr, duration=duration)
            self.mel_gen = MelSpectrogramGenerator(sample_rate=sr)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fp, label = self.file_list[idx]
        if self.mode == "spectrogram":
            img = Image.open(fp).convert("RGB")
            img = self.transform(img)
        else:
            wav = self.preprocessor.load_audio(fp)
            img = self.mel_gen.generate(wav)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img, torch.tensor(label, dtype=torch.long)


class ASVspoof2019Builder:
    LABEL_MAP = {"bonafide": 0, "spoof": 1}

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

    def _parse(self, protocol, audio_dir):
        files = []
        with open(protocol) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    name, lbl = parts[1], parts[4]
                    fp = Path(audio_dir) / f"{name}.flac"
                    if fp.exists():
                        files.append((str(fp), self.LABEL_MAP.get(lbl, 1)))
        return files

    def get_train(self, **kw):
        fl = self._parse(
            self.data_dir / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
            self.data_dir / "ASVspoof2019_LA_train/flac")
        return ASVspoof2019Dataset(fl, **kw)

    def get_dev(self, **kw):
        fl = self._parse(
            self.data_dir / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
            self.data_dir / "ASVspoof2019_LA_dev/flac")
        return ASVspoof2019Dataset(fl, **kw)

    def get_eval(self, **kw):
        fl = self._parse(
            self.data_dir / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
            self.data_dir / "ASVspoof2019_LA_eval/flac")
        return ASVspoof2019Dataset(fl, **kw)
