"""FoR Dataset Loader for HCN-TA."""

from pathlib import Path
from sklearn.model_selection import train_test_split
from .asvspoof2019 import ASVspoof2019Dataset  # reuse same Dataset class


class FoRBuilder:
    AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}

    def __init__(self, data_dir, train_ratio=0.7, val_ratio=0.15):
        self.data_dir = Path(data_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def _collect(self):
        files = []
        for label_name, label in [("real", 0), ("fake", 1)]:
            for f in self.data_dir.rglob("*"):
                if f.suffix.lower() in self.AUDIO_EXTS and label_name in str(f).lower():
                    files.append((str(f), label))
        return files

    def build(self, **kw):
        all_files = self._collect()
        if not all_files:
            raise RuntimeError(f"No audio files found in {self.data_dir}")
        fps, lbls = zip(*all_files)

        tr_fp, tmp_fp, tr_lb, tmp_lb = train_test_split(
            fps, lbls, test_size=1 - self.train_ratio, random_state=42, stratify=lbls)
        rel = 0.5
        va_fp, te_fp, va_lb, te_lb = train_test_split(
            tmp_fp, tmp_lb, test_size=rel, random_state=42, stratify=tmp_lb)

        return {
            "train": ASVspoof2019Dataset(list(zip(tr_fp, tr_lb)), **kw),
            "val": ASVspoof2019Dataset(list(zip(va_fp, va_lb)), **kw),
            "test": ASVspoof2019Dataset(list(zip(te_fp, te_lb)), **kw),
        }
