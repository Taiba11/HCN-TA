"""
Single-File Inference for HCN-TA.

Usage:
    python scripts/inference.py --checkpoint experiments/best_model.pth --audio_path path/to/audio.wav
"""

import os, sys, argparse
import torch
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import HCNTA
from datasets.preprocessing import AudioPreprocessor, MelSpectrogramGenerator


def main():
    p = argparse.ArgumentParser(description="HCN-TA Inference")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--audio_path", required=True)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = HCNTA(num_classes=2).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    preprocessor = AudioPreprocessor(sample_rate=16000, duration=4.0)
    mel_gen = MelSpectrogramGenerator(sample_rate=16000)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    print(f"\nProcessing: {args.audio_path}")
    wav = preprocessor.load_audio(args.audio_path)
    mel = normalize(mel_gen.generate(wav)).unsqueeze(0).to(device)

    with torch.no_grad():
        v, attn_weights = model(mel, return_attention=True)
        preds, confs = model.predict(mel)

    label = "REAL (Bonafide)" if preds.item() == 0 else "FAKE (Spoofed)"
    c = confs.squeeze()

    print(f"\n{'='*50}")
    print(f"  Prediction:  {label}")
    print(f"  Confidence:  Real={c[0]:.4f} | Fake={c[1]:.4f}")
    print(f"  Attention:   max={attn_weights.max():.4f} at t={attn_weights.argmax().item()}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
