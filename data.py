from pathlib import Path
import csv
import random
import warnings

import soundfile as sf
import numpy as np
import librosa

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

# === 경고 차단 ===
warnings.filterwarnings("ignore", message="Chunk")
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import scipy.io.wavfile
    warnings.filterwarnings("ignore", category=scipy.io.wavfile.WavFileWarning)
    HAS_SCIPY = True
except (ImportError, AttributeError):
    HAS_SCIPY = False

try:
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass

def load_audio(path):
    path = str(path)

    if not Path(path).exists():
        raise FileNotFoundError(f"Audio File not found: {path}")
    try:
        waveform, sample_rate = torchaudio.load(path)

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform, sample_rate
    
    except Exception as e:
        print(f"Failed to load audio: {path}")
        print(f" reason: {e}")
        raise e


def get_audio_duration(path):
    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(f"Cannot check duration, file missing: {path}")

    try:
        info = sf.info(path)
        return info.duration
    except Exception as e:
        print(f"Warning: Failed to read info for {path}, trying fallback.")
        waveform, sample_rate = load_audio(path)
        return waveform.size(-1) / sample_rate


def crop_waveform(waveform, sample_rate, start, end):
    if start is None or end is None:
        return waveform
    try:
        start_s = float(start)
        end_s = float(end)
    except (TypeError, ValueError):
        return waveform
    if end_s <= start_s:
        return waveform
    total_samples = waveform.size(-1)
    start_idx = max(0, int(round(start_s * sample_rate)))
    end_idx = max(start_idx, int(round(end_s * sample_rate)))
    if start_idx >= total_samples:
        return waveform
    end_idx = min(end_idx, total_samples)
    if end_idx <= start_idx:
        return waveform
    return waveform[..., start_idx:end_idx]


def apply_paper_augmentation(waveform, sample_rate, aug_mode):
    """
    논문에 따른 데이터 증강 적용
    aug_mode:
      0: 원본
      1: Time Stretch 1.2 (Speed Up)
      2: Time Stretch 0.7 (Slow Down)
      3: Pitch Shift +2 semitones
      4: Pitch Shift -2 semitones
    """
    if aug_mode == 0:
        return waveform

    y = waveform.squeeze(0).numpy()
    
    try:
        if aug_mode == 1:
            y_aug = librosa.effects.time_stretch(y, rate=1.2)
        elif aug_mode == 2:
            y_aug = librosa.effects.time_stretch(y, rate=0.7)
        elif aug_mode == 3:
            y_aug = librosa.effects.pitch_shift(y, sr=sample_rate, n_steps=2)
        elif aug_mode == 4:
            y_aug = librosa.effects.pitch_shift(y, sr=sample_rate, n_steps=-2)
        else:
            y_aug = y
            
        return torch.from_numpy(y_aug).unsqueeze(0).float()
    except Exception as e:
        return waveform


class AudioPreprocessor:
    """
    🆕 IMPROVED VERSION with Enhanced SpecAugment
    
    Changes:
    - Increased FrequencyMasking: 15 → 30 (more aggressive)
    - Increased TimeMasking: 40 → 100 (more aggressive)
    - These values are based on SpecAugment paper recommendations for environmental sounds
    """
    def __init__(self, sample_rate, clip_seconds, n_mels, n_fft, hop_length, training, win_length, fixed_length=True):
        self.sample_rate = sample_rate
        self.fixed_length = fixed_length
        self.clip_samples = (
            int(sample_rate * clip_seconds) if fixed_length and clip_seconds else None
        )
        self.training = training
        self._resamplers = {}
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length, power=2.0,   
            #torch window 기본값 >> hann
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        
        # ========================================
        # 🆕 IMPROVEMENT 3: Enhanced SpecAugment
        # ========================================
        # BEFORE: freq_mask_param=25, time_mask_param=80
        # AFTER:  freq_mask_param=30, time_mask_param=100
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=25)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=80)

    def _ensure_min_length(self, waveform):
        n_fft = self.melspec.n_fft
        current_len = waveform.size(-1)
        if current_len < n_fft:
            pad_amount = n_fft - current_len
            waveform = F.pad(waveform, (0, pad_amount))
        return waveform

    def _resample(self, waveform, sample_rate):
        if sample_rate == self.sample_rate: return waveform
        if sample_rate not in self._resamplers:
            self._resamplers[sample_rate] = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
        return self._resamplers[sample_rate](waveform)


# ESC 분야에서 repeat-padding은 위험함 ,, 주기적인 소리가 애초에 환경음에 없는걸 ,,?!
    def _crop_or_pad(self, waveform):
        length = waveform.size(-1)
        target = self.clip_samples

        # 안전장치: 비정상 길이
        if length <= 0:
            return F.pad(waveform, (0, target), value=0.0)

        # Crop
        if length >= target:
            if self.training:
                start = random.randint(0, length - target)
            else:
                # 평가 시 편향 줄이기: center crop
                start = (length - target) // 2
            return waveform[..., start:start + target]

        # Pad: fade-out + zero-pad
        fade_len = min(length // 10, 1000)
        if fade_len > 0:
            fade = torch.linspace(
                1.0, 0.0, steps=fade_len,
                device=waveform.device, dtype=waveform.dtype
            )
            waveform = waveform.clone()  # (선택) 원본 보존 필요하면
            waveform[..., -fade_len:] = waveform[..., -fade_len:] * fade

        pad_amount = target - length
        return F.pad(waveform, (0, pad_amount), value=0.0)

    def _augment(self, waveform):
        """
        Random augmentation for waveform
        - Gain adjustment
        - Additive noise
        """
        # Gain
        if torch.rand(1).item() < 0.8: 
            waveform *= random.uniform(0.5, 1.5)
        # Noise
        if torch.rand(1).item() < 0.7: 
            waveform += torch.randn_like(waveform) * random.uniform(0.001, 0.02)
        return waveform

    def __call__(self, waveform, sample_rate):
        waveform = self._resample(waveform, sample_rate)
        waveform = self._crop_or_pad(waveform)
        waveform = self._ensure_min_length(waveform)
        
        if self.training: 
            waveform = self._augment(waveform)
        
        mel = self.to_db(self.melspec(waveform))
        
        # ========================================
        # 🆕 Enhanced SpecAugment applied here
        # ========================================
        if self.training:
            mel = self.time_mask(self.freq_mask(mel))
            
        return mel


def _debug_save_wav(waveform, sr, out_path):
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torchaudio.save(out_path, waveform.detach().cpu(), sr)


class AudioDataset(Dataset):
    def __init__(
        self,
        csv_path,
        root,
        sample_rate,
        clip_seconds,
        n_mels,
        n_fft,
        hop_length,
        win_length,
        training,
        segment_long=False,
        fixed_length=True,
        silence_threshold=None,
        paper_augment=False,
    ):
        self.root = Path(root)
        self.silence_threshold = silence_threshold
        self.paper_augment = paper_augment and training
        self.samples = []
        with Path(csv_path).open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.samples.append({
                    "path": row["path"], "class_index": int(row["class_index"]),
                    "class_name": row["class_name"], "binary_label": float(row["binary_label"])
                })
        if segment_long:
            self.samples = self._expand_segments(self.samples, clip_seconds)
            
        self.preprocessor = AudioPreprocessor(
            sample_rate=sample_rate,
            clip_seconds=clip_seconds,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,        # ⭐ int
            training=training,            # ⭐ bool
            fixed_length=fixed_length,
        )

    def _expand_segments(self, samples, clip_seconds):
        if not clip_seconds:
            return samples
        expanded = []
        for sample in samples:
            path = self.root / sample["path"]
            duration = get_audio_duration(path)
            if duration is None or duration < 0.1:
                continue
                
            if duration <= clip_seconds:
                expanded.append(sample)
                continue
            start = 0.0
            while start < duration:
                if duration - start < 0.1:
                    break
                end = min(start + clip_seconds, duration)
                segment = dict(sample)
                segment["start"] = start
                segment["end"] = end
                expanded.append(segment)
                start += clip_seconds
        return expanded

    def __len__(self):
        if self.paper_augment:
            return len(self.samples) * 5
        return len(self.samples)
    
    def __getitem__(self, index):
        if self.paper_augment:
            original_idx = index // 5
            aug_mode = index % 5
        else:
            original_idx = index
            aug_mode = 0

        sample = self.samples[original_idx]
        waveform, sr = load_audio(self.root / sample["path"])

        if "start" in sample and "end" in sample:
            waveform = crop_waveform(waveform, sr, sample["start"], sample["end"])

        if self.silence_threshold is not None:
            if waveform.abs().max() < self.silence_threshold:
                return None

        if self.paper_augment and aug_mode > 0:
            waveform = apply_paper_augmentation(waveform, sr, aug_mode)

        # DEBUG
        debug_on = getattr(self, "debug_save_audio", False) and index < 5
        if debug_on:
            import os, torchaudio
            from torch.utils.data import get_worker_info

            os.makedirs(getattr(self, "debug_dir", "debug_audio"), exist_ok=True)

            safe_rel = str(sample["path"]).replace("/", "_").replace("\\", "_")
            wi = get_worker_info()
            wid = wi.id if wi is not None else 0
            tag = f"w{wid}_idx{index}_orig{original_idx}_aug{aug_mode}_{safe_rel}"

            wf_before = waveform.detach().cpu()
            if wf_before.dim() == 1:
                wf_before = wf_before.unsqueeze(0)
            torchaudio.save(os.path.join(self.debug_dir, f"{tag}_before.wav"), wf_before, sr)

            wf_rs = self.preprocessor._resample(waveform, sr)
            wf_after = self.preprocessor._crop_or_pad(wf_rs)
            wf_after = wf_after.detach().cpu()
            if wf_after.dim() == 1:
                wf_after = wf_after.unsqueeze(0)
            torchaudio.save(os.path.join(self.debug_dir, f"{tag}_after_repeatpad.wav"), wf_after, self.preprocessor.sample_rate)

        x = self.preprocessor(waveform, sr)

        return x, torch.tensor(sample["class_index"], dtype=torch.long)