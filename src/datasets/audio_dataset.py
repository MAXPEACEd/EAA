import json
import soundfile as sf
from torch.utils.data import Dataset
import librosa
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import random
import torchaudio
import time


class AudioEmotionDataset(Dataset):
    def __init__(self, annotation_file, max_duration=30, sr=16000, n_mels=128):
        """
        Initialize the dataset
        Args:
            annotation_file (str): Path to the annotation file (JSON).
            max_duration (int): Maximum audio duration (in seconds).
            sr (int): Sampling rate.
            n_mels (int): Number of mel frequency bands.
        """
        with open(annotation_file, 'r') as f:
            self.data = json.load(f)["annotation"]
        self.max_duration = max_duration
        self.sr = sr
        self.n_mels = n_mels
        # self.label_map = {
        #     "neutral": 0,
        #     "joy": 1,
        #     "sadness": 2,
        #     "anger": 3,
        #     "fear": 4,
        #     "disgust": 5,
        #     "surprise": 6
        # }


    def time_stretch(self, audio, rate=1.0):
        """
        Apply time stretching to the audio.
        Args:
            audio: Original audio waveform
            rate: Stretching factor (e.g., >1.0 = speed up, <1.0 = slow down)
        Returns:
            Time-stretched audio
        """
        return librosa.effects.time_stretch(audio, rate)

    def add_noise_with_snr(self, audio, target_snr_db=30):
        """
        Add noise at a target SNR.
        Args:
            audio (np.array): Original audio data
            target_snr_db (float): Target SNR (dB)
        Returns:
            np.array: Audio with added noise
        """
        # 1. Compute signal power
        signal_power = torch.mean(audio ** 2)

        # 2. Compute noise power
        noise_power = signal_power / (10 ** (target_snr_db / 10))

        # 3. Generate noise
        noise = torch.randn_like(audio) * torch.sqrt(noise_power)

        # 4. Add noise to audio
        return audio + noise

    def random_crop(self, audio, crop_length):
        """
        Randomly crop a segment from the audio.
        Args:
            audio: Original audio waveform
            crop_length: Length of the segment to crop (in samples)
        Returns:
            Cropped audio segment
        """
        if len(audio) <= crop_length:
            return audio
        start = random.randint(0, len(audio) - crop_length)
        return audio[start: start + crop_length]

    def add_noise(self, audio, noise_level=0.005):
        noise = torch.randn_like(audio) * noise_level
        return audio + noise

    def __len__(self):
        return len(self.data)

    def collater(self, samples):
        """
        Collation function to pad and batch inputs.
        """
        # Process audio
        raw_wav = [s["raw_wav"].clone().detach().to(torch.float32) for s in samples]
        raw_wav_length = torch.tensor([len(s["raw_wav"]) for s in samples])
        raw_wav = pad_sequence(raw_wav, batch_first=True, padding_value=0)
        padding_mask = torch.arange(raw_wav.size(1)).unsqueeze(0) >= raw_wav_length.unsqueeze(1)

        # Only handle audio and emotion labels; text is handled inside the model
        emotions = [s["emotion"] for s in samples]

        return {
            "raw_wav": raw_wav,
            "raw_wav_length": raw_wav_length,
            "padding_mask": padding_mask,
            "utterance_with_context": [s["utterance_with_context"] for s in samples],
            "emotion": emotions
        }

    def __getitem__(self, idx):
        ann = self.data[idx]
        audio_path = ann["path"]
        emotion = ann["emotion"]
        current_utterance = ann["utterance"]

        # Try to read previous dialogue context
        context_utterance = ""
        if idx > 0:  # If a previous utterance exists
            prev_dialogue_id = self.data[idx - 1]["path"].split("/")[-1].split("_")[0]  # Previous dialogue ID
            current_dialogue_id = audio_path.split("/")[-1].split("_")[0]
            if prev_dialogue_id == current_dialogue_id:  # Confirm same dialogue context
                context_utterance = self.data[idx - 1]["utterance"]

        # Concatenate current utterance with context
        utterance_with_context = f"{context_utterance} [SEP] {current_utterance}" if context_utterance else current_utterance

        utterance_with_context = current_utterance
        # Load audio with torchaudio
        audio, sr = torchaudio.load(audio_path)
        audio = audio.mean(dim=0)  # Convert to mono if multi-channel

        # Resample
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)
            audio = resampler(audio)

        # Ensure valid audio range
        audio = torch.clamp(audio, min=-1.0, max=1.0)

        # Pad with silence if too short
        if audio.shape[-1] < self.sr:
            silence = torch.zeros(self.sr - audio.shape[-1])
            audio = torch.cat((audio, silence), dim=0)

        actual_length = min(len(audio), self.sr * self.max_duration)
        audio = audio[:actual_length]

        return {
            "raw_wav": audio,
            "raw_wav_length": actual_length,
            "utterance_with_context": utterance_with_context,
            "emotion": emotion,
            "path": audio_path
        }


if __name__ == "__main__":
    # Example usage
    dataset = AudioEmotionDataset(annotation_file="/home/hongfei/emollm/data/train_annotation.json")
    print(f"Total samples: {len(dataset)}")

    # Get a single sample
    sample = dataset[0]
    print("Raw waveform shape:", sample["raw_wav"].shape)
    print("Sample mel spectrogram shape:", sample["mel_spectrogram"].shape)
    print("Sample emotion:", sample["emotion"])
    print("Sample audio path:", sample["path"])