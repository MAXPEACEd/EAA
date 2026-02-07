import torch
import torchaudio
from transformers import HubertModel
from src.models.hubert_beats_model import AudioClassificationModel

def test_model():
    # 模型路径
    hubert_model_path = "/home/hongfei/emollm/src/models/hubert"
    beats_model_path = "/home/hongfei/emollm/src/models/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"

    # 初始化模型
    model = AudioClassificationModel(
        hubert_model_path=hubert_model_path,
        beats_model_path=beats_model_path,
        hidden_size=256,
        num_classes=7,
        freeze_beats=True,
        freeze_hubert=True
    )
    model.eval()

    # 加载音频数据并重采样到 16kHz
    batch_size = 4
    audio_path = "/home/hongfei/emollm/src/models/duck.wav"  # 替换为你的音频文件路径
    waveform, sample_rate = torchaudio.load(audio_path)  # (channels, num_samples)
    waveform = waveform.mean(dim=0)  # 转为单通道
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # HuBERT 输入：处理波形数据
    speech_input = waveform.unsqueeze(0).repeat(batch_size, 1)  # 扩展成 batch_size

    # BEATs 输入：需要将 waveform 转换为 BEATs 所需的格式
    acoustic_input = speech_input

    # 测试模型
    with torch.no_grad():
        logits = model(speech_input=speech_input, acoustic_input=acoustic_input)

    # 输出结果
    print("Logits shape:", logits.shape)  # 应该是 [batch_size, num_classes]
    print("Logits:", logits)

if __name__ == "__main__":
    test_model()
