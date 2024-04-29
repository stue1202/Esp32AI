import os
import pyaudio
import wave
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, LlamaForCausalLM, LlamaTokenizer

# 初始化录音参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5  # 录制时长，单位：秒
WAVE_OUTPUT_FILENAME = "output.wav"

# 创建PyAudio对象
audio = pyaudio.PyAudio()

# 打开数据流
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
print("Recording...")
frames = []

# 录制音频
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Recording finished.")

# 停止数据流
stream.stop_stream()
stream.close()
audio.terminate()

# 保存录制的音频到文件中
audio_file_path = os.path.join(os.getcwd(), WAVE_OUTPUT_FILENAME)
with wave.open(audio_file_path, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

# 加载模型和处理器
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
llama_tokenizer = LlamaTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
llama_model = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# 读取录制的音频数据
with wave.open(audio_file_path, "r") as wav:
    audio_data = np.frombuffer(wav.readframes(wav.getnframes()), np.int16)
    audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max

# 使用Whisper进行语音识别
input_features = whisper_processor(audio_data, sampling_rate=RATE, return_tensors="pt").input_features
predicted_ids = whisper_model.generate(input_features)
transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

# 将转录结果输入到TinyLlama生成回复
input_ids = llama_tokenizer(transcription, return_tensors="pt").input_ids
output = llama_model.generate(input_ids, max_length=150, do_sample=True, top_k=50, top_p=0.95)
response = llama_tokenizer.decode(output[0], skip_special_tokens=True)

print(f"Transcription: {transcription}")
print(f"TinyLlama Response: {response}")