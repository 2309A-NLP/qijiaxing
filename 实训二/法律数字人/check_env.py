import gradio
import torch
import cv2
import librosa
import scipy

print("gradio", gradio.__version__)
print("torch", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
