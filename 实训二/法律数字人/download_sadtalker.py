from modelscope.hub.snapshot_download import snapshot_download
import os

save_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(save_dir, exist_ok=True)

print("Downloading SadTalker models from ModelScope...")
path = snapshot_download("wwd123/sadtalker", cache_dir=save_dir)
print("Downloaded to:", path)
