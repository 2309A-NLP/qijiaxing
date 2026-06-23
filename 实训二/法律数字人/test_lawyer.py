import sys
sys.path.append('./')

from TFG.SadTalker import SadTalker

sadtalker = SadTalker(checkpoint_path='checkpoints', config_path='src/config')

source_image = "inputs/lawyer.png"
source_audio  = "inputs/test.wav"

result = sadtalker.test2(
    source_image=source_image,
    driven_audio=source_audio,
    preprocess='crop',
    still_mode=False,
    use_enhancer=False,
    batch_size=1,
    size=256,
    result_dir='results/lawyer_test/'
)

print(f"\n✅ 生成完成！视频路径：{result}")
