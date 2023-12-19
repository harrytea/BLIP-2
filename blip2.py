import sys
print(sys.path)

# 把LAVIS这个包加入到python的第一个搜索路径中
# 这样就不会调用conda中的包了，可以不设置justMyCode为false来进行调试
sys.path.insert(1,'/data/wangyh/mllms/blip2/LAVIS')
print(sys.path)

import torch
from PIL import Image
from LAVIS.lavis.models import load_model_and_preprocess

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
raw_image = Image.open("merlion.png").convert("RGB")


# loads BLIP-2 pre-trained model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device)
# prepare the image
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
model.generate({"image": image, "prompt": "Question: which city is this? Answer:"})
