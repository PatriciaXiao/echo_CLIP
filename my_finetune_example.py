from open_clip import tokenize, create_model_and_transforms
import torchvision.transforms as T
import torch
import torch.nn.functional as F
from utils import (
    zero_shot_prompts,
    compute_binary_metric,
    compute_regression_metric,
    read_avi,
)
from finetune_encoder.eval_metrics import get_classification_metrics
import pandas as pd
import numpy as np
import os
import torch.nn as nn

# screen -S echo_clip_finetune
# conda activate echo-clip
# CTRL+A and D
# screen -r echo_clip_finetune

data_path = "/mnt/hanoverdev/data/patxiao/ECHO_numpy/20250126/"
dataset_csv = "/home/patxiao/ECHO/label_dataset_v1/HF_mini.csv"

out_csv = "/mnt/hanoverdev/data/patxiao/ECHO_results/HF_v1_mini/echo_clip_finetune.csv"

# zero shot, no training
dataset = pd.read_csv(dataset_csv)
#print(len(dataset))
dataset = dataset[dataset["split"] != "train"]
#print(len(dataset))
#print(dataset)

path_list = [os.path.join(data_path, p) for p in list(dataset["path"])]
split_list = list(dataset["split"])
label_list = list(dataset["label"])

out_data = {"path": list(), "label": list(), "split": list(), "predict": list()}
if os.path.exists(out_csv):
    saved_df = pd.read_csv(out_csv)
    for field in out_data.keys():
        out_data[field] = list(saved_df[field])

device_name = "cuda:1"
device = torch.device(device_name)


# You'll need to log in to the HuggingFace hub CLI to download the models
# You can do this with the terminal command "huggingface-cli login"
# You'll be asked to paste your HuggingFace API token, which you can find at https://huggingface.co/settings/token

# Use EchoCLIP for zero-shot tasks like ejection fraction prediction
# or pacemaker detection. It has a short context window because it
# uses the CLIP BPE tokenizer, so it can't process an entire report at once.
echo_clip, _, preprocess_val = create_model_and_transforms(
    "hf-hub:mkaichristensen/echo-clip", precision="bf16", device=device_name
)

image_encoder = echo_clip.visual #.to(device)
#print(image_encoder)

class EchoClassifier(nn.Module):
    def __init__(self, image_encoder, num_classes):
        super(EchoClassifier, self).__init__()
        self.encoder = image_encoder  # Use CLIP image encoder
        self.fc = nn.Linear(512, num_classes)  # This encoder's output dimension is indeed 512 dim

    def forward(self, x):
        x = self.encoder(x)  # Extract image features
        x = self.fc(x)  # Classification head
        return x

num_classes = 2  # Example: 4 for "normal", "mild", "moderate", "severe"
model = EchoClassifier(image_encoder, num_classes).to(device)

#Fine-tuning only the last few layers prevents overfitting:

for param in model.encoder.parameters():
    param.requires_grad = False  # Freeze all layers

# Unfreeze the last transformer block (ViT) or last few layers (ResNet)
for param in list(model.encoder.parameters())[-5:]:
    param.requires_grad = True

exit(0)

num_cases = len(dataset)
for idx,(path,split,label) in enumerate(zip(path_list, split_list, label_list)):
    if path in out_data["path"]:
        print("({}/{}): already processed".format(idx+1, num_cases, path))
    elif os.path.exists(path):
        test_video = np.load(path)
        print("({}/{}): processing {}".format(idx+1, num_cases, path))
        #print(test_video.shape) # (3, 60, 256, 256)
        """
        data = list()
        for i in range(test_video.shape[0]):
            data.append(test_video[i])
        test_video = np.stack(data, axis=-1)
        """
        test_video = np.stack(test_video, axis=-1)
        #print(test_video.shape)
        # process the data, do normalization
        test_video = torch.stack(
            [preprocess_val(T.ToPILImage()(frame)) for frame in test_video], dim=0
        )
        #print(test_video.shape) # [60, 3, 224, 224]
        #test_video = test_video.cuda()
        test_video = test_video.to(device)
        test_video = test_video.to(torch.bfloat16)

        # turn it into echo clip image encoding
        test_video_embedding = F.normalize(echo_clip.encode_image(test_video), dim=-1)
        # Add in a batch dimension because the zero-shot functions expect one
        test_video_embedding = test_video_embedding.unsqueeze(0)

        #heart_failure_predictions = compute_binary_metric(
        #    test_video_embedding, heart_failure_prompts_embeddings
        #)

        predict = heart_failure_predictions.item()

        out_data["path"].append(path)
        out_data["label"].append(label)
        out_data["split"].append(split)
        out_data["predict"].append(predict)

        df = pd.DataFrame(data=out_data)
        df.to_csv(out_csv, index=None)
        #exit(0)
    else:
        print("({}/{}): {} refused access".format(idx+1, num_cases, path))

    # debug
    #if idx > 10:
    #    break


df = pd.DataFrame(data=out_data)
df = df[df["split"] == "test"]

pred_val = np.array(df["predict"])
labels = np.array(df["label"])
output = torch.Tensor(np.column_stack((1 - pred_val, pred_val)))
target = torch.Tensor(labels.reshape(-1, 1))

label_dict = {0:0, 1:1}
mode = "binary"

print(output, target, label_dict, mode)
eval_stats = get_classification_metrics(output, target, label_dict, mode=mode)
print(eval_stats)


