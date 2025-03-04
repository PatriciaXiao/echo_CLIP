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

# screen -S echo_clip_eval
# conda activate echo-clip
# CTRL+A and D
# screen -r echo_clip_eval

data_path = "/mnt/hanoverdev/data/patxiao/ECHO_numpy/20250126/"
dataset_csv = "/home/patxiao/ECHO/label_dataset_v1/HF_mini.csv"

out_csv = "/mnt/hanoverdev/data/patxiao/ECHO_results/HF_v1_mini/echo_clip_zeroshot.csv"

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

print(echo_clip)
exit(0)

# echo_clip.encode_image

heart_failure_prompts = [
    "ECHO FINDINGS OF LEFT VENTRICULAR DILATION AND REDUCED EJECTION FRACTION, SUGGESTIVE OF LEFT-SIDED HEART FAILURE.",
    "INCREASED LEFT VENTRICULAR WALL THICKNESS AND IMPAIRED DIASTOLIC RELAXATION, CONSISTENT WITH HEART FAILURE WITH PRESERVED EJECTION FRACTION.",
    "RIGHT VENTRICULAR DILATION AND REDUCED TAPSE, SUGGESTIVE OF RIGHT-SIDED HEART FAILURE.",
    "DILATED INFERIOR VENA CAVA WITH REDUCED COLLAPSIBILITY, INDICATIVE OF ELEVATED RIGHT ATRIAL PRESSURE AND RIGHT HEART FAILURE."
]
#pacemaker_prompts = zero_shot_prompts["pacemaker"]
#print(pacemaker_prompts) # 2 lines
print("heart failure prompts: ", heart_failure_prompts)

# We'll use the CLIP BPE tokenizer to tokenize the prompts
#heart_failure_prompts = tokenize(heart_failure_prompts).cuda()
heart_failure_prompts = tokenize(heart_failure_prompts).to(device)
#print("heart failure prompts: ", heart_failure_prompts)

# Now we can encode the prompts into embeddings
heart_failure_prompts_embeddings = F.normalize(
    echo_clip.encode_text(heart_failure_prompts), dim=-1
)
#print(heart_failure_prompts_embeddings.shape)

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

        heart_failure_predictions = compute_binary_metric(
            test_video_embedding, heart_failure_prompts_embeddings
        )

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


