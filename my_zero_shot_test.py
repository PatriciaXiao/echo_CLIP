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

out_data = {"path": list(), "label": list(), "split": list()}
if os.path.exists(out_csv):
    saved_df = pf.read_csv(out_csv)
    for field in out_data.keys():
        out_data[field] = list(saved_df[field])




# You'll need to log in to the HuggingFace hub CLI to download the models
# You can do this with the terminal command "huggingface-cli login"
# You'll be asked to paste your HuggingFace API token, which you can find at https://huggingface.co/settings/token

# Use EchoCLIP for zero-shot tasks like ejection fraction prediction
# or pacemaker detection. It has a short context window because it
# uses the CLIP BPE tokenizer, so it can't process an entire report at once.
echo_clip, _, preprocess_val = create_model_and_transforms(
    "hf-hub:mkaichristensen/echo-clip", precision="bf16", device="cuda"
)

num_cases = len(dataset)
for idx,(path,split,label) in enumerate(zip(path_list, split_list, label_list)):
    if os.path.exists(path):
        test_video = np.load(path)
        print(test_video.shape)
        exit(0)

exit(0)


# We'll use random noise in the shape of a 10-frame video in this example, but you can use any image
# We'll load a sample echo video and preprocess its frames.
test_video = read_avi(
    "example_video.avi",
    (250, 250), # this also works? yes
    #(224, 224),
)
#print(test_video) # original scale
print("test_video shape: ", test_video.shape) # (113, 224, 224, 3)
test_video = torch.stack(
    [preprocess_val(T.ToPILImage()(frame)) for frame in test_video], dim=0
)
test_video = test_video[0:min(40, len(test_video)):2]
print("processed test_video shape: ", test_video.shape) # [20, 3, 224, 224]
#print(test_video) # normalized
test_video = test_video.cuda()
test_video = test_video.to(torch.bfloat16)

# Be sure to normalize the CLIP embedding after calculating it to make
# cosine similarity between embeddings easier to calculate.
test_video_embedding = F.normalize(echo_clip.encode_image(test_video), dim=-1)
print("test_video_embedding shape: ", test_video_embedding.shape) # torch.Size([20, 512])

# Add in a batch dimension because the zero-shot functions expect one
test_video_embedding = test_video_embedding.unsqueeze(0)
print("processed test_video_embedding shape: ", test_video_embedding.shape) # torch.Size([1, 20, 512])

# To perform zero-shot prediction on our "echo" image, we'll need
# prompts that describe the task we want to perform. For example,
# to zero-shot detect pacemakers, we'll use the following prompts
heart_failure_prompts = [
    "ECHO FINDINGS OF LEFT VENTRICULAR DILATION AND REDUCED EJECTION FRACTION, SUGGESTIVE OF LEFT-SIDED HEART FAILURE.",
    "INCREASED LEFT VENTRICULAR WALL THICKNESS AND IMPAIRED DIASTOLIC RELAXATION, CONSISTENT WITH HEART FAILURE WITH PRESERVED EJECTION FRACTION.",
    "RIGHT VENTRICULAR DILATION AND REDUCED TAPSE, SUGGESTIVE OF RIGHT-SIDED HEART FAILURE.",
    "DILATED INFERIOR VENA CAVA WITH REDUCED COLLAPSIBILITY, INDICATIVE OF ELEVATED RIGHT ATRIAL PRESSURE AND RIGHT HEART FAILURE."
]
#pacemaker_prompts = zero_shot_prompts["pacemaker"]
#print(pacemaker_prompts) # 2 lines

# We'll use the CLIP BPE tokenizer to tokenize the prompts
heart_failure_prompts = tokenize(heart_failure_prompts).cuda()
print(heart_failure_prompts)

# Now we can encode the prompts into embeddings
heart_failure_prompts_embeddings = F.normalize(
    echo_clip.encode_text(heart_failure_prompts), dim=-1
)
print(heart_failure_prompts_embeddings.shape) # torch.Size([2, 512])

# Now we can compute the similarity between the video and the prompts
# to get a prediction for whether the video contains a pacemaker. It's
# important to note that this prediction is not calibrated, and can
# range from -1 to 1.
heart_failure_predictions = compute_binary_metric(
    test_video_embedding, heart_failure_prompts_embeddings
)

# If we use a pacemaker detection threshold calibrated using its F1 score on
# our test set, we can get a proper true/false prediction prediction.
f1_calibrated_threshold = 0.298
print("Raw score: ", heart_failure_predictions.item())
print(f"Heart Failure detected: {heart_failure_predictions.item() > f1_calibrated_threshold}")

pred = heart_failure_predictions.item()
output = torch.Tensor([[1-pred, pred], [1-pred, pred]])
target = torch.Tensor([[0.0], [1.0]])
label_dict = {0:0, 1:1}
mode = "binary"

eval_stats = get_classification_metrics(output, target, label_dict, mode=mode)
print(eval_stats)


