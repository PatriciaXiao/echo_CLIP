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
#dataset = pd.read_csv(dataset_csv)
#print(len(dataset))
#dataset = dataset[dataset["split"] != "train"]
#print(len(dataset))
#print(dataset)

#path_list = list(dataset["path"]) #[os.path.join(data_path, p) for p in list(dataset["path"])]
#split_list = list(dataset["split"])
#label_list = list(dataset["label"])

#out_data = {"path": list(), "label": list(), "split": list(), "predict": list()}
#if os.path.exists(out_csv):
#    saved_df = pd.read_csv(out_csv)
#    for field in out_data.keys():
 #       out_data[field] = list(saved_df[field])

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

#image_encoder = echo_clip.visual #.to(device)
#print(image_encoder)

image_encoder = echo_clip #.to(device)
for param in image_encoder.transformer.parameters():  # CLIP text encoder
    param.requires_grad = False
for param in image_encoder.visual.parameters():
    param.requires_grad = True 

# for param in model.transformer.parameters():  # CLIP text encoder
#     param.requires_grad = False

class EchoClassifier(nn.Module):
    def __init__(self, image_encoder, num_classes):
        super(EchoClassifier, self).__init__()
        self.encoder = image_encoder  # Use CLIP image encoder
        self.fc = nn.Linear(512, num_classes)  # This encoder's output dimension is indeed 512 dim

    def forward(self, x):
        #print("x in: ", x.shape) # x in:  torch.Size([1, 56, 3, 224, 224])
        #x = self.encoder(x)  # Extract image features
        x = torch.stack([echo_clip.encode_image(video) for video in x])
        x = F.normalize(x, dim=-1).to(torch.float32)
        # Add in a batch dimension because the zero-shot functions expect one
        #x = x.unsqueeze(0)
        #print("x encoded: ", x.shape)
        #exit(0) # x encoded:  torch.Size([1, 56, 512])
        x = self.fc(x)  # Classification head
        # torch.Size([1, 56, 2])
        x = x.mean(dim=1)  # Shape: [1, 2]
        #print(x.shape)
        x = torch.softmax(x, dim=1) # in range of prob
        return x

num_classes = 2  # Example: 4 for "normal", "mild", "moderate", "severe"
model = EchoClassifier(image_encoder, num_classes).to(device)

#Fine-tuning only the last few layers prevents overfitting:

#for param in model.encoder.parameters():
#    param.requires_grad = False  # Freeze all layers

# Unfreeze the last transformer block (ViT) or last few layers (ResNet)
#for param in list(model.encoder.parameters())[-5:]:
#    param.requires_grad = True

# prepare the dataset

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# Define Image Preprocessing
transform = preprocess_val # Use CLIP’s preprocessing

class EchoDataset(Dataset):
    def __init__(self, img_dir, labels, transform=None):
        self.img_dir = img_dir
        self.labels = labels  # Dictionary {filename: class_index}
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = list(self.labels.keys())[idx]
        img_path = os.path.join(self.img_dir, img_name)

        #image = Image.open(img_path).convert("RGB")
        label = self.labels[img_name]

        #if self.transform:
        #    image = self.transform(image)

        image = np.load(img_path)
        image = np.stack(image, axis=-1)
        if self.transform:
            # process the data, do normalization
            #print("img shape raw: ", image.shape)
            image = torch.stack(
                [self.transform(T.ToPILImage()(frame)) for frame in image], dim=0
            )
            #print("img shape preprocessed: ", image.shape)
            # # turn it into echo clip image encoding
            #image = image.to(device)
            image = image.to(torch.bfloat16)
            # image = F.normalize(echo_clip.encode_image(image), dim=-1)
            # # Add in a batch dimension because the zero-shot functions expect one
            # image = image.unsqueeze(0)

        return image, label

# Example usage
img_dir = data_path #"path_to_images"
#labels = {"image1.jpg": 0, "image2.jpg": 1, "image3.jpg": 2}  # Replace with actual labels

dataset = pd.read_csv(dataset_csv)
#print(len(dataset))
#dataset = dataset[dataset["split"] != "train"]
#print(len(dataset))
#print(dataset)

# for debug:
debug = True
if debug:
    dataset = dataset.head(5) #dataset[:100]

train_set = dataset[dataset["split"] == "train"]
val_set = dataset[dataset["split"] == "val"]
test_set = dataset[dataset["split"] == "test"]

if debug:
    val_set = train_set
    test_set = train_set

#path_list = list(dataset["path"]) #[os.path.join(data_path, p) for p in list(dataset["path"])]
#split_list = list(dataset["split"])
#label_list = list(dataset["label"])

train_labels = dict(zip(list(train_set["path"]), list(train_set["label"])))
val_labels = dict(zip(list(val_set["path"]), list(val_set["label"])))
test_labels = dict(zip(list(test_set["path"]), list(test_set["label"])))

train_dataset = EchoDataset(img_dir, train_labels, transform=transform)
val_dataset = EchoDataset(img_dir, val_labels, transform=transform)
test_dataset = EchoDataset(img_dir, test_labels, transform=transform)

batch_size = 1 #16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# define loss and optim

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)


# training
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    total_batches = len(dataloader)
    for batch_idx,(images, labels) in enumerate(dataloader):
        #print("images shape: ", images.shape)
        #exit(0)
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        #print(outputs.shape, labels.shape)
        #exit(0)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        print("batch ({}/{}): loss {}".format(batch_idx+1, total_batches, loss.item()))

    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, device):
    model.eval()
    #correct, total = 0, 0

    output = list()
    target = list()

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            #print(outputs.shape, labels.shape)
            #exit(0)
            labels = labels.unsqueeze(1)
            #_, preds = torch.max(outputs, 1)
            #correct += (preds == labels).sum().item()
            #total += labels.size(0)

            output += outputs.cpu().numpy()
            target += labels.cpu().numpy()


    output = torch.Tensor(np.array(output))
    target = torch.Tensor(np.array(target))

    label_dict = {0:0, 1:1}
    mode = "binary"

    #print(output, target, label_dict, mode)
    eval_stats = get_classification_metrics(output, target, label_dict, mode=mode)
    print(eval_stats)

    #return correct / total  # Accuracy

# run training
n_epochs = 1 if debug else 10
for epoch in range(n_epochs):  #10 # Adjust epochs
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_acc = evaluate(model, val_loader, device)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")


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


