"""File for metric implementation across evaluation tasks."""

import torch
#import monai
import numpy as np
from typing import Dict
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score, precision_score


# Metrics for classification
def get_classification_metrics(prediction: torch.Tensor, ground_truth: torch.Tensor, label_dict: dict, mode='binary', threshold=0.5) -> Dict[str, float]:

    labels = ground_truth.cpu().numpy() if ground_truth.is_cuda else ground_truth.numpy()
    prediction = prediction.cpu() if prediction.is_cuda else prediction
    prediction = prediction.float()

    metrics = {}
    # calculate accuracy
    if mode == 'binary':
        probs = torch.functional.F.softmax(torch.tensor(prediction), dim=1)[:, 1].numpy()
        prediction = torch.argmax(torch.tensor(prediction), dim=1).numpy()
        accuracy = (prediction == labels).sum().item() / len(labels)
        auroc, auprc = roc_auc_score(labels, probs), average_precision_score(labels, probs)
        bacc = balanced_accuracy_score(labels, prediction)
        f1 = f1_score(labels, prediction)
        probs_sort = np.sort(probs)[::-1]
        threshold_value = probs_sort[int(len(probs_sort) * threshold)]
        precision = precision_score(labels, probs >= threshold_value)

        metrics["Accuracy"] = accuracy
        metrics["BACC"] = bacc
        metrics["AUROC"] = auroc
        metrics["AUPRC"] = auprc
        metrics["F1 Score"] = f1
        metrics["Precision"] = precision
    else:
        # calculate the macro bacc in multilabel classification
        acc, bacc, f1, precision = 0, 0, 0, 0
        if mode == 'multiclass':
            probs = torch.functional.F.softmax(torch.tensor(prediction), dim=1).numpy()
            prediction = torch.argmax(torch.tensor(prediction), dim=1).numpy()
            prediction = torch.nn.functional.one_hot(torch.tensor(prediction), num_classes=len(label_dict)).numpy()

        elif mode == 'multilabel':
            probs = torch.sigmoid(torch.tensor(prediction)).numpy()
            prediction = (torch.sigmoid(torch.tensor(prediction)) > 0.5).numpy()
        else:
            raise ValueError(f"Unknown metric calculation mode {mode}")
        
        for idx in range(len(label_dict)):
            acc += (prediction[:, idx] == labels[:, idx]).sum().item() / len(labels)
            bacc += balanced_accuracy_score(labels[:, idx], prediction[:, idx])
            f1 += f1_score(labels[:, idx], prediction[:, idx])
            precision += precision_score(labels[:, idx], prediction[:, idx])
        acc /= len(label_dict)
        bacc /= len(label_dict)
        f1 /= len(label_dict)
        precision /= len(label_dict)
        # calculate micro AUROC 
        micro_auroc = roc_auc_score(labels, probs, average='micro') 
        # calculate macro AUROC
        macro_auroc = roc_auc_score(labels, probs, average='macro') 
        # calculate per class AUROC 
        per_class_auroc = roc_auc_score(labels, probs, average=None) 
        # calculate micro AUPRC 
        micro_auprc = average_precision_score(labels, probs, average='micro') 
        # calculate macro AUPRC 
        macro_auprc = average_precision_score(labels, probs, average='macro') 
        # calculate per class AUPRC 
        per_class_auprc = average_precision_score(labels, probs, average=None)
        
        metrics["Accuracy"] = acc
        metrics["BACC"] = bacc
        metrics["F1 Score"] = f1
        metrics["Precision"] = precision
        for idx, label in enumerate(label_dict):

            if len(label_dict) < 10:
                metrics[f"AUROC ({label})"] = per_class_auroc[idx]
                metrics[f"AUPRC ({label})"] = per_class_auprc[idx]
            
            metrics["AUROC"] = macro_auroc
            metrics["micro AUROC"] = micro_auroc
            metrics["AUPRC"] = macro_auprc
            metrics["micro AUPRC"] = micro_auprc
    
    return metrics

