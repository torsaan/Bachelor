import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from models.pointnet import cls_loss
from ModelNet40 import ModelNet40
from hyperparameters import Hyperparameters
"""
eval.py 

1. Imports Necessary Libraries and Modules: 

2. Defines the `evaluate` Function:
   Takes inputs  model, data loader, loss function, device, and number of classes.
   Sets model to evaluation mode to disable dropout and batch normalization. 
   Loops over the validation data, computing the output and loss for batches of input data.
   After accumulating predictions and true labels, it calculates F1 score and AUC. 

3. Model Setup:
   Loads a model from a specified path.
   Configures device (GPU or CPU).

4. Dataset and DataLoader Setup:
   Initializes ModelNet40 instance for the final test (validation) set.
   Configures DataLoader to load the data in batches.

5. Loss Function Configuration:
   Initializes the loss function (cls_loss) used during the training phase.

6. Evaluation:
    Calls the (evaluate) function with the selected model, data loader, etc.
    Outputs performance metrics loss, accuracy, F1 score, and AUC.

7. Result logging:
   Saves log of the evaluation,into  text file for later analysis.

"""

def evaluate(model, val_loader, loss_func, device, num_classes=40):
    #Set model to evaluation mode , disable dropout and batchnormalization updates 
    model.eval()
    total_seen, total_correct, losses = 0, 0, []
    all_preds = []
    all_labels = []


    #Disable gradient computation for efficiency 
    with torch.no_grad():
        #Iterate over the final test loader (validation/testing)
        progress = tqdm(enumerate(val_loader, 1), total=len(val_loader), desc="Validating") 
        for batch_idx, (data, labels) in progress:
            #Move labels to device ( cpu/gpu)
            labels = labels.to(device)
            xyz, points = data[:, :, :3], data[:, :, 3:]
            #Forward pass through the modell to get predictions , output 
            output = model(xyz.to(device), points.to(device))
            loss = loss_func(output, labels)
            #Check the predicted class 
            pred = torch.max(output, dim=-1)[1]
            #Accumulate correct predictions and total predictions 
            total_correct += torch.sum(pred == labels).item()
            total_seen += xyz.shape[0]
            losses.append(loss.item())

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            #Update progress bar 
            progress.set_postfix(loss=np.mean(losses), accuracy=total_correct / total_seen)

    # Compute F1 Score using scikit learn 
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Compute AUC
    # Convert labels and predictions to one-hot encoding
    # Compute auc using one-hot encoded labels and predictions 
    one_hot_labels = label_binarize(all_labels, classes=range(num_classes))
    one_hot_preds = label_binarize(all_preds, classes=range(num_classes))

    if num_classes == 2:
        auc = roc_auc_score(one_hot_labels, one_hot_preds[:, 1])
    else:
        try:
            auc = roc_auc_score(one_hot_labels, one_hot_preds, average="micro", multi_class="ovr")
        except ValueError as e:
            auc = float('nan') # Handle cases where auc is not calculated, this ressolved an earlier issue.
            print(f"Could not compute AUC: {e}")

    return np.mean(losses), total_correct / total_seen, f1, auc
