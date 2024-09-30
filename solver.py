import numpy as np
import torch
import os
import datetime
import time
from tqdm import tqdm
from models.pointnet import cls_loss
from trainingstats import calculate_f1_score, calculate_auc_score
from trainingfunc import log_to_file, save_optimizer_state, load_optimizer_state
from trainingfunc import log_to_file, save_optimizer_state, load_optimizer_state, save_model_if_better, load_latest_model


'''

1. Library and Module Imports:

2.
train_one_epoch: for the training process for a single epoch, forward and backward passes, and updating model weights.
test_one_epoch: evaluates the model using the test dataset for one epoch and calculates performance metrics.
train: Uses train_one_epoch and test_one_epoch in a loop over multiple epochs of training and testing, controlls learning rate adjustments and checkpoints.


'''

def train_one_epoch(model, train_loader, optimizer, loss_func, device, log_file):
    """
    Train the model for one epoch.
    :param model: The neural network model to train.
    :param train_loader: DataLoader for the training dataset.
    :param optimizer: The optimizer used for training.
    :param loss_func: The loss function used for training.
    :param device: The device (CPU or GPU) that training is performed on.
    :param log_file: Path to the log file where training logs are saved.
    :return: Average loss and accuracy for the epoch.
    """
    model.train() #set model for training 
    total_seen, total_correct, losses = 0, 0, [] # Initialize values that will be used to track performance. 
    with tqdm(enumerate(train_loader, 1), total=len(train_loader), desc="Training") as progress:
        for batch_idx, (data, labels) in progress:
            optimizer.zero_grad()  # # Zero gradients to avoid accumulation
            labels = labels.to(device)#Labels to device 
            xyz, points = data[:, :, :3], data[:, :, 3:]#Split data into xyz and other features  ( normals)
        #Perform forward pass, move to device and compute
            output = model(xyz.to(device), points.to(device)) 
            loss = loss_func(output, labels) # Calculate loss based on model prediction 

            loss.backward()# Backpropagate the loss
            optimizer.step()# Update model parameters

            pred = torch.max(output, dim=-1)[1]
            total_correct += torch.sum(pred == labels).item() #Sum the correct predictions to total
            total_seen += xyz.shape[0] # Count total sampeled items
            losses.append(loss.item()) # Log the loss 
             # Update progress bar with current loss and accuracy
            progress.set_postfix(loss=np.mean(losses), accuracy=total_correct / total_seen)
            log_to_file(log_file, f"Batch: {batch_idx}, Loss: {np.mean(losses)}, Accuracy: {total_correct / total_seen}")

    return np.mean(losses), total_correct / total_seen

def test_one_epoch(model, test_loader, loss_func, device, log_file, num_classes=40):
    """
    Evaluate the model on the test dataset for one epoch.
    :param model: The neural network model to evaluate.
    :param test_loader: DataLoader for the test dataset.
    :param loss_func: The loss function used for evaluation.
    :param device: The device (CPU or GPU) that evaluation is performed on.
    :param log_file: Path to the log file where evaluation logs are saved.
    :param num_classes: Number of classes in the dataset (for calculating AUC).
    :return: Average loss, accuracy, F1 score, and AUC for the epoch.
    """
    model.eval() #Set model to evaluation
    total_seen, total_correct, losses = 0, 0, [] # For storing and counting for performance metrics 
    all_preds = [] # List for all predictions made
    all_labels = [] # List for all acutall labels 
    #Progress bar
    #
    with tqdm(enumerate(test_loader, 1), total=len(test_loader), desc="Testing") as progress:
        for batch_idx, (data, labels) in progress:
            labels = labels.to(device)#Move labels to device 
            xyz, points = data[:, :, :3], data[:, :, 3:]#Split data into xyz and other features  ( normals)
                #Forward pass , compute predictions 
            output = model(xyz.to(device), points.to(device))
            loss = loss_func(output, labels) #Calculate loss 

            pred = torch.max(output, dim=-1)[1]# Get the predicted class (highest output score).
            total_correct += torch.sum(pred == labels).item() # Count correct predictions 
            total_seen += xyz.shape[0] #Count total elements seen
            losses.append(loss.item()) # Record loss for batch 

            all_preds.extend(pred.cpu().numpy()) # Store predictions 
            all_labels.extend(labels.cpu().numpy()) #Store true labels 
            #Update progress bar  with metrics 
            progress.set_postfix(loss=np.mean(losses), accuracy=total_correct / total_seen)

    # Utilize the functions from trainingstats.py for metric calculations
    f1 = calculate_f1_score(all_labels, all_preds)
    auc = calculate_auc_score(all_labels, all_preds, num_classes)
    #Log results to file 
    log_message = f"Test Summary - Loss: {np.mean(losses):.4f}, Accuracy: {total_correct / total_seen:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}"
    log_to_file(log_file, log_message)

    return np.mean(losses), total_correct / total_seen, f1, auc
    


def train(train_loader, test_loader, model, loss_func, optimizer, scheduler, device, nepochs, log_dir, checkpoint_interval):
  log_file = os.path.join(log_dir, "training_log.txt")

  # Check if log directory exists
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  else:
    print(f"Directory {log_dir} already exists, continuing with existing directory.")

  checkpoint_dir = os.path.join(log_dir, 'checkpoints')
  # Create checkpoint directory if it doesn't exist
  if not os.path.exists(checkpoint_dir):
  	os.makedirs(checkpoint_dir, exist_ok=True)

  # Load the best model and hyperparameters (if they exist)
  best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
  best_auc_score = -float('inf')  # Use negative infinity for better initialization

  best_state_dict, best_optimizer_state = None, None

  if os.path.exists(best_model_path):
    # Load best model state and optimizer state
    best_state_dict = torch.load(best_model_path)
    best_optimizer_state = load_optimizer_state(checkpoint_dir)  # Add function to load optimizer state
    print("Loaded best model from checkpoint.")

  if best_state_dict:
    # Load model and optimizer states if they exist
    model.load_state_dict(best_state_dict)

    # If optimizer state exists, load it and update optimizer parameters after loading
    if best_optimizer_state:
      optimizer.load_state_dict(best_optimizer_state)
      for group in optimizer.param_groups:
        group['lr'] = scheduler.get_last_lr()[0]  # Update learning rate based on last epoch

  for epoch in range(nepochs):
    log_to_file(log_file, f"Epoch {epoch}/{nepochs}")

    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_func, device, log_file)
    test_loss, test_acc, f1, auc = test_one_epoch(model, test_loader, loss_func, device, log_file)

    scheduler.step()

       # Update best model based on AUC
    if not np.isnan(auc) and auc > best_auc_score:  # Check for valid and better AUC
        best_auc_score = auc
        torch.save(model.state_dict(), best_model_path)
        save_optimizer_state(optimizer, checkpoint_dir)
        log_to_file(log_file, f"Saved best model with AUC: {best_auc_score}")

    # Save checkpoint at regular intervals
    if epoch % checkpoint_interval == 0 or epoch == nepochs - 1:
      checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
      torch.save(model.state_dict(), checkpoint_path)
      log_to_file(log_file, f"Saved checkpoint to {checkpoint_path}" )
