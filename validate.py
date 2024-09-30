import torch
import numpy as np
import os
import datetime
from ModelNet40 import ModelNet40
from hyperparameters import Hyperparameters
from eval import evaluate
from models.pointnet import pointnet2_cls_ssg, cls_loss
"""
1.Library and Module Imports:
    IMport libraries and modules  needed
    
2. Model Configuration and Loading:
   Loads a pre-trained PointNet++ model.
   - It uses command line arguments to pick model path

3. Device Configuration:
   Put model on  GPU or CPU. 

4. Dataset Preparation:
   Initializes a dataloader instance for the validation(final test) split of the dataset.
   Configures DataLoader for batch processing of validation data.

5. Evaluation Procedure:
   Uses evaulate function from eval.py .
   Calculates performance metrics.

6. Logging and Results:
   Validation results are logged into a timestamped text file.

7. Output:
   Outputs confirmation messasge , path to log file
"""




def main(model_path, data_root='./Data/ModelNet40_resampled'):
    #Set device to cuda , if not then cpu 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Load model with 6 input channels  and classes  
    model = pointnet2_cls_ssg(in_channels=6, nclasses=40) #In channels is 6 because we have 3D coordinates and 3D normals , hc

    #Load the weights from the trained model
    model.load_state_dict(torch.load(model_path, map_location=device))
    #Set model to evaluation mode 
    model.eval()
    #Load hyper parameters from hyperparamaters 
    hp = Hyperparameters()
    #Setup dataset for final testing and loader 
    val_dataset = ModelNet40(data_root=data_root, split='val', npoints=hp.npoints)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=hp.batch_size, shuffle=False, num_workers=4)


    #Define loss function to use for final testing 
    loss_func = cls_loss()   # Cross entropy loss
    #Evaluate model and get metrics , loss , acc ,f1 , auc 
    val_loss, val_acc, val_f1, val_auc = evaluate(model, val_loader, loss_func, device) 

    
    # Directory and file setup for logging of evaluation results
    eval_dir = 'evaluation'
    os.makedirs(eval_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file_path = os.path.join(eval_dir, f"validation_log_{timestamp}.txt")

    # Metrics to log
    metrics = f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}, AUC: {val_auc:.4f}"
    
    
    # Write validation metrics to log file 
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"Model validated: {model_path}\n")
        log_file.write(metrics)
    #Confirmation messasge of results being saved 
    print(f"Validation details saved to {log_file_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate a Model on ModelNet40")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--data_root', type=str, default='./Data/ModelNet40_resampled', help='Root directory of the ModelNet40 dataset.')
    args = parser.parse_args()

    main(args.model_path, args.data_root)
