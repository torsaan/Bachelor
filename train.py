import torch
from torch.utils.data import DataLoader
from models.pointnet import pointnet2_cls_ssg, pointnet2_cls_msg, cls_loss
import ModelNet40
from solver import train
from trainingfunc import load_latest_model
from hyperparameters import Hyperparameters
'''
Usage: 

1. Train.py:
Install  necessary dependencies. 

Download the  ModelNet40 dataset , place in the ./Data/ModelNet40_resampled directory. Dataset should be split into 'train','test' and 'val'.

Set hyperparameters in the hyperparameters.py file.

Run the script. "python train.py".

Script assumes the existence of certain files and directories (like the ModelNet40 dataset and the ./checkpoints directory).

'''


# Main 
if __name__ == '__main__':
    # Load hyperparameters from class Hyperparameters 
    hp = Hyperparameters()

    # Set up the device to use , GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare training and testing datasets using the ModelNet40 
    train_dataset = ModelNet40.ModelNet40(data_root='Data\ModelNet40_resampled', split='train', npoints=1024, augment=True)
    test_dataset = ModelNet40.ModelNet40(data_root='Data\ModelNet40_resampled', split='test', npoints=1024)

    # Set up data loaders for batching, shuffling and loading both datasets  , num workers(Set equal to cpu cores or higher) 
    train_loader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=hp.batch_size, shuffle=False, num_workers=4)

    # Check if model is trained with single- or multi- scale-grouping 
    if hp.model == 'pointnet2_cls_ssg':
        model = pointnet2_cls_ssg(6, hp.nclasses).to(device)
    elif hp.model == 'pointnet2_cls_msg':
        model = pointnet2_cls_msg(6, hp.nclasses).to(device)
    else:
        raise ValueError("Unsupported model type")

    # Load the latest model if it exists 
    model_save_path = hp.log_dir + '/checkpoints'
    model = load_latest_model(model, model_save_path)

    # Load loss function and optimizer, scheduler
    loss_func = cls_loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr, betas=(hp.beta1, hp.beta2), weight_decay=hp.decay_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hp.step_size, gamma=hp.gamma)

    # Start training process
    train(train_loader, test_loader, model, loss_func, optimizer, scheduler, device, hp.nepochs, hp.log_dir, hp.checkpoint_interval)
