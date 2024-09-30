import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from provider import pc_normalize, rotate_point_cloud_with_normal, rotate_perturbation_point_cloud_with_normal, random_scale_point_cloud, shift_point_cloud, jitter_point_cloud, random_point_dropout


class ModelNet40(Dataset):

    
#Initialize the class with the data root, split, number of points, augment, dp and normalize
    def __init__(self, data_root, split, npoints, augment=False, dp=False, normalize=True):

        """
        Initializes the dataset object.
        :param data_root: path to the dataset directory.
        :param split: dataset split, can be 'train', 'test', or 'val'.
        :param npoints: number of points to sample from each point cloud.
        :param augment: boolean, whether to apply data augmentation.
        :param dp: boolean, whether to apply dropout to points.
        :param normalize: boolean, whether to normalize point clouds.
        """
        assert(split in ['train', 'test', 'val'])  
        self.npoints = npoints
        self.augment = augment
        self.dp = dp
        self.normalize = normalize
        
        # Decode class names and indices for label mapping
        cls2name, name2cls = self.decode_classes(os.path.join(data_root, 'modelnet40_shape_names.txt'))
        self.num_classes = len(cls2name)  # Save the number of classes as an attribute here
        self.cls2name, self.name2cls = self.decode_classes(os.path.join(data_root, 'modelnet40_shape_names.txt'))


        # Define paths for different splits (train, test, val)
        cls2name, name2cls = self.decode_classes(os.path.join(data_root, 'modelnet40_shape_names.txt'))
        files_list_dict = {
            'train': os.path.join(data_root, 'modelnet40_train.txt'), #Path for train 
            'test': os.path.join(data_root, 'modelnet40_test.txt'), #Path for test 
            'val': os.path.join(data_root, 'modelnet40_val.txt')  # Path for val 
        }
         # Load the list of files and corresponding labels for the specified split
        self.files_list = self.read_list_file(files_list_dict[split], name2cls)
        self.caches = {}

#Read the list file and return the list of files and corresponding labels
    def read_list_file(self, file_path, name2cls):
        base = os.path.dirname(file_path)
        files_list = []
        class_counts = {name: 0 for name in name2cls}  # Initialize counts for each class

        with open(file_path, 'r') as f:
            for line in f.readlines():
            # Extract class name from the line and increment count
                name = '_'.join(line.strip().split('_')[:-1])
                
             # Construct file path based on data root and class name
                cur = os.path.join(base, name, '{}.txt'.format(line.strip()))
                
            # Map class name to corresponding label index
                files_list.append([cur, name2cls[name]])
                class_counts[name] += 1  # Increment count for this class

        # Print summary of classes and their counts
        for class_name, count in class_counts.items():
            print(f'{class_name}: {count} x ')
        
        return files_list


    #Decode classes from the file path 
    def decode_classes(self, file_path):
        cls2name, name2cls = {}, {}
        with open(file_path, 'r') as f:
            for i, name in enumerate(f.readlines()):
                
                # Map label index to class name
                cls2name[i] = name.strip()
                
                # Map class name to label index
                name2cls[name.strip()] = i
        return cls2name, name2cls

#Augment point clouds , rotate, scale, shift, jitter and dropout    
    def augment_pc(self, pc_normal):
        rotated_pc_normal = rotate_point_cloud_with_normal(pc_normal)
        rotated_pc_normal = rotate_perturbation_point_cloud_with_normal(rotated_pc_normal)
        jittered_pc = random_scale_point_cloud(rotated_pc_normal[:, :3])
        jittered_pc = shift_point_cloud(jittered_pc)
        jittered_pc = jitter_point_cloud(jittered_pc)
        rotated_pc_normal[:, :3] = jittered_pc
        return rotated_pc_normal
    
#Return the length of the files list
    def __getitem__(self, index):
         # Check if data for this index is already cached
        if index in self.caches:
            return self.caches[index]
        
        # Retrieve file path and label from the list
        file, label = self.files_list[index]
        
        # Load point cloud data from the file
        xyz_points = np.loadtxt(file, delimiter=',')
        
    # Select a subset of points if specified (npoints)
        xyz_points = xyz_points[:self.npoints, :]
        if self.normalize:
            xyz_points[:, :3] = pc_normalize(xyz_points[:, :3])
        if self.augment:
            xyz_points = self.augment_pc(xyz_points)
        if self.dp:
            xyz_points = random_point_dropout(xyz_points)
            
        # Cache the processed data and return
        self.caches[index] = [xyz_points, label]
        return xyz_points, label
#Return the length of the files list
    def __len__(self):
        return len(self.files_list)


'''
https://github.com/zhulf0804/Pointnet2.PyTorch/blob/master/data/ModelNet40.py
'''

    
