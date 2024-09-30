import os
import numpy as np


def rotate_point_cloud_with_normals(pc_normal):
        """
    Rotates point cloud and their normals around the y-axis.
    :param pc_normal: Array where first 3 columns are coordinates, next 3 are normals.
    :return: Rotated point cloud with normals.
    """
    rotation_angle = np.random.uniform() * 2 * np.pi #Random rotation angle 
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    #Split points and normals 
    points = pc_normal[:, :3]
    normals = pc_normal[:, 3:]
    #Apply rotation to points and normals 
    rotated_points = np.dot(points, rotation_matrix)
    rotated_normals = np.dot(normals, rotation_matrix)
    #Save rotated points and normals 
    rotated_pc_normal = np.concatenate((rotated_points, rotated_normals), axis=1)
    
    return rotated_pc_normal


def pc_normalize(pc):
       """
    Normalizes point cloud to unit sphere.
    :param pc: Numpy array of points.
    :return: Normalized point cloud.
    """
    centroid = np.mean(pc[:, :3], axis=0)
    pc[:, :3] -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(pc[:, :3] ** 2, axis=1)))
    pc[:, :3] /= furthest_distance
    return pc
def normalize_points(pc):
          """
    Normalizes point cloud to unit sphere.
    :param pc: Numpy array of points.
    :return: Normalized point cloud.
    """
    centroid = np.mean(pc[:, :3], axis=0)
    pc[:, :3] -= centroid
    max_distance = np.max(np.sqrt(np.sum(pc[:, :3]**2, axis=1)))
    pc[:, :3] /= max_distance
    return pc

def normalize_normals(pc):
    """
    Normalizes the normals
    :param pc: Point cloud data with normals.
    :return: Point cloud with normalized normals.
    """
    normals = pc[:, 3:]
    norms = np.linalg.norm(normals, axis=1).reshape(-1, 1)
    pc[:, 3:] = normals / norms
    return pc

def final_normalization(pc):
    pc = normalize_points(pc)
    pc = normalize_normals(pc)
    return pc

"""Shuffles point cloud to remove any structure"""
def shuffle_points(pc): 
    np.random.shuffle(pc)
    return pc
"""Adds random noise for each point to "jitter" the points"""
def jitter_point_cloud(pc, sigma=0.01, clip=0.05):
    N, C = pc.shape
    jittered_data = np.clip(sigma * np.random.randn(N, C), -clip, clip)
    pc += jittered_data
    return pc
"""Randomly shift all points in the point cloud"""
def shift_point_cloud(pc, shift_range=0.1):
    shifts = np.random.uniform(-shift_range, shift_range, (1, 3))
    pc[:, :3] += shifts
    return pc
"""Randomly scales the entire point cloud """
def random_scale_point_cloud(pc, scale_low=0.8, scale_high=1.25):
    scale = np.random.uniform(scale_low, scale_high)
    pc[:, :3] *= scale
    return pc
"""Randomly drops points from the pointcloud at given ratio"""
def random_point_dropout(pc, max_dropout_ratio=0.875):
    dropout_ratio = np.random.random() * max_dropout_ratio
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
    if len(drop_idx) > 0:
        pc[drop_idx, :] = pc[0, :]
    return pc

#Read pointcloud from file 
def read_point_cloud(file_path):
    return np.loadtxt(file_path, delimiter=',')
#Write pointcloud to file 
def write_point_cloud(file_path, point_cloud):
    np.savetxt(file_path, point_cloud, delimiter=',')
#Augments read pointcloud and saves to output 
def augment_and_save(input_folder, output_folder):
    if not os.path.exists(output_folder): #Falesafe
        os.makedirs(output_folder)
    #Apply all augmentations to read file 
    augment_functions = [
        rotate_point_cloud_with_normals,  
        jitter_point_cloud,
        shift_point_cloud,
        random_scale_point_cloud,
        random_point_dropout
    ]
    
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        pc = read_point_cloud(file_path)
        
        pc = pc_normalize(pc)  
        pc = shuffle_points(pc)
        
        for i, augment_func in enumerate(augment_functions, 1):
            augmented_pc = augment_func(pc.copy())
            augmented_pc = final_normalization(augmented_pc)
            
            augmented_file_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_augment_{i}.txt")
            write_point_cloud(augmented_file_path, augmented_pc)

# Paths  , input , output 
input_folder = r'D:\Tor_Phillip\Leveranse\Dataset\PseaudoEgetNorm\flowerpot'
output_folder = r'D:\Tor_Phillip\Leveranse\Dataset\PseaudoEgetNorm\flowerpot_aug'
augment_and_save(input_folder, output_folder)
