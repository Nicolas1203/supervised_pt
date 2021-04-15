"""
Utils function for model saving, loading and others.
"""
import torch
import os


def save_model(model_weights, model_name, root_dir="./models/"):
    """Save PyTorch model weights

    Args:
        model_weights (Dict): model stat_dict
        model_name (str): name_of_the_model.pth
    """
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
        os.mkdir(root_dir + 'checkpoints/')
    
    torch.save(model_weights, os.path.join(root_dir, model_name))
