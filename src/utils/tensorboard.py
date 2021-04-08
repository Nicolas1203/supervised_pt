"""Tensorboard utils functions
"""
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os


def get_writer(writer_name):
    """Initialize tensorboard summary writer

    Args:
        writer_name (str): Name of the experiment to writer in tensorboard
    """
    if len(writer_name) < 1:
            now = datetime.now()
            exp_name = f"runs/noname/{now.year}_{now.day}_{now.hour}"
            writer = SummaryWriter(exp_name)
    else:
        writer = SummaryWriter(os.path.join("runs", writer_name))
    
    return writer