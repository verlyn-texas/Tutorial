import torch
import numpy as np
from torch.utils.data import Dataset

class shared_task_ds(Dataset):
  def __init__(self, data, targets):
      self.data = data
      self.targets = targets

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return (self.data[idx].unsqueeze(0),self.targets[idx])