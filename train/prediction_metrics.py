import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class PredictionMetrics():
    def __init__(self, model, criterion, X_test_tensor, y_test_tensor):
        self.model = model
        self.criterion = criterion
        self.X_test_tensor = X_test_tensor
        self.y_test_tensor = y_test_tensor

    def get_test_loss(self):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self.X_test_tensor).squeeze()
            test_loss = self.criterion(predictions, self.y_test_tensor)
            print(f'Test Loss: {test_loss.item()}')