import pandas as pd

import torch.nn as nn
import torch.optim as optim

from train_inference.preprocessing_helpers import PreprocessingHelpers
from train_inference.model import MLP
from train_inference.prediction_metrics import PredictionMetrics

class Train:
    def __init__(self):
        self.data = self.get_data()
        self.assess_loss = False
        self.learning_rate = 0.001
        self.num_epochs = 15
        self.models = {}

    def get_data(self):
        data = pd.read_csv("https://raw.githubusercontent.com/nbetts2020/UCSD-MBB-PROJECTIONS/main/data/Training/basketball_data.csv")
        return data
    
    def train(self):

        preprocessing_helper = PreprocessingHelpers(self.data, self.assess_loss)
        preprocessing_helper.drop_mostly_zero_rows()

        for col in preprocessing_helper.numerical_cols:

            X_test_tensor, y_test_tensor, train_loader = preprocessing_helper.preprocess(col)

            # Need to access something like train_loader[0].shape[1], but `DataLoader` objects don't support indexing, so need get it through iteration
            for batch in train_loader: 
                num_features = batch[0].shape[1]
                break

            model = MLP(input_size=num_features, output_size=1)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

            for epoch in range(self.num_epochs):
                model.train()
                running_loss = 0.0
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Column: {col}')
            
            self.models[col] = model

            if self.assess_loss:
                prediction_metric = PredictionMetrics(model=model, criterion=criterion, X_test_tensor=X_test_tensor, y_test_tensor=y_test_tensor)
                prediction_metric.get_test_loss()
        
        return self.models
        




    

    
