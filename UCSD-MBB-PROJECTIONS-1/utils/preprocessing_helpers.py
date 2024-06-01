import pandas as pd
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class PreprocessingHelpers:
    def __init__(self, data, assess_loss):
        self.data = data
        self.models = {}
        self.assess_loss = assess_loss
        self.features = self.get_features()
        self.numerical_cols = self.get_numerical_cols()
        self.categorical_features = self.get_categorical_features()
        self.preprocessor = self.get_preprocessor()
        self.batch_size = 64
        
    def drop_mostly_zero_rows(self):
        zero_counts = self.data.apply(lambda row: (row == 0).sum(), axis=1)
        self.data = self.data[zero_counts <= 10]
        self.data = self.data.sort_values(by='Year').reset_index(drop=True)
    
    def get_numerical_cols(self):
        return ['GP', 'GS', 'MIN/G', 'FG%', '3PT%', 'FT%', 'PPG', 'REB/G', 'OFF_REB/G', 'DEF_REB/G', 'PF/G', 'AST/G', 'TO/G', 'STL/G', 'BLK/G']
    
    def get_features(self):
        return ['Player', 'GP', 'GS', 'MIN/G', 'FG%', '3PT%', 'FT%', 'PPG', 'OFF_REB/G', 'DEF_REB/G', 'REB/G', 'AST/G', 'TO/G', 'PF/G', 'STL/G', 'BLK/G',
            'Position', 'Team', 'Conference', 'Conference_Grade', 'Occurrence']
    
    def get_categorical_features(self):
        return ['Player', 'Position', 'Team', 'Conference']
    
    def get_preprocessor(self):
        numerical_features = self.numerical_cols + ['Conference_Grade', 'Occurrence']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ])
        
        return preprocessor
    
    def preprocess(self, col):
        if self.assess_loss:
            self.assess_loss_true_preprocess(col)
        else:
            self.assess_loss_false_preprocess(col)
    
    def assess_loss_true_preprocess(self, col):

        self.data['Player'] = self.data['Player'].astype(str)
        self.data['Position'] = self.data['Position'].astype(str)
        self.data['Team'] = self.data['Team'].astype(str)
        self.data['Conference'] = self.data['Conference'].astype(str)
        
        last_year_range = self.data['Year'].max()

        train_data = self.data[self.data['Year'] != last_year_range]
        test_data = self.data[self.data['Year'] == last_year_range]

        X_train = train_data[self.features]
        y_train = train_data[col]
        X_test = test_data[self.features]
        y_test = test_data[col]

        self.preprocessor.fit(X_train)

        X_train_transformed = self.preprocessor.transform(X_train)
        X_test_transformed = self.preprocessor.transform(X_test)

        X_train_tensor = torch.tensor(X_train_transformed.toarray().astype(np.float32))
        y_train_tensor = torch.tensor(y_train.values.astype(np.float32))
        X_test_tensor = torch.tensor(X_test_transformed.toarray().astype(np.float32))
        y_test_tensor = torch.tensor(y_test.values.astype(np.float32))

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        return X_test_tensor, y_test_tensor, train_loader
    
    def assess_loss_false_preprocess(self, col):

        self.data['Player'] = self.data['Player'].astype(str)
        self.data['Position'] = self.data['Position'].astype(str)
        self.data['Team'] = self.data['Team'].astype(str)
        self.data['Conference'] = self.data['Conference'].astype(str)

        X = self.data[self.features]
        y = self.data[col]

        self.preprocessor.fit(X)

        X_transformed = self.preprocessor.transform(X)

        X_tensor = torch.tensor(X_transformed.toarray().astype(np.float32))
        y_tensor = torch.tensor(y.values.astype(np.float32))

        train_dataset = TensorDataset(X_tensor, y_tensor)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        return None, None, train_loader



    

