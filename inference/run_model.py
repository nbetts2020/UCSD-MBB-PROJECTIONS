import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

class RunModel():
    def __init__(self, models, assess_predictions):
        self.models = models
        self.assess_predictions = assess_predictions
        self.data = self.get_data()
        self.last_year_range = self.data['Year'].max()
        self.train_data = self.get_train_data() if assess_predictions else None
        self.test_data = self.get_test_data() if assess_predictions else None

    def get_data(self):
        data = pd.read_csv("https://raw.githubusercontent.com/nbetts2020/UCSD-MBB-PROJECTIONS/main/data/Training/basketball_data.csv")
        return data

    def get_train_data(self):
        train_data = self.data[self.data['Year'] != self.last_year_range]
        return train_data

    def get_test_data(self):
        test_data = self.data[self.data['Year'] == self.last_year_range]
        return test_data
    
    def get_player_attributes(self, data, player_name, player_position, player_team, index=None):
        if index is not None:
            player_name, player_position, player_team, player_occurrence = data.iloc[index][['Player', 'Position', 'Team', 'Occurrence']]
        else:
            player_occurrence = self.train_data.loc[(self.train_data['Player'] == player_name) &
                                        (self.train_data['Position'] == player_position) &
                                        (self.train_data['Team'] == player_team), 'Occurrence'].iloc[0] - 1
        return player_name, player_position, player_team, player_occurrence
    
    
    def run_model(player_name, player_position, player_team, change_dict={}, player_comparisions=None, random_index=False):

        if self.assess_predictions:
            self.run_model_assess_predictions_true(player_name, player_position, player_team, change_dict={}, player_comparisions=None, random_index=True)
        else:
            self.run_model_assess_predictions_false(player_name, player_position, player_team, change_dict={}, player_comparisions=None, random_index=False)

    def run_model_assess_predictions_true(self, player_name, player_position, player_team, change_dict={}, player_comparisons=None, random_index=True):
        
        # Getting all players with 'Occurrence' (years of experience) greater than 1, so we have some past data to pass to the model
        test_data_experience_df = self.test_data[self.test_data['Occurrence'] > 1].reset_index(drop=True)

        if random_index:
            random_index_val = np.random.randint(0, len(test_data_experience_df))
            
        player_name, player_position, player_team, player_occurrence = self.get_player_attributes(test_data_experience_df, player_name, player_position, player_team, index=random_index_val)    
        
        for col, model in self.models.items():
            model.eval()
            previous_year = self.train_data[(self.train_data['Player'] == player_name) &
                                (self.train_data['Position'] == player_position) &
                                (self.train_data['Team'] == player_team) &
                                (self.train_data['Occurrence'] == (player_occurrence - 1))] # Note to self - changed this from .reset_index(drop=True), need to use .iloc[0:1] (I think) when retrieving

            


