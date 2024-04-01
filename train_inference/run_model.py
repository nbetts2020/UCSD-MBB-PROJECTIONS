import pandas as pd
import numpy as np

from train_inference.preprocessing_helpers import PreprocessingHelpers
from train_inference.inference_helpers import InferenceHelpers

class RunModel(PreprocessingHelpers):
    def __init__(self, models, change_dict, player_comparisons, assess_predictions):
        super().__init__()
        self.models = models
        self.assess_predictions = assess_predictions
        self.data = self.get_data()
        self.last_year_range = self.data['Year'].max()
        self.train_data = self.get_train_data() if assess_predictions else None
        self.test_data = self.get_test_data() if assess_predictions else None
        self.change_dict = change_dict
        self.player_comparisons = player_comparisons
        self.top_k_similar_players = 150

    def get_data(self):
        data = pd.read_csv("https://raw.githubusercontent.com/nbetts2020/UCSD-MBB-PROJECTIONS/main/data/Training/basketball_data.csv")
        return data

    def get_train_data(self):
        train_data = self.data[self.data['Year'] != self.last_year_range]
        return train_data

    def get_test_data(self):
        test_data = self.data[self.data['Year'] == self.last_year_range]
        return test_data

    def run_model(self, player_name, player_position, player_team, random_index=True, assess_predictions=False):

        print("Processing data...")
        print(f"Player: {player_name}")
        print(f"Position: {player_position}")
        print(f"Team: {player_team}")

        data_df = self.test_data if assess_predictions else self.data
        data_experience_df = data_df[data_df['Occurrence'] > 1].reset_index(drop=True)

        random_index_val = np.random.randint(0, len(data_experience_df)) if random_index else None
        
        attributes = InferenceHelpers.get_player_attributes(data_experience_df, player_name, player_position, player_team, index=random_index_val)    
        player_name, player_position, player_team, player_occurrence = attributes

        for col, model in self.models.items():
            model.eval()
            previous_year = InferenceHelpers.get_previous_year(player_name, player_position, player_team, player_occurrence)
            
            all_numerical_data = InferenceHelpers.prepare_numerical_data(data_df, assess_predictions, previous_year.index)

            similar_players = InferenceHelpers.find_similar_players(previous_year, all_numerical_data, model)

            if self.player_comparisons is not None:
                similar_players = InferenceHelpers.adjust_predictions_based_on_comparison(similar_players, self.player_comparisons)
            
            selected_features_tensor = InferenceHelpers.similar_players_to_tensor(similar_players, previous_year)

            InferenceHelpers.predict_output(model, selected_features_tensor, col, previous_year, assess_predictions, data_df, random_index_val)