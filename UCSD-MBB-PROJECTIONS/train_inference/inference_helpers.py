import pandas as pd
import numpy as np

import torch

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from train_inference.run_model import RunModel

class InferenceHelpers(RunModel):
    def __init__(self):
        super().__init__()

    def get_player_attributes(self, data, player_name, player_position, player_team, index=None):
        if index is not None:
            player_name, player_position, player_team, player_occurrence = data.iloc[index][['Player', 'Position', 'Team', 'Occurrence']]
        else:
            player_occurrence = self.train_data.loc[(self.train_data['Player'] == player_name) &
                                        (self.train_data['Position'] == player_position) &
                                        (self.train_data['Team'] == player_team), 'Occurrence'].iloc[0] - 1
        return player_name, player_position, player_team, player_occurrence
    
    def calculate_mse(self, row, dict_values):
          errors = [(row[col] - value) ** 2 for col, value in dict_values.items() if col in row]
          return sum(errors) / len(errors) if errors else 0
    
    def weighted_median(self, data, weights, percentile=50):
        """
        Calculate the weighted median of data given weights.
        Adjust the function to account for different percentiles to give more value to higher or lower values.

        :param data: Iterable of data points
        :param weights: Iterable of weights corresponding to data points
        :param percentile: The percentile to calculate the weighted median at (default is 50 for median)
        :return: The weighted percentile value
        """
        data, weights = np.array(data), np.array(weights)

        if len(data) == 2:
            if weights[0] == weights[1]:
                return data[0] if percentile <= 50 else data[1]
            elif percentile <= 50:
                return data[np.argmin(weights)]
            else:
                return data[np.argmax(weights)]

        sorted_indices = np.argsort(data)
        data_sorted = data[sorted_indices]
        weights_sorted = weights[sorted_indices]

        csum_weights = np.cumsum(weights_sorted)

        target = csum_weights[-1] * percentile / 100
        percentile_idx = np.where(csum_weights >= target)[0][0]

        return data_sorted[percentile_idx]
    
    def custom_median(df, column_name):
        """
        Calculate the median of a column without sorting the data.
        Assumes the DataFrame is already sorted by another column.
        
        Parameters:
        - df: pandas DataFrame.
        - column_name: the name of the column to calculate the median for.
        
        Returns:
        - The median value as per the current sorting of the DataFrame.
        """
        non_nan_values = df[column_name].dropna()
        
        n = len(non_nan_values)
        if n == 0:
            return np.nan
        elif n % 2 == 1:
            return non_nan_values.iloc[(n - 1) // 2]
        else:
            return (non_nan_values.iloc[n // 2 - 1] + non_nan_values.iloc[n // 2]) / 2.0
        
    def get_previous_year(self, player_name, player_position, player_team, player_occurrence):
        previous_year = self.train_data[(self.train_data['Player'] == player_name) &
                                (self.train_data['Position'] == player_position) &
                                (self.train_data['Team'] == player_team) &
                                (self.train_data['Occurrence'] == (player_occurrence - 1))].iloc[0:1]
            
        return previous_year

    def prepare_numerical_data(self, data_df, exclude_index, previous_year_index):
        all_numerical_data = data_df[self.numerical_cols]

        if exclude_index:
            all_numerical_data = all_numerical_data.drop(previous_year_index)

        return all_numerical_data
    
    def find_similar_players(self, all_numerical_data, previous_year):
        scaler = StandardScaler()
        scaled_all_data = scaler.fit_transform(all_numerical_data)
        scaled_player_data = scaler.transform(previous_year[self.numerical_cols])

        similarities = cosine_similarity(scaled_player_data, scaled_all_data)[0]
        top_indices = np.argsort(-similarities)[1:self.top_k_similar_players+1]

        similar_players = self.train_data.iloc[top_indices]
        next_year = pd.DataFrame(columns=similar_players.columns)

        similar_players_copy = similar_players.copy().reset_index(drop=True)
        for i in range(len(similar_players)):
            player_next_year = self.train_data[
                (self.train_data['Player'] == similar_players_copy.iloc[i]['Player']) &
                (self.train_data['Team'] == similar_players_copy.iloc[i]['Team']) &
                (self.train_data['Occurrence'] == similar_players_copy.iloc[i]['Occurrence'] + 1)
            ]

            if not player_next_year.empty:
                next_year = next_year.append(player_next_year, ignore_index=True)

        similar_players = next_year.median()

        if len(self.change_dict) > 0:

            next_year['MSE'] = next_year.apply(lambda row: self.calculate_mse(row, self.change_dict), axis=1)
            next_year = next_year.sort_values(by='MSE')
            next_year.drop(columns='MSE', inplace=True)

            z_scores = next_year[self.numerical_cols].apply(lambda x: (x - x.mean()) / x.std(), axis=0)
            next_year['outlier_score'] = z_scores.abs().sum(axis=1)

            threshold = 30

            next_year = next_year[next_year['outlier_score'] <= threshold].drop(columns=['outlier_score'])
            for column in self.numerical_cols:
                if column in list(self.change_dict.keys()):
                    similar_players[column] = self.change_dict[column]
                else:
                    similar_players[column] = self.custom_median(next_year[column])

        return similar_players
    
    def adjust_predictions_based_on_comparison(self, similar_players):

        scaler = StandardScaler()

        combined_df = pd.concat([similar_players.to_frame().T, player_comparisons[self.numerical_cols + ['Conference_Grade', 'Occurrence']]], ignore_index=True)
        scaled_combined_df = scaler.fit_transform(combined_df)
        scaled_df1 = scaled_combined_df[0:1, :]
        scaled_df2 = scaled_combined_df[1:, :]

        similarity_scores = cosine_similarity(scaled_df1, scaled_df2)
        similarity_scores = similarity_scores.flatten()

        temp_df = player_comparisons.copy()
        temp_df['similarity'] = similarity_scores

        sorted_df2 = temp_df.sort_values(by='similarity', ascending=False).drop(columns=['similarity'])

        player_comparisons = sorted_df2

        for column in similar_players.index:
            combined_list = [similar_players.loc[column]] + list(player_comparisons[column])
            new_val = InferenceHelpers.weighted_median(combined_list, list(range(len(player_comparisons) + 1))[::-1], 50)
            similar_players[column] = new_val

        return similar_players
    
    def similar_players_to_tensor(self, similar_players, previous_year):

        similar_players_df = similar_players.to_frame().transpose()
        similar_players_df.reset_index(drop=True, inplace=True)

        previous_year_copy = previous_year.copy().reset_index(drop=True)

        similar_players_df['Player'] = previous_year_copy.loc[0, 'Player']
        similar_players_df['Position'] = previous_year_copy.loc[0, 'Position']
        similar_players_df['Team'] = previous_year_copy.loc[0, 'Team']
        similar_players_df['Conference'] = previous_year_copy.loc[0, 'Conference']
        similar_players_df['Conference_Grade'] = previous_year_copy.loc[0, 'Conference_Grade']

        selected_features_transformed = self.preprocessor.transform(similar_players_df)

        selected_features_tensor = torch.tensor(selected_features_transformed.toarray().astype(np.float32))

        return selected_features_tensor
    
    def predict_output(self, model, selected_features_tensor, col, previous_year, assess_predictions, data_df, random_index_val):
        with torch.no_grad():
            prediction = model(selected_features_tensor).item()
            if prediction < 0:
                prediction = 0
            if col == "3PT%" and previous_year["3PT%"][0] == 0 and previous_year['GP'][0] > 15:
                prediction = 0
            if len(self.change_dict) > 0 and col in list(self.change_dict.keys()):
                print(f"Predicted (adjusted) '{col}' for the selected instance: {self.change_dict[col]}")
            else:
                print(f"Predicted '{col}' for the selected instance: {prediction}")

        if assess_predictions:
            actual_value = data_df.iloc[random_index_val][col]
            print(f"Actual '{col}' for the selected instance: {actual_value}")