import pandas as pd
import numpy as np

from utils.helper_jsons_scraping import conference_scores

class CleaningHelpers:
    def __init__(self, data):
        self.data = data
        self.d2_schools = self.get_d2_schools()
        self.conference_scores = self.get_conference_scores()
        self.selected_columns = [
            'Player', 'GP', 'GS', 'MIN/G', 'FG%', '3PT%', 'FT%',
            'OFF_REB/G', 'DEF_REB/G', 'REB/G', 'PPG', 'AST/G', 'TO/G',
            'PF/G', 'STL/G', 'BLK/G', 'Position', 'Team', 'Year',
            'Conference', 'Conference_Grade', 'Occurrence'
        ]

    def get_d2_schools(self):
        csv_file_path = 'data\\Scraping\\d2_basketball_schools.csv'
        data = pd.read_csv(csv_file_path)
        return data
    
    def get_conference_scores(self):
        return conference_scores

    def change_column_scale(self):
        self.data['PPG'] = self.data['PTS'] / self.data['GP']
        self.data['TO/G'] = self.data['TO'] / self.data['GP']
        self.data['PF/G'] = self.data['PF'] / self.data['GP']
        self.data['STL/G'] = self.data['STL'] / self.data['GP']
        self.data['BLK/G'] = self.data['BLK'] / self.data['GP']
        self.data['OFF_REB/G'] = self.data['OFF REB'] / self.data['GP']
        self.data['DEF_REB/G'] = self.data['DEF REB'] / self.data['GP']
        self.data['AST/G'] = self.data['AST'] / self.data['GP']

        self.data = self.data[['Player', 'GP', 'GS', 'MIN', 'MIN/G', 'FGM', 'FGA', 'FG%', '3PT',
       '3PTA', '3PT%', 'FT', 'FTA', 'FT%', 'PTS', 'AVG', 'OFF REB', 'DEF REB', 'OFF_REB/G', 'DEF_REB/G',
       'REB', 'REB/G', 'PF', 'AST', 'TO', 'STL', 'BLK', 'PPG', 'AST/G', 'TO/G', 'PF/G', 'STL/G', 'BLK/G', 'Position', 'Team',
       'Year', 'Conference', 'Conference_Grade']]
        self.data = self.data.fillna(0)

        # Position can be somewhat interchangeable with team here, though team can mess up with transfers
        self.data['Occurrence'] = self.data.groupby(['Player', 'Position']).cumcount() + 1
        
    def get_columns(self):
        return self.data[self.selected_columns]

    def fill_missing_values(self):
        self.data = self.data.fillna(0)

    def map_team_to_conference(self):
        team_conference_dict = pd.Series(self.d2_schools['Conference'].values, index=self.d2_schools['Name']).to_dict()
        return self.data['Team'].map(team_conference_dict)
    
    def map_conference_to_score(self):
        return self.data['Conference'].map(self.conference_scores).fillna(1)

    def clean_data(self):

        self.data['Conference'] = self.map_team_to_conference()
        self.data['Conference_Grade'] = self.map_conference_to_score()

        self.change_column_scale()

        #self.fill_missing_values()

        self.data = self.get_columns()


        return self.data

    def save_to_csv(self, filename):
        self.data.to_csv(filename)
