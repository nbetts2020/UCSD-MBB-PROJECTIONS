import pandas as pd
import numpy as np

class CleaningHelpers:
    def __init__(self, data):
        self.data = data
        self.selected_columns = [
            'Player', 'GP', 'GS', 'MIN/G', 'FG%', '3PT%', 'FT%',
            'OFF_REB/G', 'DEF_REB/G', 'REB/G', 'PPG', 'AST/G', 'TO/G',
            'PF/G', 'STL/G', 'BLK/G', 'Position', 'Team', 'Year',
            'Conference', 'Conference_Grade', 'Occurrence'
        ]

    def change_column_scale(self):
        self.data['PPG'] = self.data['PTS'] / self.data['GP']
        self.data['TO/G'] = self.data['TO'] / self.data['GP']
        self.data['PF/G'] = self.data['PF'] / self.data['GP']
        self.data['STL/G'] = self.data['STL'] / self.data['GP']
        self.data['BLK/G'] = self.data['BLK'] / self.data['GP']
        self.data['OFF_REB/G'] = self.data['OFF REB'] / self.data['GP']
        self.data['DEF_REB/G'] = self.data['DEF REB'] / self.data['GP']
        self.data['AST/G'] = self.data['AST'] / self.data['GP']

        # Position can be somewhat interchangeable with team here, though team can mess up with transfers
        self.data['Occurrence'] = self.data.groupby(['Player', 'Position']).cumcount() + 1
        
    def get_columns(self):
        return self.data[self.selected_columns]

    def fill_missing_values(self):
        self.data = self.data.fillna(0)

    def save_to_csv(self, filename):
        self.data.to_csv(filename)
