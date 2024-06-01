import pandas as pd
import numpy as np
import argparse

import torch
import os

from scraping.scraping_run import scraping_main
from train_inference.train import Train
from train_inference.run_model import RunModel
from utils.cleaning_helpers import CleaningHelpers

from train_inference.model import MLP

def get_models(input_size, output_size):
    models_dir = "models"
    model_files = {
        f.replace(".pt", "").replace("-", "/"): torch.load(f"{models_dir}/{f}", map_location=torch.device('cpu'))
        for f in os.listdir(models_dir) if f.endswith('.pt')
    }

    models = {
        name: MLP(input_size, output_size)
        for name in model_files
    }
    
    for name, state_dict in model_files.items():
        models[name].load_state_dict(state_dict)
        models[name].eval()

    return models


def main():

    parser = argparse.ArgumentParser(description="Run basketball data processing and modeling.")
    parser.add_argument('--scraping', type=bool, help='Boolean specifying whether or not web scraping needs to be performed')
    parser.add_argument('--train', type=bool, help='Boolean specifying whether or not the model needs to be trained')
    parser.add_argument('--player_name', type=str, help='Player to analyze')
    parser.add_argument('--player_position', type=str, help='Position of player being analyzed')
    parser.add_argument('--player_team', type=str, help='Team of the player being analyzed')
    parser.add_argument('--change_gp', type=int, help="change_dict 'GP' portion")
    parser.add_argument('--change_gs', type=int, help="change_dict 'GS' portion")
    parser.add_argument('--change_min_g', type=int, help="change_dict 'MIN/G' portion")
    parser.add_argument('--player_comparison_names', type=list, help='List of the players to incorporate into the prediction (i.e. fill a similar role)')
    parser.add_argument('--player_comparison_positions', type=list, help="List of the players' positions to incorporate into the prediction (i.e. fill a similar role)")
    parser.add_argument('--player_comparison_teams', type=list, help="List of the players' teams to incorporate into the prediction (i.e. fill a similar role)")
    parser.add_argument('--random_player', type=bool, help='Boolean specifying whether or not a random player is to be analyzed')
    parser.add_argument('--assess_predictions', type=bool, help='Boolean specifying whether or not to set aside a portion of the data for testing')

    # Parse the arguments
    args = parser.parse_args()

    scraping = args.scraping or None
    train = args.train or None
    player_name = args.player_name or None
    player_position = args.player_position or None
    player_team = args.player_team or None
    change_gp = args.change_gp or None
    change_gs = args.change_gs or None
    change_min_g = args.change_min_g or None
    player_comparison_names = args.player_comparison_names or None
    player_comparison_positions = args.player_comparison_positions or None
    player_comparison_teams = args.player_comparison_teams or None
    random_player = args.random_player or None
    assess_predictions = args.assess_predictions or None

    if scraping:
        scraping_main()
    if train:
        train_instance = Train()
        train_instance.train()

    models = get_models(input_size=10178, output_size=1)
    
    change_dict = {k: v for k, v in [('GP', change_gp), ('GS', change_gs), ('MIN/G', change_min_g)] if v is not None}

    player_comparisons = list(zip(player_comparison_names, player_comparison_teams, player_comparison_positions)) if player_comparison_names is not None else None

    run_model_instance = RunModel(models=models, change_dict=change_dict, player_comparisons=player_comparisons, assess_predictions=assess_predictions)
    run_model_instance.run_model(player_name=player_name, player_position=player_position, player_team=player_team, random_index=random_player, assess_predictions=assess_predictions)

if __name__ == "__main__":
    main()

       