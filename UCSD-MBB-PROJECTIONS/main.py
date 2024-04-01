import pandas as pd
import numpy as np
import argparse

from scraping.scraping_run import scraping_main
from train_inference.train import Train
from train_inference.run_model import RunModel

def main():

    parser = argparse.ArgumentParser(description="Run basketball data processing and modeling.")
    parser.add_argument('--scraping', type=bool, help='Boolean specifying whether or not web scraping needs to be performed')
    parser.add_argument('--train', type=bool, help='Boolean specifying whether or not the model needs to be trained')
    parser.add_argument('--player_name', type=str, help='Player to analyze')
    parser.add_argument('--player_position', type=str, help='Position of player being analyzed')
    parser.add_argument('--player_team', type=str, help='Team of the player being analyzed')
    parser.add_argument('--random_player', type=bool, help='Boolean specifying whether or not a random player is to be analyzed')
    parser.add_argument('--assess_predictions', type=bool, help='Boolean specifying whether or not to set aside a portion of the data for testing')


    # Parse the arguments
    args = parser.parse_args()

    scraping = args.scraping or None
    train = args.train or None
    player_name = args.player_name or None
    player_position = args.player_position or None
    player_team = args.player_team or None
    random_player = args.random_player or None
    assess_predictions = args.assess_predictions or None

    if scraping:
        scraping_main()
    if train:
        Train.train()
    
    RunModel.run_model(player_name=player_name, player_position=player_position, player_team=player_team, random_index=random_player, assess_predictions=assess_predictions)

if __name__ == "__main__":
    main()

       