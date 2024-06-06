import pandas as pd
import numpy as np
import asyncio
import platform

import os

from utils.scraping_helpers import BasketballScraper
from utils.cleaning_helpers import CleaningHelpers

def scraping_main():
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    basketball_scraper = BasketballScraper()
    schools_df = basketball_scraper.d2_schools
    schools_list = list(schools_df['Name'])
    data = basketball_scraper.data_all(schools_list)

    cleaning_helper = CleaningHelpers(data)
    cleaning_helper.change_column_scale()
    data = cleaning_helper.get_columns()
    cleaning_helper.fill_missing_values()
    file_path = os.path.join('data', 'Training', 'basketball_data.csv')
    cleaning_helper.save_to_csv(file_path)

if __name__ == "__main__":
    scraping_main()


