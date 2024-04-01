import pandas as pd
import numpy as np

from scraping.scraping_helpers import BasketballScraper
from scraping.cleaning_helpers import CleaningHelpers

def scraping_main():
    basketball_scraper = BasketballScraper()
    d2_schools_df = basketball_scraper.d2_schools
    d2_schools_list = list(d2_schools_df['Name'])
    data = basketball_scraper.data_all(d2_schools_list)

    cleaning_helper = CleaningHelpers(data)
    cleaning_helper.change_column_scale()
    data = cleaning_helper.get_columns()
    cleaning_helper.fill_missing_values()
    cleaning_helper.save_to_csv("basketball_data.csv")

if __name__ == "__main__":
    scraping_main()


