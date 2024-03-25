from scraping import BasketballScraper

def main():
    basketball_scraper = BasketballScraper()
    d2_schools_df = basketball_scraper.d2_schools
    d2_schools_list = list(d2_schools_df['Name'])
    df = basketball_scraper.data_all(d2_schools_list)
    print(df)

if __name__ == "__main__":
    main()

