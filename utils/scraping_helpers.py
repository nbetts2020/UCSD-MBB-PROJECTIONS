import pandas as pd
import numpy as np

import requests
from bs4 import BeautifulSoup
import re
from langchain_community.tools import DuckDuckGoSearchResults
from datetime import datetime

import logging

import pdfplumber
from io import BytesIO
import gc
import os

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

from utils.helper_jsons_scraping import HARDCODED_MBKB_URL_SCHEMAS as mbkb_schemas
from utils.helper_jsons_scraping import HEADERS, CONFERENCE_SCORES, PDF_COLS, APPEND_FORMAT

class BasketballScraper():
    def __init__(self):
        self.driver = self.setup_chrome_driver()
        self.d2_schools = self.get_d2_schools()
        self.hardcoded_mbkb_url_schemas = self.get_hardcoded_mbkb_url_schemas()
        self.headers = self.get_headers()
        self.conference_scores = self.get_conference_scores()
        self.pdf_cols = self.get_pdf_cols()
        self.pdf_append_format_cols = self.get_pdf_append_format_cols()
    
    def setup_chrome_driver(self):
        """
        Sets up the ChromeDriver with specified options.
        """

        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        return driver
    
    def get_d2_schools(self):
        csv_file_path = os.path.join('data', 'Scraping', 'd1_d2_schools.csv')
        data = pd.read_csv(csv_file_path)
        return data
    
    def get_hardcoded_mbkb_url_schemas(self):
        return mbkb_schemas
    
    def get_headers(self):
        return HEADERS
    
    def get_conference_scores(self):
        return CONFERENCE_SCORES
    
    def get_pdf_cols(self):
        return PDF_COLS
    
    def get_pdf_append_format_cols(self):
        return APPEND_FORMAT
    
    def extract_website_prefix(self, url):
        com_index = url.find(".com")
        edu_index = url.find(".edu")

        if com_index != -1:
            prefix = url[:com_index + 4]
        elif edu_index != -1:
            prefix = url[:edu_index + 4]
        else:
            return "Not a .com or .edu URL"

        return prefix
    
    def get_athletics_website(self, team):
        search_string = f'{team} Basketball Athletics Website'
        search = DuckDuckGoSearchResults()
        search_results_string = search.run(search_string)
        cleaned_string = search_results_string.strip('[]')
        snippets = cleaned_string.split('], [')

        direct_roster_dict = {}
        for snippet in snippets:
            parts = snippet.rsplit(", link: ", 1)
            if len(parts) == 2:
                content, link = parts
                direct_roster_dict[link] = content

        website = self.extract_website_prefix(list(direct_roster_dict.keys())[0])
        return website
    
    def individual_overall_table(self, url):

        response = requests.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            section_id = 'individual-overall'

            section = soup.find(id=section_id)

            if section:
                section_html = str(section)
            else:
                print(f"Section with ID {section_id} not found.")
        else:
            print(f"Failed to fetch the webpage. Status code: {response.status_code}")

        player_names_stats = {}

        player_names_raw = section.find_all('a', class_='hide-on-medium-down')
        player_stats_raw = section.find_all('td', class_='text-center')

        for i in range(len(player_names_raw)):
            player_names_stats[player_names_raw[i].text] = {player['data-label']: player.text for player in player_stats_raw[(i*24):(24-i)+(i*25)]}

        return pd.DataFrame.from_dict(player_names_stats, orient='index')
    
    def extract_numbers(self, stat_string):
        numeric_values = stat_string.split('<td class="text-center">')
        cleaned_vals = [val.replace("\n", "").replace("</td>", "").replace("\t", "").replace("</tr>", "").replace("<tr>", "").replace("<td>", "").replace("Conference Only", "").replace('<td class="align-center">', "").strip() for val in numeric_values]
        cleaned_vals = [np.NaN if val == "-" else val for val in cleaned_vals if val != ""]

        return cleaned_vals
    
    def page_source_cleaned(self, page_source):
        page_source = page_source.split('<td colspan="29">')[1]
        stats_split = re.split(r'\.{2,}', page_source)
        stats_dict = {}
        for idx, stats in enumerate(stats_split):
            if idx % 2 == 0 and idx < len(stats_split)-3:
                name_split = stats.split(">")[-1]
                name = name_split.strip()
                if name != "Total":
                    stats_dict[name_split.strip()] = self.extract_numbers(stats_split[idx+1])

        return stats_dict
    
    def expand_range(self, lst, indices):
        result = []
        for i, item in enumerate(lst):
            if i in indices:
                start, end = item.split('-')
                result.extend([int(start), int(end)])
            else:
                result.append(item)
        return result
    
    def delete_elements(self, lst, indices_to_delete):
        n = len(lst)
        converted_indices = [i if i >= 0 else n + i for i in indices_to_delete]
        unique_indices = sorted(set(converted_indices), reverse=True)
        for index in unique_indices:
            del lst[index]
        return lst
    
    def sandwich_elements(self, lst):
        last_two = lst[-2:]
        without_last_two = lst[:-2]
        return without_last_two[:13] + last_two + without_last_two[13:]
    
    def apply_transformations_to_dict(self, dict_of_lists, indices_to_expand, indices_to_delete):
        transformed_dict = {}
        for key, lst in dict_of_lists.items():
            expanded_list = self.expand_range(lst, indices_to_expand)
            final_list = self.delete_elements(expanded_list, indices_to_delete)
            final_list = list(np.float_(self.sandwich_elements(final_list)))
            final_list[-1], final_list[-2] = final_list[-2], final_list[-1]
            transformed_dict[key] = final_list
        return transformed_dict
    
    def mbkb_individual_overall_table(self, url):
        page_source = None

        try:
            self.driver.get(url)
            self.driver.execute_script("window.print = function(){};")
            self.driver.implicitly_wait(10)

            page_source = self.driver.page_source

        finally:
            self.driver.quit()

        page_source_cleaned_dict = self.page_source_cleaned(page_source)
        for key, value in page_source_cleaned_dict.items():
            if len(value) == 1:
                split_values = value[0].split()
                processed_values = [np.nan if x == '-' else x for x in split_values]
                page_source_cleaned_dict[key] = processed_values
        page_source_cleaned_dict = self.apply_transformations_to_dict(page_source_cleaned_dict, [4,6,8], [-12,-10,-8,-7,-5,-3])
        df = pd.DataFrame.from_dict(page_source_cleaned_dict, orient='index').reset_index()
        df.columns = ["Player","GP","GS","MIN","MIN/G","FGM","FGA","FG%","3PT","3PTA","3PT%","FT","FTA","FT%","PTS","AVG","OFF REB","DEF REB","REB","REB/G","PF","AST","TO","STL","BLK"]
        return df
        
    def get_player_names(self, url):

        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            section_class = 'sidearm-roster-player-name'

            section = soup.find_all(class_=section_class)

            if section:
                section_html = str(section)
            else:
                print(f"Section with ID {section_class} not found.")
        else:
            print(f"Failed to fetch the webpage. Status code: {response.status_code}")

        bio_links = soup.find_all('a', {'aria-label': lambda L: L and 'View Full Bio' in L})

        player_names = [link['aria-label'].replace(' - View Full Bio', '') for link in bio_links]

        return list(dict.fromkeys(player_names))
    
    def get_mbkb_player_names(self, url):

        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            section_class = 'page-content roster-content'

            section = soup.find_all(class_=section_class)

            if section:
                section_html = str(section)
            else:
                print(f"Section with ID {section_class} not found.")
        else:
            print(f"Failed to fetch the webpage. Status code: {response.status_code}")

        bio_links = soup.find_all('a', {'aria-label': lambda L: L and 'full bio' in L})

        player_names = [link['aria-label'].replace(' - View Full Bio', '') for link in bio_links]
        player_names = [re.match("^(.*?):", player).group(1) for player in player_names]

        return list(dict.fromkeys(player_names))
    
    def other_get_player_info(self, url): # schools that use this type of html schema: ucla, tbd...

        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            section_class = 'c-rosterpage__content'

            section = soup.find_all(class_=section_class)

            if section:
                section = str(section)
            else:
                print(f"Section with ID {section_class} not found.")
        else:
            print(f"Failed to fetch the webpage. Status code: {response.status_code}")

        soup = BeautifulSoup(section, 'html.parser')

        player_cards = soup.find_all('div', class_='s-person-card')

        names = []
        positions = []
        years = []
        heights = []

        for card in player_cards:

            name_tag = card.find('h3')
            names.append(name_tag.text.strip() if name_tag else '')

            detail_wrapper = card.find('div', class_='s-person-details__bio-stats')
            details = detail_wrapper.find_all('span', class_='s-person-details__bio-stats-item') if detail_wrapper else []

            positions.append(details[0].text.strip().replace('Position ', '') if len(details) > 0 else '') # TODO here - think it's fixed though
            years.append(details[1].text.strip() if len(details) > 1 else '')
            heights.append(details[2].text.strip() if len(details) > 2 else '')

        df = pd.DataFrame({
            'Name': names,
            'Position': positions,
            'Academic Year': years,
            'Height': heights
        })

        return df
    
    def get_roster_hydrated(self, url): # no-js hydrated css type, used for clemson, tbd...

        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:

            soup = BeautifulSoup(response.text, 'html.parser')

            section_class = 'person__name'
            section_names = soup.find_all(class_=section_class)
            names = [sec.text.strip() for sec in section_names]

            section_class = 'person__meta'
            section_positions = soup.find_all(class_=section_class)
            positions_raw = [sec.text.strip() for sec in section_positions]
            positions = [item.split(' ')[1].replace("Guard", "G").replace("Forward", "F").replace("Center", "C") for item in positions_raw]

            df = pd.DataFrame({
                'Name': names,
                'Position': positions
            })

            return df
    
    def extract_position_before_space(self, pos):
        match = re.search(r'^([\w/]+)', pos)
        if match:
            return match.group(1)
        return pos
    
    def get_player_positions(self, url):

        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            section_class = 'sidearm-roster-player-position'

            section = soup.find_all(class_=section_class)

            if section:
                section_html = str(section)
            else:
                print(f"Section with ID {section_class} not found.")
        else:
            print(f"Failed to fetch the webpage. Status code: {response.status_code}")

        position_list = []
        for sect in section:
            position_list.append(sect.find('span', class_='text-bold').text.strip().replace('\r', '').replace('\n', '').replace('\t', ''))

        return [self.extract_position_before_space(pos).replace("Guard", "G").replace("Power Forward", "PF").replace("Forward", "F").replace("Foward","C").replace("Center", "C") for pos in position_list]
    
    def get_mbkb_player_positions(self, url):

        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            section_class = 'roster-data style-list'

            section = soup.find_all(class_=section_class)

            if section:
                section_html = str(section)
            else:
                print(f"Section with ID {section_class} not found.")
        else:
            print(f"Failed to fetch the webpage. Status code: {response.status_code}")

        position_list = []
        for sect in section:
            position_list.append(sect.find('span', class_="label font-weight-bold fw-bold d-md-none").text.strip().replace('\r', '').replace('\n', '').replace('\t', ''))

        return section
    
    def get_mbkb_player_positions(self, url):

        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            position_labels = soup.find_all('span', string='Pos.:')

            position_list = []
            for label in position_labels:
                position = label.next_sibling
                if position:
                    position_text = position.strip()
                    position_list.append(position_text)

            return position_list

        else:
            print(f"Failed to fetch the webpage. Status code: {response.status_code}")
            return None
    
    def find_pdf(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            button = soup.find('a', text='View Full Screen (PDF)')
            if not button:
                button = soup.find('a', {'aria-label': 'View Full Screen (PDF)'})

            if button and button.has_attr('href'):
                pdf_url = button['href']
                return pdf_url
            else:
                return
            
    def get_pdf_text(self, url):
        response = requests.get(url)

        if response.status_code == 200:
            with BytesIO(response.content) as pdf_file:
                with pdfplumber.open(pdf_file) as pdf:
                    text = ''
                    for page in pdf.pages:
                        text += page.extract_text()
                return text
        else:
            return
        
    def process_pdf(self, pdf_url):
        try:
            pdf_text = self.get_pdf_text(pdf_url)

            add_games = "GP-GS" + pdf_text.split("GP-GS")[1].split("Team")[0]
            s_after_first_newline = "\n".join(add_games.split("\n")[1:])

            lines = s_after_first_newline.strip().split('\n')
            rows = []
            for line in lines:
                parts = line.split()
                number_name = parts[0]
                rest_of_line = ' '.join(parts[1:])

                match_val = re.search(r'\d+-\d+', rest_of_line)

                if match_val:
                    idx = match_val.start()
                    name = rest_of_line[:idx].strip()
                    stats = rest_of_line[idx:].split()
                    row = [number_name] + [name] + stats
                    rows.append(row)

            df = pd.DataFrame(rows, columns=self.pdf_cols).iloc[:-2]
            print("ok")

            for column in df.columns:
                if '-' in column:
                    new_columns = column.split('-')
                    df[new_columns] = df[column].str.split('-', expand=True)
                    df[new_columns] = df[new_columns].apply(pd.to_numeric)
                    df.drop(columns=[column], inplace=True)
            
            print("ok2")

            # exclude_cols = ['Player', 'MIN']
            # include_cols = df.columns.difference(exclude_cols)

            # df[include_cols] = df[include_cols].astype(float)

            exclude_cols = ['Player', 'MIN', 'GP-GS']
            include_cols = df.columns.difference(exclude_cols)

            print(df)

            try:
                df[include_cols] = df[include_cols].astype(float)
                print("Conversion to float successful.")
            except Exception as e:
                print(f"An error occurred during conversion: {e}")

            print("ok6")

            df['REB/G'] = df['REB'] / df['GP']
            df['PPG'] = df['PTS'] / df['GP']
            df['TO/G'] = df['TO'] / df['GP']
            df['PF/G'] = df['PF'] / df['GP']
            df['STL/G'] = df['STL'] / df['GP']
            df['BLK/G'] = df['BLK'] / df['GP']
            df['OFF_REB/G'] = df['OFF REB'] / df['GP']
            df['DEF_REB/G'] = df['DEF REB'] / df['GP']
            df['AST/G'] = df['AST'] / df['GP']

            df = df[self.pdf_append_format_cols]
            print("ok4")

            return df

        finally:
            gc.collect()
        
    def data_table(self, individual_url, roster_url, type_of_data=None):
        if type_of_data == "normal":
            df = self.individual_overall_table(individual_url)
            df.index = df.index.map(lambda x: ' '.join(x.split(', ')[::-1]))

            positions = self.get_player_positions(roster_url)
            players = self.get_player_names(roster_url)[:len(positions)]
            player_position_dict = dict(zip(players, positions))

            positions_df = pd.DataFrame(list(player_position_dict.items()), columns=['Player', 'Position'])
            positions_df.set_index('Player', inplace=True)

            df = df.merge(positions_df, left_index=True, right_index=True, how='inner')

            df.fillna(0, inplace=True)
        elif type_of_data == "mbkb":
            df = self.mbkb_individual_overall_table(individual_url)
            df = df.set_index('Player', drop=True)
            positions = self.get_mbkb_player_positions(roster_url)
            players = self.get_mbkb_player_names(roster_url)
            player_position_dict = dict(zip(players, positions))
            if len(player_position_dict) == 0:
                df['Position'] = np.NaN
            else:
                positions_df = pd.DataFrame(list(player_position_dict.items()), columns=['Player', 'Position'])
                positions_df.set_index('Player', inplace=True)

                df = df.merge(positions_df, left_index=True, right_index=True, how='inner')

                df.fillna(0, inplace=True)
        return df
    
    def validate_website(self, url, url_type, check_type='abbr'): # validating website - schools insert the year of data a few different ways, either something like '2023-2024', '2023-24', or '2023'
        try:
            response = requests.get(url, headers=self.headers, allow_redirects=True)
            final_url = response.url

            if response.status_code == 200:
                return final_url

            if response.history or response.status_code in range(300, 405):
                if url_type == "roster":
                    if check_type == 'abbr':
                        new_url = url[:-4] + url[-2:]
                        return self.validate_website(new_url, url_type, 'year')
                    elif check_type == 'year':
                        new_url = url[:-7] + '20' + url[-2:]
                        return self.validate_website(new_url, url_type, None)
                elif url_type == "individual_overall":
                    if check_type == 'abbr':
                        new_url = url[:-15] + url[-13:]
                        return self.validate_website(new_url, url_type, 'year')
                    elif check_type == 'year':
                        new_url = url[:-18] + '20' + url[-13:]
                        return self.validate_website(new_url, url_type, None)
                elif url_type == "pdf":
                    if check_type == 'abbr':
                        new_url = url[:-2] + '20' + url[-2:]
                        return self.validate_website(new_url, url_type, 'year')
                    elif check_type == 'year':
                        new_url = url[:-5]
                        return self.validate_website(new_url, url_type, None)
                elif url_type == "pdf_hydrated":
                    if check_type == 'full':
                        new_url = url[:-12] + '20' + url[-12:]
                        return self.validate_website(new_url, url_type, 'year')
                    elif check_type == 'year':
                        new_url = url[:-15] + url[-10:]
                        return self.validate_website(new_url, url_type, None)
            return 'Website not found'
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None
    
    def team_data(self, team):
        df = pd.DataFrame()
        website = self.get_athletics_website(team)

        start_year = 2018
        current_year = datetime.now().year
        current_month_day = datetime.now().month, datetime.now().day
        cutoff_month_day = (3, 18)

        end_year = current_year if current_month_day >= cutoff_month_day else current_year - 1
        years = [f"{year}-{year+1}" for year in range(start_year, end_year)]

        original_suffix = '/sports/mens-basketball/'
        roster_suffix = original_suffix + 'roster/'
        individual_overall_suffix = original_suffix + 'stats/xxxYEARxxx#individual'
        modified_suffix = '/sports/mbkb/'
        modified_roster_suffix = modified_suffix + 'xxxYEARxxx/roster'
        pdf_suffix = "/sports/mens-basketball/stats/xxxYEARxxx/pdf"
        pdf_hydrated_suffix = f"{website[:8]}data.{website[8:]}/pdf/m_basketball/xxxYEARxxx/Stats.pdf"

        def get_conference_info(team, new_df):
            new_df['Team'] = team
            new_df['Year'] = year
            new_df['Conference'] = self.d2_schools.loc[self.d2_schools['Name'] == team, 'Conference'].values[0]
            new_df['Conference_Grade'] = new_df['Conference'].map(self.conference_scores).fillna(1)
            return new_df

        for year in years:
            try:
                individual_overall_website = self.validate_website(website + individual_overall_suffix.replace("xxxYEARxxx", year), 'individual_overall')
                roster_website = self.validate_website(website + roster_suffix + year, 'roster')

                if not individual_overall_website or not roster_website:
                    raise ValueError("Initial URLs failed validation")

                new_df = self.data_table(individual_overall_website, roster_website, "normal")
                new_df = get_conference_info(team, new_df)
                df = df.append(new_df)

                print(team, year, "Success - Initial URLs")
            except Exception as initial_error: 
                try: # mbkb case
                    hit = False
                    team_name = website.split(".com")[0].replace("https://www.", "")
                    remove_mascot_search_list = [team_name[:i + 1] for i in range(len(team_name))]

                    for mascot in remove_mascot_search_list:
                        modified_individual_overall_suffix = website + modified_suffix + f'{year[:-4] + year[-2:]}/teams/' + mascot + '?tmpl=teaminfo-network-monospace-template&sort=ptspg'
                        if requests.get(modified_individual_overall_suffix, headers=self.headers).status_code == 200:
                            hit = True
                            break

                    if not hit:
                        modified_individual_overall_suffix = website + modified_suffix + f'{year[:-4] + year[-2:]}/teams/' + self.hardcoded_mbkb_url_schemas[team] + '?tmpl=teaminfo-network-monospace-template&sort=ptspg'

                    individual_overall_website = modified_individual_overall_suffix
                    roster_website = self.validate_website(website + modified_roster_suffix.replace("xxxYEARxxx", year[:-4] + year[-2:]), 'roster')
                    if not individual_overall_website or not roster_website:
                        raise ValueError("Modified URLs failed validation")

                    new_df = self.data_table(individual_overall_website, roster_website, type_of_data="mbkb")
                    new_df = get_conference_info(team, new_df)
                    df = df.append(new_df)

                    print(team, year, "Success - Modified URLs")
                except Exception as modified_error:
                    try: # pdf case
                        pdf_site = self.validate_website(website + pdf_suffix.replace("xxxYEARxxx", year[:4]), 'pdf')
                        pdf_cloud_link = self.find_pdf(pdf_site)
                        pdf_df = self.process_pdf(pdf_cloud_link)

                        roster_website = self.validate_website(website + roster_suffix + year, "roster")
                        pdf_roster = self.other_get_player_info(roster_website)

                        # pdf_df 'Player' column's last names are all caps, so accounting for case when last name may have multiple capital letters
                        pdf_df['Player'] = pdf_df['Player'].map(lambda x: ' '.join([x.split(', ')[1], x.split(', ')[0].title()]))
                        pdf_df['Player_lower'] = pdf_df['Player'].str.lower()
                        pdf_roster['Name_lower'] = pdf_roster['Name'].str.lower() # lowercasing the name column temporarily then reverting back to pdf_df's 'Player' column once merged
                        
                        df = pdf_df.merge(pdf_roster, left_on='Player_lower', right_on='Name_lower', how='inner')
                        df.drop(columns=['Player_lower', 'Name_lower', 'Academic Year', 'Height'], inplace=True)
                        df.fillna(0, inplace=True)

                        print(team, year, "Success - PDF URLs")
                    except Exception as pdf_error:
                        try: # pdf hydrated case
                            roster_suffix = '/sports/mens-basketball/roster/season/xxxYEARxxx/'
                            pdf_site = self.validate_website(pdf_hydrated_suffix.replace("xxxYEARxxx", year[:-4] + year[-2:]), 'pdf_hydrated', 'full')
                            print(pdf_site)
                            pdf_df = self.process_pdf(pdf_site)
                            print(website + roster_suffix.replace("xxxYEARxxx", year[:4]))
                            roster_website = self.validate_website(website + roster_suffix.replace("xxxYEARxxx", year[:4]), "roster")
                            pdf_roster = self.get_roster_hydrated(roster_website)

                            pdf_df['Player'] = pdf_df['Player'].map(lambda x: ' '.join([x.split(', ')[1], x.split(', ')[0].title()]))
                            pdf_df['Player_lower'] = pdf_df['Player'].str.lower()
                            pdf_roster['Name_lower'] = pdf_roster['Name'].str.lower()

                            df = pdf_df.merge(pdf_roster, left_on='Player_lower', right_on='Name_lower', how='inner')
                            df.drop(columns=['Player_lower', 'Name_lower'], inplace=True)
                            df.fillna(0, inplace=True)

                            print(team, year, "Success - Hydrated PDF URLs")
                        except Exception as outer_error:
                            print(f"Failed to process team, year {team, year} with all URL formats: {initial_error}; {modified_error}; {pdf_error}; {outer_error}")

        return df

    def data_all(self, team_list):
        df = pd.DataFrame()
        for team in team_list:
            try:
                new_df = self.team_data(team)
                df = df.append(new_df)
            except Exception as e:
                print(f"Failed to process team {team}: {e}")
                continue
        return df
            
        
                    

                

                
                
