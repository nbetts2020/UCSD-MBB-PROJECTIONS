HARDCODED_MBKB_URL_SCHEMAS = {
    'Anderson University (South Carolina)': 'andersonsc', 'American International College': 'americanintl'
}

HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

CONFERENCE_SCORES = {
    "Pacific West Conference": 0.75,
    "Rocky Mountain Athletic Conference": 0.75,
    "Northeast-10 Conference": 0.75,
    "Southern Intercol. Ath. Conf.": 0.75,
    "South Atlantic Conference": 0.75,
    "Lone Star Conference": 1,
    "Great American Conference": 0.75,
    "Great Midwest Athletic Conference": 0.75,
    "Gulf South Conference": 1,
    "Peach Belt Conference": 0.75,
    "Northern Sun Intercollegiate Conference": 0.75,
    "Sunshine State Conference": 0.75,
    "Conference Carolinas": 0.75,
    "Atlantic Sun Conference": 1,
    "Central Atlantic Collegiate Conference": 0.75,
    "Pennsylvania State Athletic Conference": 0.75,
    "Central Intercollegiate Athletic Association": 0.75,
    "California Collegiate Athletic Association": 0.75,
    "Great Northwest Athletic Conference": 0.75,
    "East Coast Conference": 0.75,
    "Mountain East Conference": 0.75,
    "Great Lakes Intercollegiate Athletic Conference": 0.75,
    "Great Lakes Valley Conference": 0.75,
    "Mid-America Intercollegiate Athletics Association": 0.75,
    "Northeast Conference": 0.75,
    "Ohio Valley Conference": 0.75,
    "American Southwest Conference": 0.75,
    "Western Athletic Conference": 1.1,
    "Southland Conference": 0.75,
    "Big West Conference": 1,
    "Atlantic Coast Conference": 2,
    "Southeastern Conference": 1.9,
    "Big 12 Conference": 1.9,
    "Big Ten Conference": 1.9,
    "Pacific-12 Conference": 1.8,
    "Big East Conference": 1.8,
    "American Athletic Conference": 1.5,
    "The american athletic Conference": 1.5,
    "Mountain West Conference": 1.5,
    "Atlantic 10 Conference": 1.4,
    "Atlantic 10 conference": 1.4,
    "Altantic 10 Conference": 1.4,
    "Atlantic Ten Conference": 1.4,
    "West Coast Conference": 1.4,
    "Missouri Valley Conference": 1.3,
    "Misouri Valley Conference": 1.3,
    "Conference USA": 1.2,
    "Mid-American Conference": 1.1,
    "Sun Belt Conference": 1.1,
    "Sunbelt Conference": 1.1,
    "Southern Conference": 1,
    "Ivy League": 1,
    "Horizon League": 1,
    "Metro Atlantic Athletic Conference": 1,
    "Colonial Athletic Association": 1,
    "Northeast Conference": 1,
    "Ohio Valley Conference": 1,
    "Patriot League": 1,
    "Big Sky Conference": 1,
    "Big West Conference": 1,
    "Big South Conference": 1,
    "America East Conference": 1,
    "American East Conference": 1,
    "Atlantic Sun Conference": 1,
    "ASUN Conference": 1,
    "Asun Conference": 1,
    "Summit League": 1,
    "The Summit League": 1,
    "Summit League Conference": 1,
    "Southland Conference": 1,
    "Mid-Eastern Athletic Conference": 1,
    "Southwestern Athletic Conference": 1,
    "Division I-A Independents": 1
}

NUMERICAL_COLS = ['GP', 'GS', 'MIN/G', 'FG%', '3PT%', 'FT%', 'PPG', 'REB/G', 'OFF_REB/G', 'DEF_REB/G', 'PF/G', 'AST/G', 'TO/G', 'STL/G', 'BLK/G']

FEATURES = ['Player', 'GP', 'GS', 'MIN/G', 'FG%', '3PT%', 'FT%', 'PPG', 'OFF_REB/G', 'DEF_REB/G', 'REB/G', 'AST/G', 'TO/G', 'PF/G', 'STL/G', 'BLK/G',
            'Position', 'Team', 'Conference', 'Conference_Grade', 'Occurrence']

PDF_COLS = ['Number', 'Player', 'GP-GS', 'MIN', 'MIN/G', 'FGM-FGA', 'FG%', '3PT-3PTA', '3PT%', 'FT-FTA', 'FT%', 'OFF REB', 'DEF REB', 'REB', 'AVG', 'PF', 'DQ', 'AST', 'TO', 'BLK', 'STL', 'PTS', 'PPG']

APPEND_FORMAT = ['Player', 'GP', 'GS', 'MIN' ,'MIN/G', 'FGM', 'FGA', 'FG%', '3PT', '3PTA', '3PT%', 'FT', 'FTA', 'FT%', 'PTS', 'AVG', 'OFF REB', 'DEF REB', 'REB', 'REB/G', 'PF', 'AST', 'TO', 'STL', 'BLK']

INTERMEDIATE_COLS = ['Player', 'GP', 'GS', 'MIN', 'MIN/G', 'FGM', 'FGA', 'FG%', '3PT',
       '3PTA', '3PT%', 'FT', 'FTA', 'FT%', 'PTS', 'AVG', 'OFF REB', 'DEF REB', 'OFF_REB/G', 'DEF_REB/G',
       'REB', 'REB/G', 'PF', 'AST', 'TO', 'STL', 'BLK', 'PPG', 'AST/G', 'TO/G', 'PF/G', 'STL/G', 'BLK/G', 'Position', 'Team',
       'Year', 'Conference', 'Conference_Grade']
