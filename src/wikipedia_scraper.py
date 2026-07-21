import requests
from io import StringIO
from bs4 import BeautifulSoup
import pandas as pd

WIKI_HEADERS = {
    "User-Agent": "euro2028-prediction research project (student Jugend forscht entry; contact: coding@liamcheung.de)"
}


def fetch_tournament_squads(wiki_page: str) -> pd.DataFrame:
    """
    Scrape all national squad rosters from a Wikipedia "XXXX squads" page
    (e.g. "UEFA_Euro_2024_squads"). Returns one DataFrame with all
    countries' rosters combined, tagged with a `country` column.

    Roster tables are identified by content shape ("No." and "Pos."
    columns present), not by heading position or count — this survives
    tournaments with different team counts or a differently-placed stats
    section, unlike hardcoding "the first N headings".
    """
    url = f"https://en.wikipedia.org/wiki/{wiki_page}"
    resp = requests.get(url, headers=WIKI_HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    all_squads = []
    for heading in soup.find_all("h3"):
        country = heading.get_text(strip=True)
        table = heading.find_next("table")
        if table is None:
            continue

        header_text = table.get_text()
        if "No." not in header_text or "Pos." not in header_text:
            continue

        df = pd.read_html(StringIO(str(table)))[0]
        # shortcut for pandas to parse an html <table> element directly into a dataframe
        # which handles header rows and cell structure automatically 
        df["country"] = country
        all_squads.append(df)

    return pd.concat(all_squads, ignore_index=True)
    # stacks all countries' individual DataFrames into one combined table