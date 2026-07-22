import time
import pandas as pd
from src.wikipedia_scraper import fetch_tournament_squads
from src.database import get_connection

WORLD_CUP_YEARS = [1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022]
EURO_YEARS = [1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024]

tournaments = [
    (f"{year}_FIFA_World_Cup_squads", f"World Cup {year}") for year in WORLD_CUP_YEARS
] + [
    (f"UEFA_Euro_{year}_squads", f"Euro {year}") for year in EURO_YEARS
]

all_data = []
failed = []

for wiki_page, tournament_name in tournaments:
    print(f"Fetching {tournament_name}...")
    try:
        df = fetch_tournament_squads(wiki_page, tournament_name)
        all_data.append(df)
        print(f"  -> {len(df)} players, {df['country'].nunique()} countries")
    except Exception as e:
        print(f"  -> FAILED: {e}")
        failed.append((wiki_page, str(e)))
    time.sleep(2)  # courtesy delay between requests

combined = pd.concat(all_data, ignore_index=True)
print(f"\nTotal: {len(combined)} rows across {len(all_data)} tournaments")

if failed:
    print(f"\n{len(failed)} tournament(s) failed:")
    for page, err in failed:
        print(f"  {page}: {err}")

conn = get_connection()
combined.to_sql("squads", conn, if_exists="replace", index=False)
conn.close()
print("Saved to database table 'squads'.")