import pandas as pd
from src.data_pipeline import RAW_DIR
from src.database import get_connection

conn = get_connection()

matches = pd.read_csv(RAW_DIR / "results.csv")
matches.to_sql("matches", conn, if_exists="replace", index=False)

former_names = pd.read_csv(RAW_DIR / "former_names.csv")
former_names.to_sql("former_names", conn, if_exists="replace", index=False)

conn.close()
print("Migration complete.")