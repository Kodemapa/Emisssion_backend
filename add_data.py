import sqlite3
import pandas as pd

# Database file
DB_FILE = "prediction.db"

# Transaction ID
transaction_id = "01b23513-7473-4418-988e-a933a7e11164"

# CSV file path
csv_file = r"d:\Prediction_models\inputdata\Traffic Volume and Speed\California\Processed CA Tracts_2024\tract=06037206020.csv"

# Read CSV
df = pd.read_csv(csv_file)

# Connect to DB
conn = sqlite3.connect(DB_FILE)
cur = conn.cursor()

# Insert data
for index, row in df.iterrows():
    speed = row['Traffic Speed']
    cur.execute("INSERT INTO traffic_volume_data (TransactionID, Speed) VALUES (?, ?)", (transaction_id, speed))

# Commit and close
conn.commit()
conn.close()

print(f"Added {len(df)} records to traffic_volume_data")
