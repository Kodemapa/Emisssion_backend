import sqlite3
import pandas as pd

# Database file
DB_FILE = "prediction.db"

# Transaction ID - using consistent global ID
transaction_id = "emission-analysis-2025"

# Create sample traffic data since the CSV file might not exist
sample_data = {
    'Traffic Speed': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
    'VehicleType': ['Passenger Car'] * 12,
    'City': ['Los Angeles'] * 12,
    'Year': [2024] * 12,
    'Tract': ['06037206020'] * 12,
    'Count': [100] * 12,
    'Distance': [1.0] * 12,
    'Time': [speed/60 for speed in [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]]  # Convert to hours
}

# Create DataFrame
df = pd.DataFrame(sample_data)

# Connect to DB
conn = sqlite3.connect(DB_FILE)
cur = conn.cursor()

# Clear existing data for this transaction_id
cur.execute("DELETE FROM traffic_volume_data WHERE TransactionID = ?", (transaction_id,))

# Insert data
for index, row in df.iterrows():
    cur.execute("""
        INSERT INTO traffic_volume_data 
        (TransactionID, Speed, VehicleType, City, Year, Tract, Count, Distance, Time) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        transaction_id, 
        row['Traffic Speed'], 
        row['VehicleType'],
        row['City'],
        row['Year'],
        row['Tract'],
        row['Count'],
        row['Distance'],
        row['Time']
    ))

# Commit and close
conn.commit()
conn.close()

print(f"Added {len(df)} records to traffic_volume_data with transaction_id: {transaction_id}")
print("Sample traffic speeds:", df['Traffic Speed'].tolist())
