from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import io, os
import json
import uuid
from models import ModifiedNet
import random

# ---------- Allow safe sklearn globals for torch.load ----------
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
import sklearn.compose._column_transformer

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([
        ColumnTransformer,
        sklearn.compose._column_transformer.ColumnTransformer,
        StandardScaler,
        OneHotEncoder,
        PowerTransformer
    ])

# ---------- Flask ----------
app = Flask(__name__)
CORS(app)

DB_FILE = "prediction.db"

# ---------- Global Transaction ID ----------
# Use a single consistent transaction ID for all operations
GLOBAL_TRANSACTION_ID = "emission-analysis-2025"

# ---------- Init DB ----------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    cur.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )""")

    cur.execute("""CREATE TABLE IF NOT EXISTS vehicle_classification (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        city TEXT,
        vehicle_type TEXT,
        fuel_type TEXT,
        year INTEGER,
        penetration REAL
    )""")

    cur.execute("""CREATE TABLE IF NOT EXISTS penetration_rate (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        city TEXT,
        vehicle_type TEXT,
        fuel_type TEXT,
        year INTEGER,
        penetration REAL
    )""")

    cur.execute("""CREATE TABLE IF NOT EXISTS traffic_volume (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tract TEXT,
        year INTEGER,
        volume REAL
    )""")

    cur.execute("""CREATE TABLE IF NOT EXISTS projected_traffic (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tract TEXT,
        year INTEGER,
        volume REAL
    )""")

    cur.execute("""CREATE TABLE IF NOT EXISTS mfd_params (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tract TEXT,
        param_key TEXT,
        param_value REAL
    )""")

    cur.execute("""CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        city TEXT,
        year INTEGER,
        vehicle_type TEXT,
        fuel_type TEXT,
        co2 REAL,
        nox REAL,
        energy REAL
    )""")

    cur.execute("""CREATE TABLE IF NOT EXISTS vehicle_classification_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id TEXT,
        city TEXT,
        vehicle_type TEXT,
        BaseYear INTEGER,
        data_city TEXT,
        data_vehicle_type TEXT,
        fuel_type TEXT,
        age REAL,
        vehicle_count INTEGER,
        user_id INTEGER
    )""")

    cur.execute("""CREATE TABLE IF NOT EXISTS traffic_volume_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        TransactionID TEXT,
        City TEXT,
        Year INTEGER,
        Tract TEXT,
        VehicleType TEXT,
        Count INTEGER,
        Distance REAL,
        Time REAL,
        Speed REAL
    )""")
    
    # Master table: projected_traffic_volume
    cur.execute("""CREATE TABLE IF NOT EXISTS projected_traffic_volume (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id TEXT,
        city_name TEXT,
        year INTEGER,
        tract TEXT,
    volume REAL
    )""")

    cur.execute("""CREATE TABLE IF NOT EXISTS projected_traffic_details (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id TEXT,
        time TEXT,
        datetime TEXT,
        traffic_volume REAL,
        traffic_speed REAL,
        adjusted_traffic REAL,
        density REAL,
        speed REAL
    )""")

    # default admin user
    cur.execute("INSERT OR IGNORE INTO users (username,password) VALUES (?,?)", ("admin","admin"))
    conn.commit()
    conn.close()

init_db()

# ---------- Model ----------
class MLPRegressor(nn.Module):
    def __init__(self, in_features: int, hidden=(32,16)):
        super().__init__()
        layers, last = [], in_features
        for h in hidden:
            layers += [nn.Linear(last,h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last,1)]
        self.net = nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

# ---------- Categories ----------
VEHICLE_CATS = [
    "Combination Long-haul Truck","Combination Short-haul Truck","Light Commercial Truck","Single Unit Short-haul Truck",
    "Motorhome - Recreational Vehicle","Motorcycle","Other Buses","Passenger Car","Passenger Truck",
    "Refuse Truck","School Bus","Single Unit Long-haul Truck","Single Unit short-haul Truck","Transit Bus"
]
FUEL_CATS = ["Gasoline","Diesel Fuel","Electricity","Compressed Natural Gas - CNG","Ethanol - E-85"]
STATE_CATS = ["CA","GA","NY","WA"]
CITY_TO_STATE = {
    "Atlanta": "GA",
    "Los Angeles": "CA", 
    "NewYork": "NY",
    "Seattle": "WA"
}

# ---------- Stats ----------
STATS_AS = {"Age":{"mean":5,"std":3},"Speed":{"mean":60,"std":15}}
STATS_WS = {"Vehicle Weight":{"mean":1500,"std":300},"Speed":{"mean":60,"std":15}}
STATS_SG = {"Speed":{"mean":60,"std":15},"Road Gradient":{"mean":5,"std":3}}

def _z(val, mean, std): return (float(val)-float(mean))/(std if std else 1.0)

def preprocess(payload, numeric_stats):
    num_feats = [_z(payload[k], s["mean"], s["std"]) for k,s in numeric_stats.items()]
    v_vec = [1.0 if payload["Vehicle Type"]==v else 0.0 for v in VEHICLE_CATS]
    f_vec = [1.0 if payload["Fuel Type"]==f else 0.0 for f in FUEL_CATS]
    return torch.from_numpy(np.array(num_feats+v_vec+f_vec,dtype=np.float32)).unsqueeze(0), payload["State"]

def load_checkpoint_into(model, path):
    if os.path.exists(path):
        try:
            ckpt = torch.load(path, map_location="cpu")  # PyTorch 2.6+ tries weights_only=True
        except Exception:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)  # fallback
        if isinstance(ckpt,dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)

# ---------- Load models ----------
IN_AS = 2+len(VEHICLE_CATS)+len(FUEL_CATS)
IN_WS = 2+len(VEHICLE_CATS)+len(FUEL_CATS)
IN_SG = 2+len(VEHICLE_CATS)+len(FUEL_CATS)
models = {}
for st in STATE_CATS:
    models[st] = {"co2":MLPRegressor(IN_AS),"energy":MLPRegressor(IN_AS),
                  "nox":MLPRegressor(IN_AS),"brake":MLPRegressor(IN_WS),"tire":MLPRegressor(IN_SG)}
    load_checkpoint_into(models[st]["co2"], f"models/{st}/best_model_{st}CO2.pth")
    load_checkpoint_into(models[st]["energy"], f"models/{st}/best_model_{st}Energy.pth")
    load_checkpoint_into(models[st]["nox"], f"models/{st}/best_model_{st}NOx.pth")
    load_checkpoint_into(models[st]["brake"], f"models/{st}/best_model_{st}PM25Brake.pth")
    load_checkpoint_into(models[st]["tire"], f"models/{st}/best_model_{st}PM25Tire.pth")
    for m in models[st].values(): m.eval()


def load_model(file_name, input_features):
    base_path = os.path.join(os.path.dirname(__file__), "ml_models")
    model_path = os.path.join(base_path, file_name)

    # Register ModifiedNet to resolve deserialization
    import sys
    sys.modules["__main__"].ModifiedNet = ModifiedNet

    # Load the checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    except Exception:
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)

    # Initialize the model architecture
    model = ModifiedNet(input_features)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  

    return {
        "model": model,
        "preprocessor": checkpoint["preprocessor"],
        "target_scaler": checkpoint.get("target_scaler", checkpoint.get("power_transformer", None)),
    }

# Load all models with specific input features
models = {
    "gaCo2": load_model("best_model_GACO2.pth", input_features=19),
    "gaTotalEnergyRate": load_model("best_model_GAEnergy.pth", input_features=20),
    "gaNOx": load_model("best_model_GANOx.pth", input_features=19),
    "gaPM25BrakeWear": load_model("best_model_GAPM25Brake.pth", input_features=20),
    "gaPM25TireWear": load_model("best_model_GAPM25Tire.pth", input_features=19),
}

def predict_one(inputs, model_key, models):
    """
    Runs a single prediction using the specified model key
    from the dynamically loaded model dictionary.
    """
    if model_key not in models:
        raise ValueError(f"Model key '{model_key}' not found in loaded models.")

    model_info = models[model_key]
    model = model_info["model"]
    preprocessor = model_info["preprocessor"]
    target_scaler = model_info["target_scaler"]

    df = pd.DataFrame([inputs])
    processed_input = preprocessor.transform(df)

    if hasattr(processed_input, "toarray"):
        input_tensor = torch.tensor(processed_input.toarray(), dtype=torch.float32)
    else:
        input_tensor = torch.tensor(processed_input, dtype=torch.float32)

    with torch.no_grad():
        raw_prediction = model(input_tensor)
        prediction = target_scaler.inverse_transform(raw_prediction.numpy())

    return float(prediction[0][0])


def predict_ones(inputs, model_key):
    if model_key not in models:
        raise ValueError(f"Model key '{model_key}' not found.")
    model_info = models[model_key]
    model = model_info["model"]
    preprocessor = model_info["preprocessor"]
    target_scaler = model_info["target_scaler"]

    df = pd.DataFrame([inputs])
    processed_input = preprocessor.transform(df)
    if hasattr(processed_input, "toarray"):
        input_tensor = torch.tensor(processed_input.toarray(), dtype=torch.float32)
    else:
        input_tensor = torch.tensor(processed_input, dtype=torch.float32)

    with torch.no_grad():
        raw_prediction = model(input_tensor)
        prediction = target_scaler.inverse_transform(raw_prediction.numpy())

    return float(prediction[0][0])


# ---------- Auth ----------
@app.route("/auth/login",methods=["POST"])
def login():
    data=request.get_json()
    conn=sqlite3.connect(DB_FILE); cur=conn.cursor()
    cur.execute("SELECT id FROM users WHERE username=? AND password=?", (data["username"],data["password"]))
    row=cur.fetchone(); conn.close()
    if row: return jsonify({"status":"ok","token":"fake-jwt"}),200
    return jsonify({"status":"fail"}),401

# ---------- Database Management ----------
@app.route("/admin/clear_db", methods=["POST"])
def clear_database():
    """Clear all data from database tables (except users table for safety)"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        
        # List of tables to clear (excluding users for safety)
        tables_to_clear = [
            "vehicle_classification",
            "penetration_rate", 
            "traffic_volume",
            "projected_traffic",
            "mfd_params",
            "results",
            "vehicle_classification_data",
            "traffic_volume_data",
            "projected_traffic_volume"
        ]
        
        cleared_tables = []
        for table in tables_to_clear:
            try:
                cur.execute(f"DELETE FROM {table}")
                rows_affected = cur.rowcount
                cleared_tables.append({"table": table, "rows_deleted": rows_affected})
            except sqlite3.OperationalError as e:
                # Table might not exist, continue with others
                cleared_tables.append({"table": table, "error": str(e)})
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "status": "success",
            "message": "Database cleared successfully",
            "cleared_tables": cleared_tables
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Failed to clear database: {str(e)}"
        }), 500

@app.route("/admin/transaction_id", methods=["GET"])
def get_transaction_id():
    """Get the current global transaction ID being used"""
    return jsonify({
        "transaction_id": GLOBAL_TRANSACTION_ID,
        "latest_transaction_id": LATEST_TRANSACTION_ID,
        "message": "This is the consistent transaction ID used for all operations"
    }), 200

# ---------- Upload (CSV/Excel into DB) ----------
def save_to_db(table, df):
    conn=sqlite3.connect(DB_FILE)
    df.to_sql(table, conn, if_exists="append", index=False)
    conn.close()

def save_to_db_data(type, df):
    table = f"{type}_data"
    conn = sqlite3.connect(DB_FILE)
    df.to_sql(table, conn, if_exists="append", index=False)
    conn.close()

# @app.route("/upload/<dataset>",methods=["POST"])
def upload(dataset):
    if "file" not in request.files: return jsonify({"error":"file missing"}),400
    file=request.files["file"]
    df=pd.read_csv(file) if file.filename.endswith(".csv") else pd.read_excel(file)
    valid={"vehicle_classification","penetration_rate","traffic_volume","projected_traffic","mfd_params"}
    if dataset not in valid: return jsonify({"error":"invalid dataset"}),400
    save_to_db(dataset, df)
    return jsonify({"status":"ok","rows":len(df)})

# ---------- Predictions ----------
@app.route("/predict/<kind>",methods=["POST"])
def predict(kind):
    data=request.get_json(force=True)
    st=data["State"]
    if kind not in models[st]: return jsonify({"error":"bad kind"}),400
    if kind in ["co2","energy","nox"]: x,_=preprocess(data,STATS_AS)
    elif kind=="brake": x,_=preprocess(data,STATS_WS)
    elif kind=="tire": x,_=preprocess(data,STATS_SG)
    with torch.no_grad(): y=models[st][kind](x).item()
    units={"co2":"g/km","energy":"kWh/100km","nox":"g/km","brake":"g/km","tire":"g/km"}
    return jsonify({"prediction_type":kind,"value":round(float(y),6),"unit":units[kind]})

# ---------- Scenarios ----------
@app.route("/scenarios/vehicles",methods=["POST"])
def vehicle_scenarios():
    data=request.get_json()
    years=data["years"]; st=data["state"]; payload=data["payload"]
    res={}
    for yr in years:
        p={**payload,"Age":yr%20,"Speed":60,"State":st}
        x,_=preprocess(p,STATS_AS)
        with torch.no_grad():
            co2=models[st]["co2"](x).item()
            nox=models[st]["nox"](x).item()
        res[yr]={"CO2":round(co2,4),"NOx":round(nox,4)}
    return jsonify(res)

# ---------- Results ----------
@app.route("/results/emissions",methods=["GET"])
def emissions():
    city=request.args.get("city"); year=int(request.args.get("year",2025))
    conn=sqlite3.connect(DB_FILE); cur=conn.cursor()
    cur.execute("SELECT vehicle_type,fuel_type FROM vehicle_classification WHERE city=?", (city,))
    rows=cur.fetchall(); conn.close()
    results=[]
    for v,f in rows:
        payload={"Age":year%20,"Speed":60,"Vehicle Type":v,"Fuel Type":f,"State":"GA"}
        x,_=preprocess(payload,STATS_AS)
        with torch.no_grad(): co2=models["GA"]["co2"](x).item()
        results.append({"vehicle":v,"fuel":f,"CO2":round(co2,4)})
    return jsonify({"city":city,"year":year,"results":results})

@app.route("/results/download",methods=["GET"])
def download():
    conn=sqlite3.connect(DB_FILE)
    df=pd.read_sql("SELECT * FROM results",conn); conn.close()
    out=io.StringIO(); df.to_csv(out,index=False); out.seek(0)
    return send_file(io.BytesIO(out.getvalue().encode()),mimetype="text/csv",
                     as_attachment=True,download_name="results.csv")


LATEST_TRANSACTION_ID = GLOBAL_TRANSACTION_ID  # Initialize with global transaction ID
@app.route("/upload/<type>", methods=["POST"])
def upload_vehicle_data(type):
    global LATEST_TRANSACTION_ID

    # Allowed dataset types
    valid = {
        "vehicle_classification",
        "penetration_rate",
        "traffic_volume",
        "projected_traffic",
        "mfd_params"
    }
    if type not in valid:
        return jsonify({"error": "invalid type"}), 400

    # Handle transaction_id logic
    transaction_id = request.form.get("transaction_id")

    if type == "vehicle_classification":
        # Use global consistent transaction_id
        transaction_id = GLOBAL_TRANSACTION_ID
        LATEST_TRANSACTION_ID = transaction_id  # store globally for reuse
    else:
        # For other uploads, use global transaction_id or provided one
        if not transaction_id:
            transaction_id = GLOBAL_TRANSACTION_ID  # use global consistent ID

    # Common metadata
    main_city = request.form.get("city_name") or request.form.get("city")
    main_vehicle_type = request.form.get("vehicle_type")
    main_year = request.form.get("year") or request.form.get("base_year")
    user_id = request.form.get("user_id")

    # ---------- Case: mfd_params (JSON payload) ----------
    if type == "mfd_params":
        # Handle JSON payload for MFD parameters table
        data = request.get_json(force=True) or {}
        city = data.get("city")
        base_year = data.get("base_year")
        mfd_table_data = data.get("mfd_table_data", [])
        mfd_table_headers = data.get("mfd_table_headers", [])
        
        if not mfd_table_data:
            return jsonify({"error": "mfd_table_data is required"}), 400
            
        # Convert table data to DataFrame
        df = pd.DataFrame(mfd_table_data, columns=mfd_table_headers)
        
        # Add metadata
        df["city"] = city
        df["base_year"] = base_year
        df["transaction_id"] = transaction_id
        
        save_to_db_safe("mfd_params", df)
        
        return jsonify({
            "status": "ok",
            "rows": len(df),
            "transaction_id": transaction_id
        })

    # ---------- Case: traffic_volume + mfd_params together ----------
    if type == "traffic_volume":
        print(f"[DEBUG] Processing traffic_volume upload")
        print(f"[DEBUG] Transaction ID: {transaction_id}")
        print(f"[DEBUG] Main city: {main_city}")
        
        file1 = request.files.get("file1")
        file2 = request.files.get("file2")
        print(f"[DEBUG] File1: {file1.filename if file1 else 'None'}")
        print(f"[DEBUG] File2: {file2.filename if file2 else 'None'}")
        
        if file1 is None or file2 is None:
            return jsonify({"error": "file1 and file2 are required"}), 400

        # --- Process file1 (traffic_volume) ---
        df1 = pd.read_csv(file1) if file1.filename.endswith(".csv") else pd.read_excel(file1)
        print(f"[DEBUG] File1 original shape: {df1.shape}")
        print(f"[DEBUG] File1 original columns: {list(df1.columns)}")
        
        # Calculate speed from distance and time if available
        if "Distance" in df1.columns and "Time" in df1.columns:
            df1["Speed"] = df1["Distance"] / df1["Time"]
            print(f"[DEBUG] Calculated Speed column")
        
        # Prepare data for traffic_volume_data table
        df1_processed = df1.copy()
        df1_processed.rename(columns={
            "VehicleType": "VehicleType",
            "base year ": "Year",
            "Distance": "Distance", 
            "Time": "Time"
        }, inplace=True)
        
        # Add required columns for traffic_volume_data table
        df1_processed["TransactionID"] = transaction_id
        df1_processed["City"] = main_city
        df1_processed["Tract"] = None  # Set to None if not provided
        df1_processed["Count"] = None  # Set to None if not provided
        
        # Ensure all required columns exist
        required_columns = ["TransactionID", "City", "Year", "Tract", "VehicleType", "Count", "Distance", "Time", "Speed"]
        for col in required_columns:
            if col not in df1_processed.columns:
                df1_processed[col] = None
                
        print(f"[DEBUG] About to save traffic_volume_data with shape: {df1_processed.shape}")
        save_to_db_safe("traffic_volume_data", df1_processed)

        # --- Process file2 (mfd_params) ---
        df2 = pd.read_csv(file2) if file2.filename.endswith(".csv") else pd.read_excel(file2)
        print(f"[DEBUG] File2 (MFD) original shape: {df2.shape}")
        print(f"[DEBUG] File2 (MFD) original columns: {list(df2.columns)}")
        print(f"[DEBUG] File2 (MFD) sample data:\n{df2.head()}")
        
        # The mfd_params table expects: tract, param_key, param_value
        # We need to transform the data to match this schema
        
        # If the file has multiple columns that represent different parameters,
        # we need to melt it into key-value pairs
        if len(df2.columns) > 1:
            # Assume first column is tract or identifier, rest are parameters
            id_cols = [df2.columns[0]] if len(df2.columns) > 1 else []
            value_cols = [col for col in df2.columns if col not in id_cols]
            
            # Melt the dataframe to convert columns to rows
            df2_melted = pd.melt(df2, id_vars=id_cols, value_vars=value_cols, 
                               var_name='param_key', value_name='param_value')
            
            # Set tract column
            if id_cols:
                df2_melted['tract'] = df2_melted[id_cols[0]].astype(str)
                df2_melted = df2_melted.drop(columns=id_cols)
            else:
                df2_melted['tract'] = None
                
            df2_processed = df2_melted
        else:
            # Single column case - treat as param_value with generic key
            df2_processed = pd.DataFrame({
                'tract': None,
                'param_key': 'value',
                'param_value': df2.iloc[:, 0]
            })
        
        print(f"[DEBUG] File2 (MFD) processed shape: {df2_processed.shape}")
        print(f"[DEBUG] File2 (MFD) processed columns: {list(df2_processed.columns)}")
        print(f"[DEBUG] File2 (MFD) processed sample:\n{df2_processed.head()}")
        
        save_to_db_safe("mfd_params", df2_processed)

        return jsonify({
            "status": "ok",
            "transaction_id": transaction_id,
            "rows": {
                "traffic_volume_data": len(df1_processed),
                "mfd_params": len(df2_processed)
            },
            "debug": {
                "traffic_columns": list(df1_processed.columns),
                "mfd_columns": list(df2_processed.columns)
            }
        })

    # ---------- Default Case: other datasets ----------
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "file missing"}), 400

    df = pd.read_csv(file) if file.filename.endswith(".csv") else pd.read_excel(file)

    if type == "vehicle_classification":
        df.rename(columns={
            "City": "data_city",
            "Vehicle Type": "data_vehicle_type",
            "vehicle_type": "data_vehicle_type",
            "Fuel Type": "fuel_type",
            "Vehicle Count": "vehicle_count",
            "penetration": "age"
        }, inplace=True)
        # Add metadata for vehicle_classification
        if main_city and "city" not in df.columns:
            df["city"] = main_city
        if main_vehicle_type and "vehicle_type" not in df.columns:
            df["vehicle_type"] = main_vehicle_type
        if main_year and "BaseYear" not in df.columns:
            df["BaseYear"] = main_year
        df["transaction_id"] = transaction_id
        df["user_id"] = user_id
        # Keep relevant columns for vehicle_classification_data
        vehicle_cols = ["data_city", "data_vehicle_type", "fuel_type", "vehicle_count", "age", 
                       "city", "vehicle_type", "BaseYear", "transaction_id", "user_id"]
        df = df[[col for col in vehicle_cols if col in df.columns]]

    elif type == "penetration_rate":
        df.rename(columns={
            "City": "city",
            "Vehicle Type": "vehicle_type",
            "Fuel Type": "fuel_type",
            "Penetration": "penetration"
        }, inplace=True)
        # Add metadata for penetration_rate
        if main_city and "city" not in df.columns:
            df["city"] = main_city
        if main_vehicle_type and "vehicle_type" not in df.columns:
            df["vehicle_type"] = main_vehicle_type
        df["transaction_id"] = transaction_id
        df["user_id"] = user_id
        # Keep only relevant columns
        penetration_cols = ["city", "vehicle_type", "fuel_type", "penetration", "transaction_id", "user_id"]
        df = df[[col for col in penetration_cols if col in df.columns]]

    elif type == "projected_traffic":
        df.rename(columns={
            "Tract": "tract",
            "Year": "year",
            "Volume": "volume"
        }, inplace=True)
        # Add metadata for projected_traffic
        df["transaction_id"] = transaction_id
        # Keep only relevant columns
        traffic_cols = ["tract", "year", "volume", "transaction_id"]
        df = df[[col for col in traffic_cols if col in df.columns]]

    else:
        # Add metadata for other types
        df["city"] = main_city
        df["vehicle_type"] = main_vehicle_type
        df["BaseYear"] = main_year
        df["transaction_id"] = transaction_id
        df["user_id"] = user_id

    save_to_db_data(type, df)

    return jsonify({
        "status": "ok",
        "rows": len(df),
        "transaction_id": transaction_id
    })

def save_to_db_safe(table, df):
    print(f"[DEBUG] Saving to table: {table}")
    print(f"[DEBUG] DataFrame shape: {df.shape}")
    print(f"[DEBUG] DataFrame columns: {list(df.columns)}")
    print(f"[DEBUG] Sample data:\n{df.head()}")
    
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    db_cols = [col[1] for col in cur.fetchall()]
    print(f"[DEBUG] Database columns for {table}: {db_cols}")
    
    df_filtered = df[[col for col in df.columns if col in db_cols]]
    print(f"[DEBUG] Filtered DataFrame columns: {list(df_filtered.columns)}")
    
    for col in db_cols:
        if col not in df_filtered.columns:
            df_filtered[col] = None
            print(f"[DEBUG] Added missing column: {col}")
    
    print(f"[DEBUG] Final DataFrame before saving:\n{df_filtered.head()}")
    
    try:
        rows_added = df_filtered.to_sql(table, conn, if_exists="append", index=False)
        print(f"[DEBUG] Successfully saved {len(df_filtered)} rows to {table}")
        
        # Verify the data was actually saved
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        total_count = cur.fetchone()[0]
        print(f"[DEBUG] Total rows in {table} after save: {total_count}")
        
    except Exception as e:
        print(f"[ERROR] Failed to save to {table}: {str(e)}")
        conn.close()
        raise e
    
    conn.close()
    
    
    
@app.route("/upload/projected_traffic", methods=["POST"])
def upload_projected_traffic():
    try:
        global LATEST_TRANSACTION_ID

        # -------- Get form inputs --------
        city_name = request.form.get("city_name")
        year = request.form.get("year")
        file_csv = request.files.get("file_csv")          # master CSV
        file_table_file = request.files.get("file_table") # detail file
        file_table_text = request.form.get("file_table")  # detail JSON text

        if not all([city_name, year, file_csv]) or (not file_table_file and not file_table_text):
            return jsonify({"error": "Missing required parameters"}), 400

        # -------- Transaction ID logic --------
        transaction_id = request.form.get("transaction_id")
        if not transaction_id:
            transaction_id = GLOBAL_TRANSACTION_ID  # Use global consistent ID

        # -------- Parse master CSV --------
        df_csv = pd.read_csv(file_csv) if file_csv.filename.endswith(".csv") else pd.read_excel(file_csv)

        # ✅ Rename columns to match DB schema
        df_csv.rename(columns={
            "Tract ID": "tract",
            "Tract_ID": "tract",
            "Traffic Volume": "volume"
        }, inplace=True)

        # Add metadata to every row of CSV
        df_csv["transaction_id"] = transaction_id
        df_csv["city_name"] = city_name
        df_csv["year"] = int(year)
        
        # ✅ Keep only columns that exist in the projected_traffic_volume table
        required_cols = ["transaction_id", "city_name", "year", "tract", "volume"]
        df_csv = df_csv[[col for col in required_cols if col in df_csv.columns]]
        
        print(f"[DEBUG] CSV columns before save: {list(df_csv.columns)}")
        print(f"[DEBUG] CSV shape: {df_csv.shape}")

        # -------- Parse file_table (detail data) --------
        if file_table_file:
            try:
                df_table = pd.read_csv(file_table_file)
            except Exception:
                try:
                    df_table = pd.read_json(file_table_file)
                except Exception:
                    return jsonify({"error": "file_table must be valid CSV or JSON"}), 400
        else:
            try:
                data = json.loads(file_table_text)
                df_table = pd.DataFrame(data)
            except Exception as e:
                return jsonify({"error": f"Invalid JSON in file_table: {str(e)}"}), 400

        # -------- Normalize detail table columns --------
        df_table.rename(columns={
            "time": "time",
            "datetime": "datetime",
            "Traffic Volume": "traffic_volume",
            "Traffic Speed": "traffic_speed",
            "Adjusted Traffic": "adjusted_traffic",
            "Density": "density",
            "Speed": "speed"
        }, inplace=True)

        # Add transaction_id
        df_table["transaction_id"] = transaction_id
        
        # ✅ Keep only columns that exist in the projected_traffic_details table
        detail_cols = ["transaction_id", "time", "datetime", "traffic_volume", 
                      "traffic_speed", "adjusted_traffic", "density", "speed"]
        df_table = df_table[[col for col in detail_cols if col in df_table.columns]]
        
        print(f"[DEBUG] Detail table columns before save: {list(df_table.columns)}")
        print(f"[DEBUG] Detail table shape: {df_table.shape}")

        # -------- Save to DB --------
        conn = sqlite3.connect(DB_FILE)
        df_csv.to_sql("projected_traffic_volume", conn, if_exists="append", index=False)
        df_table.to_sql("projected_traffic_details", conn, if_exists="append", index=False)
        conn.close()

        # -------- Response --------
        return jsonify({
            "status": "ok",
            "transaction_id": transaction_id,
            "volume_rows": len(df_csv),
            "detail_rows": len(df_table)
        })

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


def load_state_models(city_name: str):
    """
    Dynamically load models based on the city/state.
    Each state's models are stored under 'models/<STATE>/'.
    """
    state = CITY_TO_STATE.get(city_name)

    # Build absolute path using this file's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    state_path = os.path.join(base_dir, "models", state)

    print(f"[INFO] City={city_name}, State={state}, Path={state_path}")

    feature_map_all = {
        "CA": {"CO2": 19, "Energy": 20, "NOx": 19, "PM25Brake": 20, "PM25Tire": 20},
        "GA": {"CO2": 19, "Energy": 20, "NOx": 19, "PM25Brake": 20, "PM25Tire": 19},
        "NY": {"CO2": 18, "Energy": 19, "NOx": 18, "PM25Brake": 19, "PM25Tire": 19},
        "WA": {"CO2": 19, "Energy": 20, "NOx": 19, "PM25Brake": 20, "PM25Tire": 20}
    }

    # Select feature map for this state
    feature_map = feature_map_all.get(state, feature_map_all["GA"])

    try:
        return {
            f"{state.lower()}Co2": load_model(os.path.join(state_path, f"best_model_{state}CO2.pth"), input_features=feature_map["CO2"]),
            f"{state.lower()}TotalEnergyRate": load_model(os.path.join(state_path, f"best_model_{state}Energy.pth"), input_features=feature_map["Energy"]),
            f"{state.lower()}NOx": load_model(os.path.join(state_path, f"best_model_{state}NOx.pth"), input_features=feature_map["NOx"]),
            f"{state.lower()}PM25BrakeWear": load_model(os.path.join(state_path, f"best_model_{state}PM25Brake.pth"), input_features=feature_map["PM25Brake"]),
            f"{state.lower()}PM25TireWear": load_model(os.path.join(state_path, f"best_model_{state}PM25Tire.pth"), input_features=feature_map["PM25Tire"]),
        }
    except Exception as e:
        raise FileNotFoundError(f"❌ Failed to load models for state '{state}' from {state_path}: {e}")
    
    
@app.route("/predict_emissions", methods=["POST"])
def predict_emissions():
    try:
        # Step 1: Get the input data from the request
        data = request.get_json(force=True)

        if not data:
            return jsonify({"error": "No JSON payload received"}), 400

        # Step 2: Extract required fields from the payload
        cityname = data.get("City", "").strip()
        fuel_type = data.get("FuelType", "").strip()
        vehicle_type = data.get("VehicleType", "").strip()
        age = data.get("Age", 5)  # Default age if not provided
        emission_type = data.get("EmissionType", "").strip()  
        
        # Step 3: Validate the required fields
        if not all([cityname, fuel_type, vehicle_type]):
            return jsonify({"error": f"Missing required fields: city, fuelType, vehicleType"}), 400

        state = CITY_TO_STATE.get(cityname, "GA")
        state_models = load_state_models(cityname)

        # Step 4: Get the dynamic speed from the predefined list
        speeds = list(range(0, 71))  # Get a random speed from the list

        # Step 5: Prepare the input data for the prediction
        payload = {
            "Age": age,
            "Vehicle Type": vehicle_type,
            "Fuel Type": fuel_type,
            "State": state
        }

        all_predictions = []

        emission_type_mapping = {
            "CO2 Emissions": [f"{state.lower()}Co2", "g/mile"],
            "Energy Rate": [f"{state.lower()}TotalEnergyRate", "kWh/100km"],
            "NOx": [f"{state.lower()}NOx", "g/mile"],
            "PM2.5 Brake Wear": [f"{state.lower()}PM25BrakeWear", "g/mile"],
            "PM2.5 Tire Wear": [f"{state.lower()}PM25TireWear", "g/mile"]
        }
        
        if emission_type and emission_type not in emission_type_mapping:
            valid_types = ", ".join(emission_type_mapping.keys())
            return jsonify({
                "error": f"Invalid EmissionType '{emission_type}'. "
                        f"Valid options: {valid_types}"
            }), 400

        # Step 7: Loop over the speed values and call predict_one for each speed
        for speed in speeds:
            payload["Speed"] = speed
            if emission_type == "PM2.5 Brake Wear":
                payload["Vehicle Weight"] = data.get("VehicleWeight", 1500)
            elif emission_type == "PM2.5 Tire Wear":
                payload["Road Gradient"] = data.get("RoadGradient", 5)

            if emission_type:
                # Predict only one emission type
                model_key, unit = emission_type_mapping[emission_type]
                try:
                    prediction = predict_one(payload, model_key, models=state_models)
                except Exception as e:
                    return jsonify({
                        "error": f"Prediction failed at speed {speed}: {str(e)}"
                    }), 400

                all_predictions.append({
                    "EmissionType": emission_type,
                    "PredictedValue": round(prediction, 6),
                    "Unit": unit,
                    "Speed": speed,
                    "VehicleType": vehicle_type,
                    "FuelType": fuel_type,
                    "City": cityname,
                    "State": state
                })

            else:
                # Predict all emission types
                multi_result = {"Speed": speed, "City": cityname, "State": state}
                for label, (key, unit) in emission_type_mapping.items():
                    try:
                        multi_result[label] = round(predict_one(payload, key, models=state_models), 6)
                    except Exception:
                        multi_result[label] = None
                all_predictions.append(multi_result)
                # If no EmissionType is provided, return all predictions for all models
                result = {}
                for key in models.keys():
                    try:
                        result[key] = predict_one(payload, key)
                    except Exception as e:
                        result[key] = None
                result["speed"] = speed
                all_predictions.append(result)

        return jsonify(all_predictions), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 400

@app.route("/predict_consumption", methods=["POST"])
def predict_consumption():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON payload received"}), 400

        # Required parameters
        cityname = data.get("city", "").strip()
        fuel_type = data.get("fuelType", "").strip()
        vehicle_type = data.get("vehicleType", "").strip()
        age = data.get("vehicleAge", 5)
        speeds = list(range(0, 70))  # speeds 0–69

        # Validate required
        if not all([cityname, fuel_type, vehicle_type]):
            return jsonify({"error": "Missing required parameters"}), 400

        # Map city to state
        normalized_city = cityname.strip().lower()
        CITY_TO_STATE_NORMALIZED = {
            "atlanta": "GA",
            "los angeles": "CA",
            "losangeles": "CA",
            "newyork": "NY",
            "new york": "NY",
            "seattle": "WA"
        }
        state = CITY_TO_STATE_NORMALIZED.get(normalized_city, "GA")

        # Step 5: Load correct models dynamically based on city
        state_models = load_state_models(cityname)
        # Validate age
        try:
            age = float(age) if age else 5.0
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid age value"}), 400

        # Validate vehicle_type
        if vehicle_type and vehicle_type not in VEHICLE_CATS:
            return jsonify({"error": f"Invalid vehicle_type: {vehicle_type}"}), 400
        
        KJ_PER_UNIT = {
            "Gasoline": 123462.432,                
            "Diesel Fuel": 138451.739,
            "Ethanol - E-85": 85729.28,
            "Compressed Natural Gas - CNG": 124854.529,
            "Electricity": 3600.0
        }

        UNIT_TYPE = {
            "Gasoline": "gallon/mile",
            "Diesel Fuel": "gallon/mile",
            "Ethanol - E-85": "gallon/mile",
            "Compressed Natural Gas - CNG": "GGE/mile",
            "Electricity": "kWh/mile"
        }

        # Use GA energy model
        energy_model_key = f"{state.lower()}TotalEnergyRate"
        if energy_model_key not in state_models:
            return jsonify({"error": f"Energy model '{energy_model_key}' not found for {state}"}), 400

        results = []  # collect results for all speeds
        SCALING_FACTOR = 0.001

        for speed in speeds:
            payload = {
                "Age": age,
                "Speed": speed,
                "Vehicle Type": vehicle_type,
                "Fuel Type": fuel_type,
                "State": state  
            }

            try:
                consumptions = predict_one(payload, energy_model_key, models=state_models)
            except Exception as e:
                return jsonify({"error": f"Prediction failed at speed {speed}: {str(e)}"}), 400

            energy_kj_per_mile = (consumptions * 3600) / 62.1371
            
            kj_per_unit = KJ_PER_UNIT.get(fuel_type)
            if not kj_per_unit:
                return jsonify({"error": f"Fuel type '{fuel_type}' not recognized"}), 400
            
            fuel_consumption = energy_kj_per_mile / kj_per_unit
            fuel_unit = UNIT_TYPE.get(fuel_type, "unit/mile")

            
            results.append({
                "city": cityname,
                "state": state,
                "vehicle_type": vehicle_type,
                "fuel_type": fuel_type,
                "age": age,
                "speed": speed,
                "energy_kj_per_mile": round(energy_kj_per_mile, 6),
                "fuel_consumption": round(fuel_consumption, 8),
                "fuel_unit": fuel_unit
            })

        return jsonify(results), 200

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500



    
if __name__=="__main__":
    app.run(host="0.0.0.0",port=5003,debug=True)
