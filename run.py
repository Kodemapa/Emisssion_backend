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
    "Combination long-haul Truck","Combination short-haul Truck","Light Commercial Truck",
    "Motorhome - Recreational Vehicle","Motorcycle","Other Buses","Passenger Car","Passenger Truck",
    "Refuse Truck","School Bus","Single Unit long-haul Truck","Single Unit short-haul Truck","Transit Bus"
]
FUEL_CATS = ["Gasoline","Diesel Fuel","Electricity","Compressed Natural Gas - CNG","Ethanol - E-85"]
STATE_CATS = ["CA","GA","NY","WA"]
CITY_TO_STATE = {
    "Atlanta": "CA",
    "Los Angeles": "GA",
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

# ---------- Auth ----------
@app.route("/auth/login",methods=["POST"])
def login():
    data=request.get_json()
    conn=sqlite3.connect(DB_FILE); cur=conn.cursor()
    cur.execute("SELECT id FROM users WHERE username=? AND password=?", (data["username"],data["password"]))
    row=cur.fetchone(); conn.close()
    if row: return jsonify({"status":"ok","token":"fake-jwt"}),200
    return jsonify({"status":"fail"}),401

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


LATEST_TRANSACTION_ID = None  
@app.route("/upload/<type>", methods=["POST"])
def upload_vehicle_data(type):
    global LATEST_TRANSACTION_ID

    # Allowed dataset types
    valid = {
        "vehicle_classification",
        "penetration_rate",
        "traffic_volume",
        "projected_traffic"
    }
    if type not in valid:
        return jsonify({"error": "invalid type"}), 400

    # Handle transaction_id logic
    transaction_id = request.form.get("transaction_id")

    if type == "vehicle_classification":
        # Always create a fresh transaction_id
        transaction_id = str(uuid.uuid4())
        LATEST_TRANSACTION_ID = transaction_id  # store globally for reuse
    else:
        # For other uploads, reuse or require transaction_id
        if not transaction_id:
            if LATEST_TRANSACTION_ID:
                transaction_id = LATEST_TRANSACTION_ID  # reuse last created
            else:
                return jsonify({"error": "transaction_id is required (no vehicle_classification uploaded yet)"}), 400

    # Common metadata
    main_city = request.form.get("city_name") or request.form.get("main_city")
    main_vehicle_type = request.form.get("vehicle_type")
    main_year = request.form.get("year")
    user_id = request.form.get("user_id")

    # ---------- Case: traffic_volume + mfd_params together ----------
    if type == "traffic_volume":
        file1 = request.files.get("file1")
        file2 = request.files.get("file2")
        if file1 is None or file2 is None:
            return jsonify({"error": "file1 and file2 are required"}), 400

        # --- Process file1 (traffic_volume) ---
        df1 = pd.read_csv(file1) if file1.filename.endswith(".csv") else pd.read_excel(file1)
        hour_cols = [c for c in df1.columns if c.lower().startswith("hour")]
        df1_long = df1.melt(
            id_vars=["VehicleType", "base year "],
            value_vars=hour_cols,
            var_name="hour",
            value_name="volume"
        )
        df1_long.rename(columns={"base year ": "BaseYear"}, inplace=True)
        df1_long["city"] = main_city
        df1_long["transaction_id"] = transaction_id
        df1_long["user_id"] = user_id
        save_to_db_safe("traffic_volume", df1_long)

        # --- Process file2 (mfd_params) ---
        df2 = pd.read_csv(file2) if file2.filename.endswith(".csv") else pd.read_excel(file2)
        df2.rename(columns={
            "VehicleType": "vehicle_type",
            "Distance": "distance",
            "Time": "time"
        }, inplace=True)
        df2["tract"] = None
        df2["city"] = main_city
        df2["transaction_id"] = transaction_id
        save_to_db_safe("mfd_params", df2)

        return jsonify({
            "status": "ok",
            "transaction_id": transaction_id,
            "rows": {
                "traffic_volume": len(df1_long),
                "mfd_params": len(df2)
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

    elif type == "penetration_rate":
        df.rename(columns={
            "City": "city",
            "Vehicle Type": "vehicle_type",
            "Fuel Type": "fuel_type",
            "Penetration": "penetration"
        }, inplace=True)

    elif type == "projected_traffic":
        df.rename(columns={
            "Tract": "tract",
            "Year": "year",
            "Volume": "volume"
        }, inplace=True)

    # Add metadata
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
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    db_cols = [col[1] for col in cur.fetchall()]
    df = df[[col for col in df.columns if col in db_cols]]
    for col in db_cols:
        if col not in df.columns:
            df[col] = None
    df.to_sql(table, conn, if_exists="append", index=False)
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
            if LATEST_TRANSACTION_ID:
                transaction_id = LATEST_TRANSACTION_ID
            else:
                return jsonify({"error": "transaction_id is required (no vehicle_classification uploaded yet)"}), 400

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


@app.route("/predict_emissions", methods=["POST"])
def predict_emissions():
    try:
        # Collect form data
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON payload received"}), 400

        # If frontend sends a list, loop through it
        if isinstance(data, list):
            requests_list = data
        else:
            requests_list = [data]

        results_all = []

        for req in requests_list:
            prediction_type_raw = req.get("emissionType", "").strip()
            cityname = req.get("city", "").strip()
            fuel_type = req.get("fuelType", "").strip()
            age = req.get("vehicleAge")
            transaction_id = req.get("transaction_id", "").strip() if req.get("transaction_id") else None
            vehicle_type = req.get("vehicleType", "").strip()

            # Validate required
            if not all([prediction_type_raw, cityname, fuel_type]):
                return jsonify({"error": "Missing required parameters"}), 400

            # Normalize prediction_type
            prediction_map = {
                "co2 emission": "co2",
                "CO2 Emissions": "co2",
                "co2 emissions": "co2",
                "energy rate": "energy",
                "nox": "nox",
                "pm2.5 brake wear": "brake",
                "pm2.5 tire wear": "tire"
            }
            prediction_type = prediction_map.get(prediction_type_raw.lower())
            if not prediction_type:
                return jsonify({"error": f"Invalid prediction_type: {prediction_type_raw}"}), 400

            # Map city to state
            state = CITY_TO_STATE.get(cityname, None)
            if not state:
                return jsonify({"error": f"City '{cityname}' not mapped to any state"}), 400
            if state not in models:
                return jsonify({"error": f"No models found for state '{state}'"}), 400

            # Validate age
            try:
                age = age if age else 5
            except (ValueError, TypeError):
                return jsonify({"error": "Invalid age value"}), 400

            # Validate vehicle_type if provided
            if vehicle_type and vehicle_type not in VEHICLE_CATS:
                return jsonify({"error": f"Invalid vehicle_type: {vehicle_type}"}), 400

            # Fetch traffic data
            conn = sqlite3.connect(DB_FILE)
            cur = conn.cursor()
            query = """
                SELECT Speed 
                FROM traffic_volume_data 
                WHERE TransactionID = ?
            """
            params = [transaction_id]
            cur.execute(query, params)
            rows = cur.fetchall()
            conn.close()

            if not rows:
                return jsonify({"error": f"No traffic data found for transaction_id {transaction_id}"}), 404

            # Stats selector
            stats_map = {"co2": STATS_AS, "energy": STATS_AS, "nox": STATS_AS,
                         "brake": STATS_WS, "tire": STATS_SG}

            # Prediction loop
            results = []
            for row in rows:
                speed = row[0]

                if prediction_type in ["co2", "energy", "nox"]:
                    payload = {"Age": age,"Speed": speed,"Vehicle Type": vehicle_type,
                               "Fuel Type": fuel_type,"State": state}
                elif prediction_type == "brake":
                    payload = {"Vehicle Weight": 1500,"Speed": speed,"Vehicle Type": vehicle_type,
                               "Fuel Type": fuel_type,"State": state}
                elif prediction_type == "tire":
                    payload = {"Speed": speed,"Road Gradient": 5,"Vehicle Type": vehicle_type,
                               "Fuel Type": fuel_type,"State": state}

                x, _ = preprocess(payload, stats_map[prediction_type])
                with torch.no_grad():
                    y = models[state][prediction_type](x).item()

                results.append({
                    "vehicle_type": vehicle_type,
                    "speed": speed,
                    "prediction": round(float(y), 6)
                })

            # Final response for this request
            units = {
                "co2": "g/km", "energy": "kWh/100km", "nox": "g/km",
                "brake": "g/km", "tire": "g/km"
            }
            display_names = {
                "co2": "CO2 Emissions", "energy": "Energy Rate", "nox": "NOx",
                "brake": "PM2.5 Brake Wear", "tire": "PM2.5 Tire Wear"
            }

            results_all.append({
                "city": cityname,
                "state": state,
                "transaction_id": transaction_id,
                "prediction_type": display_names[prediction_type],
                "unit": units[prediction_type],
                "results": results
            })

        return jsonify(results_all)

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500
    
    
    
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
        speed = data.get("speed", 60)

        # Validate required
        if not all([cityname, fuel_type, vehicle_type]):
            return jsonify({"error": "Missing required parameters"}), 400

        # Map city to state
        state = CITY_TO_STATE.get(cityname, None)
        if not state:
            return jsonify({"error": f"City '{cityname}' not mapped to any state"}), 400
        if state not in models:
            return jsonify({"error": f"No models found for state '{state}'"}), 400

        # Validate age
        try:
            age = float(age) if age else 5.0
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid age value"}), 400

        # Validate vehicle_type
        if vehicle_type and vehicle_type not in VEHICLE_CATS:
            return jsonify({"error": f"Invalid vehicle_type: {vehicle_type}"}), 400

        # Preprocess for energy model
        payload = {"Age": age, "Speed": speed, "Vehicle Type": vehicle_type,
                   "Fuel Type": fuel_type, "State": state}
        x, _ = preprocess(payload, STATS_AS)

        # Predict consumption
        with torch.no_grad():
            consumption = models[state]["energy"](x).item()

        # Fuel consumption conversion (optional)
        fuel_consumption = None
        if fuel_type == "Gasoline":
            fuel_consumption = round(consumption / 8.9, 6)  # L/100km
        elif fuel_type == "Diesel Fuel":
            fuel_consumption = round(consumption / 10.0, 6)  # L/100km
        elif fuel_type == "Electricity":
            fuel_consumption = None  # Already in kWh

        return jsonify({
            "city": cityname,
            "state": state,
            "vehicle_type": vehicle_type,
            "fuel_type": fuel_type,
            "age": age,
            "speed": speed,
            "energy_consumption": round(consumption, 6),
            "energy_unit": "kWh/100km",
            "fuel_consumption": fuel_consumption,
            "fuel_unit": "L/100km" if fuel_consumption else None
        })

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500
    


    
if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)
