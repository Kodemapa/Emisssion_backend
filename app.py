from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
import torch.nn as nn

# ---------- Flask ----------
app = Flask(__name__)
CORS(app)

# ---------- Allow safe sklearn globals for torch.load ----------
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([
        ColumnTransformer,
        StandardScaler,
        OneHotEncoder,
        PowerTransformer
    ])

# ---------- Model ----------
class MLPRegressor(nn.Module):
    def __init__(self, in_features: int, hidden=(32, 16)):
        super().__init__()
        layers, last = [], in_features
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ---------- Categories ----------
VEHICLE_CATS = [
    "Combination long-haul Truck",
    "Combination short-haul Truck",
    "Light Commercial Truck",
    "Motorhome - Recreational Vehicle",
    "Motorcycle",
    "Other Buses",
    "Passenger Car",
    "Passenger Truck",
    "Refuse Truck",
    "School Bus",
    "Single Unit long-haul Truck",
    "Single Unit short-haul Truck",
    "Transit Bus"
]

FUEL_CATS = [
    "Gasoline",
    "Diesel Fuel",
    "Electricity",
    "Compressed Natural Gas - CNG",
    "Ethanol - E-85"
]

STATE_CATS = ["CA", "GA", "NY", "WA"]

# ---------- Training stats ----------
STATS_AS = {"Age": {"mean": 5.0, "std": 3.0}, "Speed": {"mean": 60.0, "std": 15.0}}
STATS_WS = {"Vehicle Weight": {"mean": 1500.0, "std": 300.0}, "Speed": {"mean": 60.0, "std": 15.0}}
STATS_SG = {"Speed": {"mean": 60.0, "std": 15.0}, "Road Gradient": {"mean": 5.0, "std": 3.0}}

# ---------- Utils ----------
def _z(val, mean, std): 
    std = 1.0 if std in (None, 0, 0.0) else std
    return (float(val) - float(mean)) / float(std)

def preprocess(payload: dict, numeric_stats: dict, v_key="Vehicle Type", f_key="Fuel Type", s_key="State"):
    num_feats = []
    for k, s in numeric_stats.items():
        if k not in payload:
            raise ValueError(f"Missing numeric field: {k}")
        num_feats.append(_z(payload[k], s["mean"], s["std"]))
    vehicle, fuel, state = str(payload[v_key]), str(payload[f_key]), str(payload[s_key])
    if vehicle not in VEHICLE_CATS: raise ValueError(f"'{v_key}' must be one of {VEHICLE_CATS}")
    if fuel not in FUEL_CATS: raise ValueError(f"'{f_key}' must be one of {FUEL_CATS}")
    if state not in STATE_CATS: raise ValueError(f"'{s_key}' must be one of {STATE_CATS}")
    v_vec = [1.0 if vehicle == v else 0.0 for v in VEHICLE_CATS]
    f_vec = [1.0 if fuel == f else 0.0 for f in FUEL_CATS]
    feats = np.array(num_feats + v_vec + f_vec, dtype=np.float32)
    return torch.from_numpy(feats).unsqueeze(0), state

def load_checkpoint_into(model: nn.Module, path: str):
    try:
        ckpt = torch.load(path, map_location="cpu")  # PyTorch 2.6+ tries weights_only=True
    except Exception:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)  # fallback

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt, strict=False)
    else:
        model.load_state_dict(ckpt.state_dict(), strict=False)

# ---------- Load models ----------
IN_AS = 2 + len(VEHICLE_CATS) + len(FUEL_CATS)
IN_WS = 2 + len(VEHICLE_CATS) + len(FUEL_CATS)
IN_SG = 2 + len(VEHICLE_CATS) + len(FUEL_CATS)

models = {}
for state in STATE_CATS:
    models[state] = {
        "co2": MLPRegressor(IN_AS),
        "energy": MLPRegressor(IN_AS),
        "nox": MLPRegressor(IN_AS),
        "brake": MLPRegressor(IN_WS),
        "tire": MLPRegressor(IN_SG)
    }
    load_checkpoint_into(models[state]["co2"], f"models/{state}/best_model_{state}CO2.pth")
    load_checkpoint_into(models[state]["energy"], f"models/{state}/best_model_{state}Energy.pth")
    load_checkpoint_into(models[state]["nox"], f"models/{state}/best_model_{state}NOx.pth")
    load_checkpoint_into(models[state]["brake"], f"models/{state}/best_model_{state}PM25Brake.pth")
    load_checkpoint_into(models[state]["tire"], f"models/{state}/best_model_{state}PM25Tire.pth")
    for model in models[state].values():
        model.eval()

# ---------- Routes ----------
@app.route("/")
def index():
    return "Prediction API: use /predict/co2, /predict/energy, /predict/nox, /predict/brake, /predict/tire"

@app.route("/predict/co2", methods=["POST"])
def predict_co2():
    data = request.get_json(force=True) or {}
    x, state = preprocess(data, STATS_AS)
    with torch.no_grad():
        y = models[state]["co2"](x).item()
    return jsonify({
        "prediction_type": "CO2 Emissions",
        "prediction": {"value": round(float(y), 6), "unit": "g/km"}
    }), 200

@app.route("/predict/energy", methods=["POST"])
def predict_energy():
    data = request.get_json(force=True) or {}
    x, state = preprocess(data, STATS_AS)
    with torch.no_grad():
        y = models[state]["energy"](x).item()
    return jsonify({
        "prediction_type": "Energy Rate",
        "prediction": {"value": round(float(y), 6), "unit": "kWh/100km"}
    }), 200

@app.route("/predict/nox", methods=["POST"])
def predict_nox():
    data = request.get_json(force=True) or {}
    x, state = preprocess(data, STATS_AS)
    with torch.no_grad():
        y = models[state]["nox"](x).item()
    return jsonify({
        "prediction_type": "NOx",
        "prediction": {"value": round(float(y), 6), "unit": "g/km"}
    }), 200

@app.route("/predict/brake", methods=["POST"])
def predict_brake():
    data = request.get_json(force=True) or {}
    x, state = preprocess(data, STATS_WS)
    with torch.no_grad():
        y = models[state]["brake"](x).item()
    return jsonify({
        "prediction_type": "PM2.5 Brake Wear",
        "prediction": {"value": round(float(y), 6), "unit": "g/km"}
    }), 200

@app.route("/predict/tire", methods=["POST"])
def predict_tire():
    data = request.get_json(force=True) or {}
    x, state = preprocess(data, STATS_SG)
    with torch.no_grad():
        y = models[state]["tire"](x).item()
    return jsonify({
        "prediction_type": "PM2.5 Tire Wear",
        "prediction": {"value": round(float(y), 6), "unit": "g/km"}
    }), 200

# ---------- Run ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

# ---------- Model ----------
class MLPRegressor(nn.Module):
    def __init__(self, in_features: int, hidden=(32, 16)):
        super().__init__()
        layers, last = [], in_features
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ---------- Categories ----------
VEHICLE_CATS = [
    "Combination long-haul Truck",
    "Combination short-haul Truck",
    "Light Commercial Truck",
    "Motorhome - Recreational Vehicle",
    "Motorcycle",
    "Other Buses",
    "Passenger Car",
    "Passenger Truck",
    "Refuse Truck",
    "School Bus",
    "Single Unit long-haul Truck",
    "Single Unit short-haul Truck",
    "Transit Bus"
]

FUEL_CATS = [
    "Gasoline",
    "Diesel Fuel",
    "Electricity",
    "Compressed Natural Gas - CNG",
    "Ethanol - E-85"
]

STATE_CATS = ["CA", "GA", "NY", "WA"]

# ---------- Training stats ----------
STATS_AS = {"Age": {"mean": 5.0, "std": 3.0}, "Speed": {"mean": 60.0, "std": 15.0}}
STATS_WS = {"Vehicle Weight": {"mean": 1500.0, "std": 300.0}, "Speed": {"mean": 60.0, "std": 15.0}}
STATS_SG = {"Speed": {"mean": 60.0, "std": 15.0}, "Road Gradient": {"mean": 5.0, "std": 3.0}}

# ---------- Utils ----------
def _z(val, mean, std): 
    std = 1.0 if std in (None, 0, 0.0) else std
    return (float(val) - float(mean)) / float(std)

