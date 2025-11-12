import os
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import interp1d  # <-- FIX: Added this import
import warnings
from run import CITY_TO_STATE # Import the city mapping from your run.py

# --- State-Specific Configurations ---
# We move the hardcoded GA logic here and add the new CA logic
STATE_CONFIGS = {
    "GA": {
        "tract_ids": [
            '13121001800', '13121001900', '13121003600', '13121012000', '13121011800',
            '13121011900', '13121001202', '13121001201', '13121001001', '13121004300',
            '13121001002', '13121002100', '13121002600', '13121002800', '13121003500',
            '13121003800', '13121004400', '13121004800', '13121004900', '13121005800'
        ],
        "thresholds": {
            '13121001800': 15.05, '13121001900': 6.65,  '13121003600': 0,
            '13121012000': 0,     '13121011800': 0.77,  '13121011900': 0,
            '13121001202': 0,     '13121001201': 0,     '13121001001': 36.52,
            '13121004300': 1.75,  '13121001002': 0,     '13121002100': 0,
            '13121002600': 0,     '13121002800': 4.55,  '13121003500': 3.13,
            '13121003800': 0,     '13121004400': 6.172, '13121004800': 5.09,
            '13121004900': 0,     '13121005800': 10.41,
            'default': 1.0  # Default GA threshold if tract not in list
        },
        # GA uses the simple solver for these 3 tracts
        "simple_solver_tracts": ['13121004800','13121004900','13121012000']
    },
    "CA": {
        # This data is from webpage 5_plot_2024_2030_CA.ipynb
        "tract_ids": [
            '06037206020', '06037206200', '06037206300', '06037207101', '06037207102',
            '06037207200', '06037207301', '06037207302', '06037207403', '06037207404',
            '06037207405', '06037207406', '06037207501', '06037207502', '06037207710',
            '06037207720', '06037207800', '06037208001', '06037208002', '06037208100',
            '06037208200', '06037208301', '06037208302', '06037209200'
        ],
        "thresholds": {
            '06037208301': 4.0,
            '06037209200': 6.0,
            'default': 0.0  # Default CA threshold (from notebook)
        },
        # CA uses the complex solver (_solve_for_k_mfd) for all tracts
        "simple_solver_tracts": []
    },
    "NY": {
        # This data is from webpage 5_plot_2024_2030_NY.ipynb
        "tract_ids": [
            '36061003602', '36061005502', '36061004800', '36061003200', '36061004100',
            '36061006300', '36061005900', '36061003002', '36061003400', '36061005000',
            '36061001800', '36061004000', '36061002201', '36061002202', '36061002601',
            '36061002602', '36061002800', '36061003001', '36061003601', '36061003800',
            '36061004200', '36061004300', '36061006100', '36061004500', '36061005501', 
            '36061005700', '36061006400', '36061006500', '36061007100', '36061005200'
        ],
        "thresholds": {
            '36061003602': 0,     '36061005502': 3.5,   '36061004800': 1.5,
            '36061003200': 2.5,   '36061004100': 9.2,   '36061006300': 4.5,
            '36061005900': 0,     '36061003002': 0,     '36061003400': 7.5,
            '36061005000': 0,     '36061001800': 0,     '36061004000': 0,
            '36061002201': 0,     '36061002202': 7.22,  '36061002601': 0.74,
            '36061002602': 0,     '36061002800': 0,     '36061003001': 0,
            '36061003601': 9.31,  '36061003800': 0,     '36061004200': 0,
            '36061004300': 0,     '36061006100': 8.34,  '36061004500': 6.5,
            '36061005501': 7.79,  '36061005700': 0,     '36061006400': 0.5,
            '36061006500': 0,     '36061007100': 5.96,  '36061005200': 0,
            'default': 1.0 # Default NY threshold
        },
        # NY uses the complex solver for all tracts
        "simple_solver_tracts": []
    },
    "WA": {
        # This data is from webpage 5_plot_2024_2030_WA.ipynb
        "tract_ids": [
            '53033011402', '53033008200', '53033008002', '53033010500', '53033026500', 
            '53033008400', '53033008300', '53033008500', '53033009100', '53033011300', 
            '53033011500', '53033026600', '53033011401', '53033009800', '53033009900', 
            '53033010800', '53033010701', '53033008100', '53033008600', '53033009200',  
            '53033009300', '53033012000', '53033010702'
        ],
        "thresholds": {
            '53033011402': 0,     '53033008200': 0,     '53033008002': 3.0,
            '53033010500': 0,     '53033026500': 6.0,   '53033008400': 0,
            '53033008300': 0,     '53033008500': 0,     '53033009100': 0,
            '53033011300': 0,     '53033011500': 0,     '53033026600': 0,
            '53033010702': 6.2,   '53033011401': 0,     '53033009800': 0,
            '53033009900': 10.81, '53033010800': 0,     '53033010701': 0,
            '53033008100': 0,     '53033008600': 0,     '53033009200': 0,
            '53033009300': 1.71,  '53033012000': 0,
            'default': 1.0 # Default WA threshold
        },
        # WA uses the complex solver for all tracts
        "simple_solver_tracts": []
    }
}
# --- End of Configurations ---


# --- Helper function from your notebook ---
def _q_k_function(k, lam, vf, Qm, kj, w):
    """Internal helper function for the MFD model."""
    return -lam * np.log(
        np.exp(-vf * k / lam) +
        np.exp(-Qm / lam) +
        np.exp(-(kj - k) * w / lam)
    )

# --- Helper function for MFD solver ---
def _solve_for_k_mfd(q, prev_speed, prev_k, prev_q, max_q, lam, vf, Qm, kj, w, is_before_max_q):
    """Internal helper function for the MFD solver."""
    def root_function(k):
        return _q_k_function(k, lam, vf, Qm, kj, w) - q
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        k_solution = fsolve(root_function, 10)[0]

    if k_solution <= 0: return np.nan
    if q >= Qm:
        speed = q / k_solution if k_solution > 0 else np.nan
        if prev_speed is not None and prev_k is not None and prev_q is not None:
            if is_before_max_q:
                if q > prev_q: speed = prev_speed * (prev_q / q)
                elif q < prev_q: speed = prev_speed * (q / prev_q)
            else:
                if q < prev_q: speed = prev_speed * (prev_q / q)
                elif q > prev_q: speed = prev_speed * (q / prev_q)
            k_solution = q / speed if speed > 0 else np.nan
    return k_solution

# --- Helper function for simple solver ---
def _solve_for_k_simple(q, lam, vf, Qm, kj, w):
    """Internal helper function for the simple solver."""
    def root_function(k):
        return _q_k_function(k, lam, vf, Qm, kj, w) - q
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        k_solution = fsolve(root_function, 10)
        
    return k_solution[0] if k_solution[0] > 0 else np.nan


# --- Main function to be called by Flask ---
def run_traffic_analysis(params_file_obj, city_name, year):
    """
    Runs the traffic density and speed calculation for a given state and year.
    
    :param params_file_obj: The uploaded parameters file object (from request.files).
    :param city_name: The city name (e.g., "Atlanta", "Los Angeles").
    :param year: The year as a string (e.g., "2030").
    """
    try:
        # 1. Determine State and get config
        state = CITY_TO_STATE.get(city_name)
        if not state:
            return {"status": "error", "message": f"City '{city_name}' not found in CITY_TO_STATE map."}
        
        if state not in STATE_CONFIGS:
            return {"status": "error", "message": f"Configuration for state '{state}' not found in traffic_processor.py"}
        
        # Get the correct configuration for the determined state
        config = STATE_CONFIGS[state]

        # Build dynamic paths based on state
        base_dir = os.path.dirname(os.path.abspath(__file__))
        input_excel_file = os.path.join(base_dir, 'Combined_sheet_Input_volume', state, f'{state}_{year}.xlsx')
        output_excel_file = os.path.join(base_dir, 'Combined_sheet_output_speed', state, f'{state}_{year}.xlsx')

        if not os.path.exists(input_excel_file):
            return {"status": "error", "message": f"Input file not found: {input_excel_file}"}

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_excel_file)
        os.makedirs(output_dir, exist_ok=True)

        # 2. Load parameters (from the uploaded file)
        parameters_df = pd.read_csv(params_file_obj)
        # Fix: Use the actual Greek letter lambda for renaming
        parameters_df = parameters_df.rename(columns={
            'λ': 'lam', 'v_f': 'vf', 'Q_m': 'Qm', 'k_j': 'kj', 'w': 'w'
        }).set_index('Tract ID')

        required_parameters = ['lam', 'vf', 'Qm', 'kj', 'w']

        # 3. Get state-specific processing lists from config
        tract_ids_to_process = config["tract_ids"]
        state_thresholds = config["thresholds"]
        simple_solver_list = config["simple_solver_tracts"]


        # 4. Process the file
        excel_file = pd.ExcelFile(input_excel_file)
        sheet_names = excel_file.sheet_names
        processed_sheets = {}

        for sheet_name in sheet_names:
            if 'tract=' in sheet_name:
                tract_id = sheet_name.split('tract=')[1]
            else:
                tract_id = sheet_name
            
            # Check if the tract is in our list AND has parameters
            if tract_id in tract_ids_to_process and int(tract_id) in parameters_df.index:
                print(f"Processing tract: {tract_id}")
                
                tract_parameters = parameters_df.loc[int(tract_id)][required_parameters].to_dict()
                Qm = tract_parameters["Qm"]

                traffic_volume_data = pd.read_excel(input_excel_file, sheet_name=sheet_name)
                traffic_volume_data['Adjusted Traffic Volume'] = traffic_volume_data['Traffic Volume'].copy()
                traffic_volume = traffic_volume_data['Adjusted Traffic Volume'].values

                # 5. Get dynamic threshold from config
                threshold = state_thresholds.get(tract_id, state_thresholds['default'])
                
                # --- This adjustment logic is now FIXED to match the notebooks ---
                qm_indices = np.where(traffic_volume >= Qm)[0]
                if len(qm_indices) > 0:
                    first_qm_idx = qm_indices[0]
                    before_qm_indices = np.where(np.abs(traffic_volume[:first_qm_idx] - Qm) <= threshold)[0]
                    if len(before_qm_indices) > 0:
                        # FIX 1: Use [-1] to get the last point BEFORE congestion
                        traffic_volume[before_qm_indices[-1]] = Qm
                        qm_indices = np.insert(qm_indices, 0, before_qm_indices[-1])
                
                    after_qm_indices = np.where(np.abs(traffic_volume[first_qm_idx:] - Qm) <= threshold)[0] + first_qm_idx
                    if len(after_qm_indices) > 0:
                        traffic_volume[after_qm_indices[-1]] = Qm
                        qm_indices = np.append(qm_indices, after_qm_indices[-1])
                
                # FIX 2: Replaced the custom interpolation with the notebook's linear interpolation
                if len(qm_indices) > 1:
                    for j in range(len(qm_indices) - 1):
                        start_idx, end_idx = qm_indices[j], qm_indices[j + 1]
                        if end_idx - start_idx > 1:
                            # This is the linear interpolation logic from the notebooks
                            x = [start_idx, end_idx]
                            y = [traffic_volume[start_idx], traffic_volume[end_idx]]
                            f = interp1d(x, y)
                            for k in range(start_idx + 1, end_idx):
                                traffic_volume[k] = f(k)
                
                max_q_index = np.argmax(traffic_volume)
                # --- End of adjustment logic ---

                # --- Dynamic Density and Speed Calculation ---
                calculated_density = []
                calculated_speed = []
                prev_speed, prev_q, prev_k = None, None, None

                if tract_id in simple_solver_list:
                    # Use simple solver
                    calculated_density = [_solve_for_k_simple(q, **tract_parameters) for q in traffic_volume]
                    calculated_speed = [q / k if k > 0 else np.nan for q, k in zip(traffic_volume, calculated_density)]
                else:
                    # Use MFD (complex) solver
                    for i, q in enumerate(traffic_volume):
                        is_before_max_q = i <= max_q_index
                        k = _solve_for_k_mfd(q, prev_speed, prev_k, prev_q, traffic_volume[max_q_index], **tract_parameters, is_before_max_q=is_before_max_q)
                        speed = q / k if k > 0 else np.nan
                        calculated_density.append(k)
                        calculated_speed.append(speed)
                        prev_speed, prev_q, prev_k = speed, q, k
                
                traffic_volume_data['Density'] = calculated_density
                traffic_volume_data['Speed'] = calculated_speed
                
                processed_sheets[sheet_name] = traffic_volume_data
                print(f"Completed tract: {tract_id} (Sheet: {sheet_name})")

        # 6. Write to output file
        with pd.ExcelWriter(output_excel_file, engine='openpyxl') as writer:
            for sheet_name, df in processed_sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        result_message = f"Processing complete! Output saved to: {output_excel_file}. Total sheets processed: {len(processed_sheets)}"
        print(result_message)
        return {"status": "success", "message": result_message, "output_file": output_excel_file}
    
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        return {"status": "error", "message": f"File not found: {e.filename}"}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"status": "error", "message": str(e)}