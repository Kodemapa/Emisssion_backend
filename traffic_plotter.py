import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import interp1d
import io
from run import CITY_TO_STATE # Import the city mapping from your run.py

# --- Main function to be called by Flask ---
def generate_plot_image(city_name, year):
    """
    Generates the traffic speed/volume plot for a given state and year
    and returns it as a PNG image in a memory buffer.
    """
    try:
        # 1. Determine State and build dynamic paths
        state = CITY_TO_STATE.get(city_name)
        if not state:
            return {"status": "error", "message": f"City '{city_name}' not found in CITY_TO_STATE map."}

        base_dir = os.path.dirname(os.path.abspath(__file__))
        excel_file_path = os.path.join(base_dir, 'Combined_sheet_output_speed', state, f'{state}_{year}.xlsx')

        if not os.path.exists(excel_file_path):
            raise FileNotFoundError(f"Data file not found. Run processing first: {excel_file_path}")

        # --- Logic from your notebook starts here ---
        tract_ids = []
        all_data = []

        excel_file = pd.ExcelFile(excel_file_path)
        sheet_names = excel_file.sheet_names

        time_axis = None # Store a sample 'time' for x-axis ticks

        for sheet_name in sheet_names:
            data = pd.read_excel(excel_file_path, sheet_name=sheet_name)
            if 'tract=' in sheet_name:
                tract_id = sheet_name.split('tract=')[1]
            else:
                tract_id = sheet_name
            tract_ids.append(tract_id)
            time = data['Time']
            if time_axis is None:
                time_axis = time # Store the first time axis found
            traffic_volume = data['Traffic Volume']
            speed = data['Speed']
            all_data.append((time, traffic_volume, speed, tract_id))
        if time_axis is None or len(all_data) == 0:
            raise ValueError("No data found to plot in the Excel file.")
        colormap = plt.get_cmap('tab20')
        colors = [colormap(i) for i in np.linspace(0, 1, len(tract_ids))]
        fig, ax1 = plt.subplots(figsize=(15, 5))
        plt.title(f'Year {year} - {city_name} ({state})', fontname='Sans Serif', fontsize=19, pad=18)
        for idx, (time, traffic_volume, speed, tract_id) in enumerate(all_data):
            x = np.arange(len(time))
            # Handle potential NaNs in speed/volume data before interpolation
            valid_speed = ~np.isnan(speed)
            valid_volume = ~np.isnan(traffic_volume)
            if np.sum(valid_speed) > 1: # Need at least 2 points to interpolate
                traffic_speed_interp = interp1d(x[valid_speed], speed[valid_speed], kind='linear', fill_value="extrapolate")
                x_new = np.linspace(0, len(time) - 1, num=len(time))
                color = colors[idx]
                ax1.plot(x_new, traffic_speed_interp(x_new), color=color, label=f'Speed - Tract {tract_id}')
            if idx == 0:
                ax2 = ax1.twinx()
                ax2.set_ylabel('Traffic Volume (veh/hr/ln)', fontname='Sans Serif', fontsize=18, labelpad=16)
            if np.sum(valid_volume) > 1:
                traffic_volume_interp = interp1d(x[valid_volume], traffic_volume[valid_volume], kind='linear', fill_value="extrapolate")
                x_new = np.linspace(0, len(time) - 1, num=len(time))
                ax2.plot(x_new, traffic_volume_interp(x_new), color=color, linestyle='--', label=f'Volume - Tract {tract_id}')
        ax1.set_xlabel('Time (hr)', fontname='Sans Serif', fontsize=18, labelpad=16)
        ax1.set_ylabel('Traffic Speed (mph)', fontname='Sans Serif', fontsize=18, labelpad=16)
        # ------------------- START OF X-AXIS LOGIC -------------------
        # Set custom x-ticks to show every hour (0, 1, 2, ..., 23)
        # 1. Define the hour labels you want to see (0 to 23 only)
        hour_labels_to_show = list(range(0, 24))  # [0, 1, ..., 23]
        # 2. Convert these hour labels into data index positions
        # (Assuming 4 data points = 1 hour)
        x_ticks = [h * 4 for h in hour_labels_to_show]
        # 3. Convert the hour labels into strings for the plot
        x_labels = [str(h) for h in hour_labels_to_show]
        ax1.set_xticks(x_ticks)
        # 4. Set font size to 14 to match Y-axis
        ax1.set_xticklabels(x_labels, fontname='Sans Serif', fontsize=14)
        # -------------------- END OF X-AXIS LOGIC --------------------
        # Set Y-axis font size
        for label in ax1.get_yticklabels():
            label.set_fontsize(14)
            label.set_fontname('Sans Serif')
        if 'ax2' in locals():
            for label in ax2.get_yticklabels():
                label.set_fontsize(14)
                label.set_fontname('Sans Serif')
        ax1.minorticks_on()
        ax1.grid(True, which='both', linestyle=':', linewidth='0.5')
        margin = 1
        # Adjust xlim to properly include the 0 and 96 tick marks
        ax1.set_xlim(left=-margin, right=len(time_axis) - 1 + margin)
        plt.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.15)
        # --- Save plot to a memory buffer ---
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        plt.close(fig)  # Close the plot to free up memory
        return img_buffer
    except Exception as e:
        print(f"An error occurred during plotting: {e}")
        return None