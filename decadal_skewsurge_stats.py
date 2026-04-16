import os
import glob
import sys
import logging
import numpy as np
import pandas as pd
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_station_number(filename):
    base = os.path.basename(filename)
    match = re.search(r'station_(\d+)\.parquet$', base)
    return int(match.group(1)) if match else None

skewsurge_decadal_stats_dir = "/GTSM_ERA5_Extension/Skewsurge_decadal_stats/std_dev_statistics/"

station_files = glob.glob('/GTSM_ERA5_Extension/sksurge_parquets/*.parquet')
def main(cluster):
    try:
        logging.info(f"Processing station {cluster}")
        
        target_station = cluster
        matching_file = None
        for file in station_files:
            station_num = extract_station_number(file)
            #print(f"Checking {file} → Station: {station_num}")  # Debug print
            if station_num == target_station:
                matching_file = file
                break
        if matching_file:
            print(f"\n✅ Found and loading: {matching_file}")
            one_station = pd.read_parquet(matching_file)
        else:
            print(f"\n❌ No file found for station {target_station}")  
            
        one_station["time_diff"] = (one_station["tide_time"] - one_station["surge_time"]).abs()
        # For each surge_time, keep the row with the smallest difference
        closest_rows = one_station.loc[one_station.groupby("surge_time")["time_diff"].idxmin()]
        closest_rows = closest_rows.drop(columns="time_diff")
        one_station = closest_rows
        one_station
            
        preferred_df = one_station[['surge_time', 'skew_surge']] 
        preferred_df = preferred_df[~((preferred_df['surge_time'] >= '1950-01-01') & 
                    (preferred_df['surge_time'] <= '1950-01-04'))]
        preferred_df = preferred_df[~((preferred_df['surge_time'] >= '2021-03-09') & 
                            (preferred_df['surge_time'] < '2021-03-10'))]
        preferred_df = preferred_df.reset_index(drop=True)
        preferred_df_std = preferred_df.resample('10YS', on='surge_time').std()
        preferred_df_mean = preferred_df.resample('10YS', on='surge_time').mean()
        preferred_df_cv = (preferred_df_std / preferred_df_mean) * 100
        print (preferred_df_std)

        os.makedirs(skewsurge_decadal_stats_dir, exist_ok=True)
        np.savez(os.path.join(skewsurge_decadal_stats_dir, f"station_{cluster}_decadal_std_dev.npz"),
            Std_dev = preferred_df_std,
            Mean = preferred_df_mean,
            CV = preferred_df_cv
        )
        print ("The work is done...")
        logging.info(f"Finished processing station {cluster}")
        
    except Exception as e:
        logging.exception(f"Error processing station {cluster}: {e}")
        print(f"❌ Error: {e}")
        
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Normality_check.py <cluster> <log_file>")
        sys.exit(1)
        
    station = int(sys.argv[1])
    main(station)