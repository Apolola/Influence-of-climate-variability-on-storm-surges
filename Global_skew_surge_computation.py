import os
import glob
import sys
import logging
import gc
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import argrelextrema

def find_extrema(data, order=1): 
    """Find indices of local maxima and minima."""
    highs = argrelextrema(data.values, np.greater_equal, order=order)[0]
    lows = argrelextrema(data.values, np.less_equal, order=order)[0]
    return highs, lows

def decluster(series, order=1, window=3):
    """
    Find independent peaks in a series using local maxima and time-based declustering.
    """
    highs, _ = find_extrema(series, order=order)
    high_vals = series[highs]

    independent_peaks = []
    prev_peak = None

    for observed_peak in high_vals:
        peak_time = pd.Timestamp(observed_peak.time.values)
        window_start = peak_time - pd.Timedelta(hours=window)
        window_end = peak_time + pd.Timedelta(hours=window)

        if prev_peak and prev_peak['window_end'] >= window_start:
            if observed_peak.values > prev_peak['peak'].values:
                prev_peak = {'window_start': window_start, 'window_end': window_end, 'peak': observed_peak}
            else:
                continue
        else:
            if prev_peak:
                independent_peaks.append(prev_peak['peak'])
            prev_peak = {'window_start': window_start, 'window_end': window_end, 'peak': observed_peak}

    if prev_peak:
        independent_peaks.append(prev_peak['peak'])

    return xr.concat(independent_peaks, dim='time')

def associate_surges_to_tides(df_tides, df_surges, window=3, window2=6):
    Skew_surge_time = []
    associated_tides = []
    surge_to_tide_map = {}

    for tide_time in df_tides['time']:
        window_start = tide_time - pd.Timedelta(hours=window)
        window_end = tide_time + pd.Timedelta(hours=window)
        window_start2 = tide_time - pd.Timedelta(hours=window2)
        window_end2 = tide_time + pd.Timedelta(hours=window2)

        surge_candidates = df_surges[
            (df_surges['time'] >= window_start) &
            (df_surges['time'] <= window_end)
        ]

        if not surge_candidates.empty:
            surge_time = surge_candidates.iloc[0]['time']
            Skew_surge_time.append(surge_time)
            associated_tides.append(tide_time)
            surge_to_tide_map[surge_time] = tide_time
        else:
            surge_candidates_6hr = df_surges[
                (df_surges['time'] >= window_start2) &
                (df_surges['time'] <= window_end2)
            ]

            if len(surge_candidates_6hr) == 1:
                surge_time = surge_candidates_6hr.iloc[0]['time']
                if surge_time not in surge_to_tide_map:
                    Skew_surge_time.append(surge_time)
                    associated_tides.append(tide_time)
                    surge_to_tide_map[surge_time] = tide_time
                else:
                    prev_tide = surge_to_tide_map[surge_time]
                    if abs(tide_time - surge_time) < abs(prev_tide - surge_time):
                        idx = Skew_surge_time.index(surge_time)
                        associated_tides[idx] = tide_time
                        surge_to_tide_map[surge_time] = tide_time

            elif len(surge_candidates_6hr) == 2:
                surge_times = surge_candidates_6hr['time'].tolist()
                diffs = [abs(tide_time - st) for st in surge_times]
                min_idx = diffs.index(min(diffs))
                chosen_surge = surge_times[min_idx]

                if chosen_surge not in surge_to_tide_map:
                    Skew_surge_time.append(chosen_surge)
                    associated_tides.append(tide_time)
                    surge_to_tide_map[chosen_surge] = tide_time
                else:
                    prev_tide = surge_to_tide_map[chosen_surge]
                    if abs(tide_time - chosen_surge) < abs(prev_tide - chosen_surge):
                        idx = Skew_surge_time.index(chosen_surge)
                        associated_tides[idx] = tide_time
                        surge_to_tide_map[chosen_surge] = tide_time

    return pd.DataFrame({
        'tide_time': associated_tides,
        'surge_time': Skew_surge_time
    })

def compute_stats(df: pd.DataFrame, freq: str):
    series = df.set_index('surge_time')['skew_surge']
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)

    mean = series.resample(freq).mean().values
    std = series.resample(freq).std().values
    cv = (std / mean) * 100

    return {'mean': mean, 'std': std, 'cv': cv}

def process_station(cluster: int, Tide_dir: str, TWL_dir: str, sksurge_var: str, sksurge_parq: str, start_year: int, end_year: int):
    logging.info(f"Processing station {cluster} for years {start_year}-{end_year}")

    tide_list = []
    twl_list = []

    for year in range(start_year, end_year + 1):
        tide_paths = sorted(glob.glob(Tide_dir.format(year=year)))
        twl_paths = sorted(glob.glob(TWL_dir.format(year=year)))

        if not tide_paths:
            logging.warning(f"No tide files found for year {year}")
            continue
        if not twl_paths:
            logging.warning(f"No TWL files found for year {year}")
            continue

        with xr.open_mfdataset(tide_paths, engine="h5netcdf") as ds_tide:
            tide = ds_tide.isel(stations=cluster).tide.load()
            tide_list.append(tide)

        with xr.open_mfdataset(twl_paths, engine="h5netcdf") as ds_wl:
            wl = ds_wl.isel(stations=cluster).waterlevel.load()
            twl_list.append(wl)
        del ds_tide, ds_wl#
        gc.collect()

    full_tide = xr.concat(tide_list, dim='time')
    full_wl = xr.concat(twl_list, dim='time')

    selected_tide = decluster(full_tide)
    selected_wl = decluster(full_wl)

    df_selected_tides = selected_tide.to_dataframe().reset_index()
    df_selected_wl = selected_wl.to_dataframe().reset_index()

    assoc_df = associate_surges_to_tides(df_selected_tides, df_selected_wl)

    merged_df = assoc_df.merge(df_selected_wl, left_on='surge_time', right_on='time', suffixes=('', '_surge'))
    merged_df = merged_df.merge(df_selected_tides, left_on='tide_time', right_on='time', suffixes=('', '_tide'))
    merged_df.drop(columns=['time_tide', 'time'], inplace=True)

    merged_df['skew_surge'] = merged_df['waterlevel'] - merged_df['tide']

    os.makedirs(sksurge_parq, exist_ok=True)
    file_path = os.path.join(sksurge_parq, f"station_{cluster}.parquet")
    merged_df.to_parquet(file_path)

    ann_stats = compute_stats(merged_df, 'YS')
    dec_stats = compute_stats(merged_df, '10YS')

    os.makedirs(sksurge_var, exist_ok=True)
    np.savez(
        os.path.join(sksurge_var, f"station_{cluster}_stats.npz"),
        DT_mean=dec_stats['mean'], DT_dev=dec_stats['std'], DT_cv=dec_stats['cv'],
        AT_mean=ann_stats['mean'], AT_dev=ann_stats['std'], AT_cv=ann_stats['cv']
    )
    
    #del full_tide, full_wl, selected_tide, selected_wl
    #del df_selected_tides, df_selected_wl, assoc_df, merged_df
    #gc.collect()

    logging.info(f"Finished writing results for station {cluster}")
    return {'annual': ann_stats, 'decadal': dec_stats}

def main(cluster: int, log_path: str, start_year: int, end_year: int):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')

    sksurge_var = "/gpfs/work5/0/prjs0911/GTSM_ERA5_Extension/sksurge_global_results"
    sksurge_parq = "/gpfs/work5/0/prjs0911/GTSM_ERA5_Extension/sksurge_parquets"

    Tide_dir = "/gpfs/work5/0/prjs0911/Tidal_elevation/tide_{year}_*_v1.nc"
    TWL_dir = "/gpfs/work5/0/prjs0911/GTSM_ERA5_Extension/TWL_10mins/reanalysis_waterlevel_10min_{year}_*_v2.nc"

    try:
        logging.info(f"Starting station {cluster} for years {start_year}-{end_year}")
        process_station(cluster, Tide_dir, TWL_dir, sksurge_var, sksurge_parq, start_year, end_year)
        logging.info(f"Completed station {cluster}")
    except Exception as e:
        logging.error(f"Error processing station {cluster}: {str(e)}")
        print(f"Exception occurred: {str(e)}", file=sys.stderr)
        raise

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: Global_skew_surge_computation.py <cluster> <log_file> <start_year> <end_year>")
        sys.exit(1)

    station = int(sys.argv[1])
    log_file = sys.argv[2]
    start_year = int(sys.argv[3])
    end_year = int(sys.argv[4])
    main(station, log_file, start_year, end_year)