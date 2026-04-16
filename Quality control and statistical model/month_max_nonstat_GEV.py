import os
import glob
import signal as timesignal
import sys
import time
import logging
import re
from functools import reduce
import numpy as np
import pandas as pd
from scipy import signal
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.stattools import omni_normtest
from bluemath_tk.distributions.nonstat_gev import NonStatGEV
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy.stats import genextreme as GEV

#%%% Configure logging analysis
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

station_files = glob.glob('/GTSM_ERA5_Extension/sksurge_parquets/*.parquet')
climate_files = sorted(glob.glob('/GTSM_ERA5_Extension/Climate_indices/*.txt'))
print(climate_files)

result_nonstat_GEV_params_dir = "/GTSM_ERA5_Extension/Selected_Non_stat_GEV/monthly_max_nonstat_GEV_params/"
result_RP_nonstat_dir = "/GTSM_ERA5_Extension/Selected_Non_stat_GEV/monthly_max_nonstat_GEV_return_periods/"
result_confInt_nonstat_dir = "/GTSM_ERA5_Extension/Selected_Non_stat_GEV/monthly_max_nonstat_GEV_confidence_intervals/"
result_RP_stat_dir = "/GTSM_ERA5_Extension/Selected_Non_stat_GEV/monthly_max_stat_GEV_return_periods/"

def extract_station_number(filename):
    base = os.path.basename(filename)
    match = re.search(r'station_(\d+)\.parquet$', base)
    return int(match.group(1)) if match else None

def run_regression(X, y):
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    
    dw_test = durbin_watson(model.resid)
    omnibus_stat, omnibus_p = omni_normtest(model.resid)
    
    return model, X_const, dw_test, omnibus_p, y

def check_normality(omnibus_p):
    if omnibus_p < 0.05:
        #print("Residuals not normal (Omnibus).")
        return False  # Residuals are not normal
    else:
        #print("Residuals appear normal (Omnibus).")
        return True  # Residuals are normal
    
def summarize_model(model, X_const):
    
    param_names = X_const.columns
    
    params = pd.Series(model.params, index=param_names)
    std_errors = pd.Series(model.bse, index=param_names)
    p_values = pd.Series(model.pvalues, index=param_names)

    summary = {
        'intercept': params['const'] if 'const' in param_names else np.nan,
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj, 
        'f_statistic': getattr(model, 'fvalue', np.nan),
        'f_pvalue': getattr(model, 'f_pvalue', np.nan),
        'model_summary': model.summary()
    }

    for name in params.index:
        if name != 'const':
            summary[f'slope_{name}'] = params[name]
            summary[f'p_value_{name}'] = p_values[name]
            summary[f'CI_{name}'] = std_errors[name] * 1.96
    return summary

# Timeout handler
def timeout_handler(signum, frame):
    print("⏰ Script exceeded 40 minutes — stopping.")
    sys.exit(1)

#%%% 
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
        
        preferred_df['detrended_surge'] = signal.detrend(preferred_df['skew_surge'])
        preferred_df['Year'] = preferred_df['surge_time'].dt.year
        preferred_df['Month_name'] = preferred_df['surge_time'].dt.month_name().str[:3]
        preferred_df.set_index("surge_time", inplace=True)
        
        monthly_max = preferred_df.resample('ME')['detrended_surge'].idxmax()
        monthly_max = preferred_df.loc[monthly_max]
        print(f"Monthly max DataFrame:\n{monthly_max}")
        
        indices =['AMM', 'Nao', 'Nino1+2_anom', 'Nino3_4_anom', 'Nino3_anom', 'Nino4_anom', 'ONI', 'PNA', 'SOI', 'WHWP', 'WP', 'PDO']
        dfs = {}
        for mode, file in zip(indices, climate_files):
            df = pd.read_fwf(file, skiprows=1, header=None)
            df.columns = ["Year", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            df.set_index("Year", inplace=True)
            dfs[mode] = df
        print(len(dfs))

        melted_dfs = []
        for mode, df in dfs.items():
            df_reset = df.reset_index()  # Year back as a column
            df_long = df_reset.melt(id_vars='Year', var_name='Month_name', value_name=mode)  # value_name = mode name
            melted_dfs.append(df_long)
            
        pd.set_option('display.expand_frame_repr', False) 
        merged_indices = reduce(lambda left, right: pd.merge(left, right, on=['Year', 'Month_name']), melted_dfs)
        merged_indices['SOI'] = merged_indices['SOI'] * (-1)
        
        month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr':4, 'May':5, 'Jun':6,
                    'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
        merged_indices['Month_num'] = merged_indices['Month_name'].map(month_map)
        merged_indices['Date'] = pd.to_datetime(dict(year=merged_indices['Year'], month=merged_indices['Month_num'], day=1)) 
        merged_indices = merged_indices.sort_values('Date')
        merged_indices = merged_indices.reset_index(drop=True)

        df_merged = pd.merge(merged_indices, monthly_max, on=['Year', 'Month_name'])
        df_merged = df_merged.drop(columns=['Year','Month_name','Month_num', 'Date'])
        print(f"Merged DataFrame:\n{df_merged.head()}") 
        
#%%% Forward stepwise regression
        # Perform forward selection
        df_subset = df_merged.drop(columns=['skew_surge','detrended_surge'])
        X = df_subset
        Y = df_merged['detrended_surge']

        def forward_selection(X, y):
            selected_features = []
            while True:
                remaining_features = [f for f in X.columns if f not in selected_features]
                new_pval = pd.Series(index=remaining_features)
                for feature in remaining_features:
                    model = sm.OLS(y, sm.add_constant(X[selected_features + [feature]])).fit()
                    new_pval[feature] = model.pvalues[feature]
                min_pval = new_pval.min()
                if min_pval < 0.05:  # Adjust threshold as needed
                    selected_features.append(new_pval.idxmin())
                else:
                    break
            return selected_features
        
#%%% Select Best ENSO variable for the model       
        enso_vars = ["SOI", "Nino1+2_anom", "ONI", "Nino3_4_anom", "Nino3_anom", "Nino4_anom"]
        all_results = []

        for enso in enso_vars:
            # Take all predictors except ENSO, then add back the chosen one
            allowed_features = X.columns.drop(enso_vars).tolist()
            allowed_features.append(enso)
            X_subset = X[allowed_features]

            # Run forward selection
            selected = forward_selection(X_subset, Y)
            # Final regression on selected features
            model, X_const, dw, omnibus_p, _ = run_regression(X[selected], Y)
            is_normal = check_normality(omnibus_p)
            summary = summarize_model(model, X_const)
            
            # Store results
            result = {
                'enso_variable': enso,
                'predictors': selected,
                'r_squared': summary['r_squared'],
                'adj_r_squared': summary['adj_r_squared'],
                'f_pvalue': summary['f_pvalue'],
                'combo_num': len(selected),
                'omnibus_p': omnibus_p,
                'is_normal': is_normal,
                'aic': model.aic,
                'bic': model.bic
            }
            all_results.append(result)
        results_fwd_df = pd.DataFrame(all_results)
        print("Forward_stepwise_regression:\n", results_fwd_df)
        
#%%% Fit non-stationary GEV with selected predictors if identified as significant
        selected_predictor = results_fwd_df.loc[results_fwd_df['f_pvalue'].idxmin(), "predictors"]
        print (selected_predictor)
        
        if not selected_predictor:  # Check if list is empty
            print("No predictor explains the surge variance. Skip model fitting.")
        else:
            dfs_list = [merged_indices[col] for col in selected_predictor]
            climate_mode = pd.concat(dfs_list, axis=1)
            climate_mode.columns = selected_predictor
            print(climate_mode)

            days_in_month = {
                1: 31,
                2: 28.25,
                3: 31,
                4: 30,
                5: 31,
                6: 30,
                7: 31,
                8: 31,
                9: 30,
                10: 31,
                11: 30,
                12: 31,
            }
            days_in_month
            monthly_max["month"] = monthly_max.index.month
            monthly_max["day"] = monthly_max.index.day
            monthly_max["hour"] = monthly_max.index.hour
            monthly_max["minute"] = monthly_max.index.minute
            monthly_max["second"] = monthly_max.index.second

            monthly_max["time"] = (
                monthly_max["Year"]
                - np.min(monthly_max["Year"])
                + (monthly_max["month"] - 1) / 12
                + (monthly_max["day"] - 1) / monthly_max["month"].map(days_in_month) / 12
            )
            monthly_max

#%%% Standardize climate indices and fit non-stationary GEV
            """
            scaler = StandardScaler()
            new_scale = pd.DataFrame(scaler.fit_transform(climate_mode), columns=climate_mode.columns)
            climate_mode = new_scale
            print (climate_mode)
            """
            nonstatgev = NonStatGEV(
                xt=monthly_max["detrended_surge"].values,
                t=monthly_max["time"].values,
                covariates= climate_mode,
                harms=True,
                trends=False,
                var_name="skew_surge",
            )
            nonstatgev

#%%% Fit the GEV model using the automatic adjustment method and extract relevant parameters
            fit_results = nonstatgev.auto_adjust(stationary_shape=True)
            fit_results
            
            summary_dict = nonstatgev.summary()            
            loc_summary = summary_dict["location"]
            sc_summary = summary_dict["scale"]

            # Get only the key that isn't a Beta and Alpha term
            location_cov = [k for k in loc_summary.keys() if not k.lower().startswith("beta")]
            scale_cov = [k for k in sc_summary.keys() if not k.lower().startswith("alpha")]
            print(location_cov)
            print(scale_cov)
            
            if location_cov is None or len(location_cov) == 0:
                print("No climate indicator with significant contribution in the location parameter.")
                list_location = None
            else:
                print("The selected climate indicators contributing to the model in the location parameter are:", location_cov)
                list_location = [selected_predictor.index(c) for c in location_cov if c in selected_predictor]
                print("The indexes of the climate indicators contributing to the model in the location parameter are:", list_location)

            if scale_cov is None or len(scale_cov) == 0:
                print("No climate indicator with significant contribution in the scale parameter.")
                list_scale = None
            else:
                print("The selected climate indicators contributing to the model in the scale parameter are:", scale_cov)
                list_scale = [selected_predictor.index(c) for c in scale_cov if c in selected_predictor]
                print("The indexes of the climate indicators contributing to the model in the scale parameter are:", list_scale)
                
            if (location_cov is None or len(location_cov) == 0) and (scale_cov is None or len(scale_cov) == 0):
                print("No climate indicator with significant contribution in the model parameter, end the run")
            else:
                Location_harms = nonstatgev.beta
                scale_harms = nonstatgev.alpha

                # Count harmonic pairs
                nmu_est = len(Location_harms) // 2
                npsi_est = len(scale_harms) // 2

                # Cap at 2 if greater than 2
                nmu = min(nmu_est, 2)
                npsi = min(npsi_est, 2)

#%%% Only refit the model if at least one one of the model parameters has > 2 harmonics
                if nmu_est > 2 or npsi_est > 2:
                    print(f"Refitting nonstatgev model with nmu={nmu}, npsi={npsi}, list_loc={list_location}, list_sc={list_scale}")
                    nonstatgev.fit(
                        nmu=nmu,
                        npsi=npsi,
                        ngamma=0,
                        ntrend_loc=0,
                        list_loc=list_location,
                        ntrend_sc=0,
                        list_sc=list_scale,     
                        ntrend_sh=0,
                        list_sh=None,
                    )
                else:
                    print("No refitting — both nmu and npsi are ≤ 2.")
                    
                summary_dict = nonstatgev.summary()
                print("Model Summary:\n", summary_dict)
                
                print("Final fitted parameters:")
                
#%%% extract the relevant model parameters and save in a npz file
                loc_summary = summary_dict["location"]
                sc_summary = summary_dict["scale"]
                sh_summary = summary_dict["shape"]
                
                location_intercept = {k: v for k, v in loc_summary.items() if k.lower().startswith("beta0")}
                scale_intercept = {k: v for k, v in sc_summary.items() if k.lower().startswith("alpha0")}
                shape_intercept = {k: v for k, v in sh_summary.items() if k.lower().startswith("gamma0")}
                
                location_harms1 = {k: v for k, v in loc_summary.items() if k.lower().startswith("beta1")}
                scale_harms1 = {k: v for k, v in sc_summary.items() if k.lower().startswith("alpha1")}
                
                location_harms2 = {k: v for k, v in loc_summary.items() if k.lower().startswith("beta2")}
                scale_harms2 = {k: v for k, v in sc_summary.items() if k.lower().startswith("alpha2")}
                
                location_covariates = {k: v for k, v in loc_summary.items() if not k.lower().startswith("beta")}
                scale_covariates = {k: v for k, v in sc_summary.items() if not k.lower().startswith("alpha")}

                print("Location Intercept:\n", location_intercept)
                print("Scale Intercept:\n", scale_intercept)
                print("Shape Intercept:\n", shape_intercept)
                print("Location Harmonics 1:\n", location_harms1)
                print("Scale Harmonics 1:\n", scale_harms1)
                print("Location Harmonics 2:\n", location_harms2)
                print("Scale Harmonics 2:\n", scale_harms2)
                print("Location Covariates:\n", location_covariates)
                print("Scale Covariates:\n", scale_covariates)
                
                #save the final extrcated parameters in a npz file
                os.makedirs(result_nonstat_GEV_params_dir, exist_ok=True)
                np.savez(os.path.join(result_nonstat_GEV_params_dir, f"station_{cluster}_nonstat_GEV_params.npz"),
                    location_intercept=location_intercept,
                    scale_intercept=scale_intercept,
                    shape_intercept=shape_intercept,
                    location_harms1=location_harms1,
                    scale_harms1=scale_harms1,
                    location_harms2=location_harms2,
                    scale_harms2=scale_harms2,
                    location_covariates=location_covariates,
                    scale_covariates=scale_covariates
                )
#%%% Compute aggregated return levels and confidence intervals
# Register the timeout
                timesignal.signal(timesignal.SIGALRM, timeout_handler)
                timesignal.alarm(180 * 60)  # 40 minutes in seconds
                n_years = int(np.ceil(nonstatgev.t[-1]))
                rt_100 = np.zeros(n_years)
                rt_98 = np.zeros(n_years)
                rt_95 = np.zeros(n_years)
                rt_90 = np.zeros(n_years)
                rt_88 = np.zeros(n_years)
                rt_85 = np.zeros(n_years)
                rt_80 = np.zeros(n_years)
                rt_75 = np.zeros(n_years)
                rt_70 = np.zeros(n_years)
                rt_65 = np.zeros(n_years)
                rt_60 = np.zeros(n_years)
                rt_55 = np.zeros(n_years)
                rt_50 = np.zeros(n_years)
                rt_10 = np.zeros(n_years)
                

                for year in range(n_years):
                    rt_100[year] = nonstatgev._aggquantile(1 - 1/100, year, year + 1)  # 100-year return level
                    rt_98[year] = nonstatgev._aggquantile(1 - 1/98, year, year + 1)  # 98-year return level
                    rt_95[year] = nonstatgev._aggquantile(1 - 1/95, year, year + 1)   # 95-year return level
                    rt_90[year] = nonstatgev._aggquantile(1 - 1/90, year, year + 1)   # 90-year return level
                    rt_88[year] = nonstatgev._aggquantile(1 - 1/88, year, year + 1)   # 88-year return level
                    rt_85[year] = nonstatgev._aggquantile(1 - 1/85, year, year + 1)   # 85-year return level
                    rt_80[year] = nonstatgev._aggquantile(1 - 1/80, year, year + 1)   # 80-year return level
                    rt_75[year] = nonstatgev._aggquantile(1 - 1/75, year, year + 1)   # 75-year return level
                    rt_70[year] = nonstatgev._aggquantile(1 - 1/70, year, year + 1)   # 70-year return level
                    rt_65[year] = nonstatgev._aggquantile(1 - 1/65, year, year + 1)   # 65-year return level
                    rt_60[year] = nonstatgev._aggquantile(1 - 1/60, year, year + 1)   # 60-year return level
                    rt_55[year] = nonstatgev._aggquantile(1 - 1/55, year, year + 1)   # 55-year return level
                    rt_50[year] = nonstatgev._aggquantile(1 - 1/50, year, year + 1)   # 50-year return level
                    rt_10[year] = nonstatgev._aggquantile(1 - 1/10, year, year + 1)   # 10-year return level

                print("Aggregated 100-year return period estimates:\n", rt_100)
                print("Aggregated 98-year return period estimates:\n", rt_98)
                print("Aggregated 95-year return period estimates:\n", rt_95)
                print("Aggregated 90-year return period estimates:\n", rt_90)
                print("Aggregated 88-year return period estimates:\n", rt_88)
                print("Aggregated 85-year return period estimates:\n", rt_85)
                print("Aggregated 80-year return period estimates:\n", rt_80)
                print("Aggregated 75-year return period estimates:\n", rt_75)
                print("Aggregated 70-year return period estimates:\n", rt_70)
                print("Aggregated 65-year return period estimates:\n", rt_65)
                print("Aggregated 60-year return period estimates:\n", rt_60)
                print("Aggregated 55-year return period estimates:\n", rt_55)
                print("Aggregated 50-year return period estimates:\n", rt_50)
                print("Aggregated 10-year return period estimates:\n", rt_10)

                os.makedirs(result_RP_nonstat_dir, exist_ok=True)
                np.savez(os.path.join(result_RP_nonstat_dir, f"station_{cluster}_nonstat_GEV_RP.npz"),
                    rp_100=rt_100,
                    rp_98=rt_98,
                    rp_95=rt_95,
                    rp_90=rt_90,
                    rp_88=rt_88,
                    rp_85=rt_85,
                    rp_80=rt_80,
                    rp_75=rt_75,
                    rp_70=rt_70,
                    rp_65=rt_65,
                    rp_60=rt_60,
                    rp_55=rt_55
                    rp_50=rt_50,
                    rp_10=rt_10
                )
                '''
                # Confidence intervals for aggregated return levels
                RP = [98, 95, 90, 85, 88, 80]  # Return periods
                conf_intervals = {}
                return_levels = {}

                for r in RP:
                    print(f"Computing confidence interval for {r}-year return level...")

                    # Aggregated quantile
                    quanaggrA = nonstatgev._aggquantile(q=1 - 1/r, t0=0, t1=1)
                    quanval = 0.95  # 95% confidence level

                    # Standard deviation of quantile estimate
                    stdQuan = nonstatgev._ConfidInterQuanAggregate(1 - 1/r, 0, 1)
                    z_score = norm.ppf(1 - (1 - quanval) / 2)

                    # Upper and lower bounds
                    stdup = quanaggrA + stdQuan * z_score
                    stdlo = quanaggrA - stdQuan * z_score

                    conf_interval = (stdlo, stdup)
                    print(f"{r}-year CI: {conf_interval}")

                    conf_intervals[r] = conf_interval
                    return_levels[r] = quanaggrA

                np.savez(
                    os.path.join(result_confInt_nonstat_dir, f"station_{cluster}_nonstat_GEV_ConfInt.npz"),
                    agg_rt_levels=np.array(list(return_levels.items()), dtype=object),
                    conf_intervals=np.array(list(conf_intervals.items()), dtype=object)
                )
                '''
#%%% Compute stationary return period 
                sort_monthlymax = monthly_max.sort_values(by='detrended_surge').reset_index()
                month_surge_series = sort_monthlymax["detrended_surge"].values
                gev_params = GEV.fit(month_surge_series)
                target_periods = np.array([100, 98, 95, 90, 88, 85, 80, 75, 70, 65, 60, 55, 50, 10])
                target_values = GEV.ppf(1 - 1 / target_periods, *gev_params)
                for T, val in zip(target_periods, target_values):
                    print(f"{int(T)}-year return value: {val:.3f}")
                    
                os.makedirs(result_RP_stat_dir, exist_ok=True)              
                np.savez(os.path.join(result_RP_stat_dir, f"station_{cluster}_statgev_return_values.npz"),
                    gev_params=gev_params,
                    target_periods=target_periods,
                    target_values=target_values
                )     
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

