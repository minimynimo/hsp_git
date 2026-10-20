import pandas as pd
import numpy as np
import os
import glob

# --- Configuration ---
loc = "US"

# Update this base path if needed
base_dir = "/data0/funato/hsp2/0_out"
output_dir = f"/data0/funato/hsp2/0_out/analysis/{loc}/summary_ranking_multi/automated_ranks"
os.makedirs(output_dir, exist_ok=True)

loc_dataset = "camels"
model_list = ["HBV", "RF", "XGB", "TCN", "LSTM"]
# Worst rank = number of models (was hard-coded 6 when BC was included)
worst_rank = len(model_list)

"""# --- Helper to map model names to file paths ---
def get_model_path_name(model_name):
    return "BC" if model_name == "BC" else model_name
"""
# ==========================================
# 1. MASS CONSERVATION RANKING (Composite)
# ==========================================
def calculate_mass_ranks():
    print("\n--- Calculating Mass Conservation Ranks (Mass Bias + ET Bias) ---")
    
    # 1.1 Calculate Mass Bias Metric
    mass_file = f"{base_dir}/analysis/{loc}/mass_conservation/mass_bias_data.csv"
    mass_bias_scores = {}
    
    if os.path.exists(mass_file):
        try:
            df_mass = pd.read_csv(mass_file)
            if 'Model' in df_mass.columns:
                df_mass['Model'] = df_mass['Model'].astype(str).str.strip().replace({'BC': 'BC'})
            
            for model in model_list:
                model_data = df_mass[df_mass['Model'] == model]
                if not model_data.empty:
                    # Metric A: Median Absolute Mass Bias
                    mass_bias_scores[model] = np.median(np.abs(model_data['Mass_Bias']))
                else:
                    mass_bias_scores[model] = np.nan
        except Exception as e:
            print(f"Error processing Mass Bias file: {e}")
            mass_bias_scores = {m: np.nan for m in model_list}
    else:
        print(f"Warning: Mass bias file not found at {mass_file}")
        mass_bias_scores = {m: np.nan for m in model_list}

    # 1.2 Calculate ET Bias Metric
    et_file = f"{base_dir}/analysis/{loc}/mass_conservation/water_balance_data.csv"
    et_bias_scores = {}
    
    if os.path.exists(et_file):
        try:
            df_et = pd.read_csv(et_file)
            if 'Model' in df_et.columns:
                df_et['Model'] = df_et['Model'].astype(str).str.strip().replace({'BC': 'BC'})
            
            # Extract Observed Data
            df_obs = df_et[df_et['Model'] == 'Observed'][['File_num', 'Daily_Mean_ET']].copy()
            df_obs.rename(columns={'Daily_Mean_ET': 'Obs_ET'}, inplace=True)
            
            if df_obs.empty:
                print("Warning: No Observed ET data found in water balance file.")
            
            for model in model_list:
                model_data = df_et[df_et['Model'] == model]
                if not model_data.empty and not df_obs.empty:
                    # Merge with Observed to get basin-by-basin comparison
                    merged = pd.merge(model_data, df_obs, on='File_num', how='inner')
                    if not merged.empty:
                        # Metric B: Median Absolute Difference in ET
                        merged['ET_Diff'] = np.abs(merged['Daily_Mean_ET'] - merged['Obs_ET'])
                        et_bias_scores[model] = merged['ET_Diff'].median()
                        print(f"{model} - Median Mass Bias: {mass_bias_scores.get(model, np.nan):.4f}, Median ET Diff: {et_bias_scores[model]:.4f}")
                    else:
                        et_bias_scores[model] = np.nan
                else:
                    et_bias_scores[model] = np.nan
        except Exception as e:
            print(f"Error processing ET file: {e}")
            et_bias_scores = {m: np.nan for m in model_list}
    else:
        print(f"Warning: Water balance file not found at {et_file}")
        et_bias_scores = {m: np.nan for m in model_list}

    # 1.3 Combine Ranks
    # Rank each metric independently (1 = Best/Lowest Error)
    s_mass = pd.Series(mass_bias_scores)
    s_et = pd.Series(et_bias_scores)
    
    rank_mass = s_mass.rank(ascending=True)
    rank_et = s_et.rank(ascending=True)
    
    print("\nIntermediate Ranks:")
    print("Mass Bias Ranks:\n", rank_mass)
    print("ET Bias Ranks:\n", rank_et)
    
    # Composite Score: Average of the two ranks
    composite_score = (rank_mass + rank_et) / 2
    
    # Final Ranking based on composite score (Low score = Better)
    final_rank = composite_score.rank(ascending=True)
    return final_rank

# ==========================================
# 2. DROUGHT CONDITION RANKING
# ==========================================
def calculate_drought_ranks():
    print("\n--- Calculating Drought Condition Ranks ---")
    drought_file = f"{base_dir}/analysis/{loc}/no_precip_discharge/no_precip_model_summary.csv"

    if not os.path.exists(drought_file):
        print(f"CRITICAL WARNING: Drought summary file not found at {drought_file}")
        return pd.Series(dtype=float)

    try:
        df = pd.read_csv(drought_file, index_col=None)

        # Fix column names if needed
        if 'Model' not in df.columns and df.index.name == 'Model':
            df = df.reset_index()
        elif 'Model' not in df.columns:
             # Fallback: assume first column
             df.rename(columns={df.columns[0]: 'Model'}, inplace=True)

        df['Model'] = df['Model'].astype(str).str.strip()
        df['Model'] = df['Model'].replace({'BC': 'BC'})
        
        scores = {}
        for model in model_list:
            model_data = df[df['Model'] == model]
            if not model_data.empty:
                # Metric: Mean Discharge during zero precip
                if 'Mean_Mean_Discharge' in df.columns:
                    val = model_data['Mean_Mean_Discharge'].iloc[0]
                elif 'Mean_Discharge' in df.columns:
                    val = model_data['Mean_Discharge'].iloc[0]
                else:
                    val = np.nan
                scores[model] = val
                print(f"{model} Mean Zero-Precip Discharge: {val:.4f}")
            else:
                scores[model] = np.nan

        return pd.Series(scores).rank(ascending=True)
        
    except Exception as e:
        print(f"Error in Drought Ranks: {e}")
        return pd.Series(dtype=float)


# ==========================================
# 3. FLOOD CONDITION RANKING (Strict Physical Validity)
# ==========================================
def calculate_flood_ranks():
    print("\n--- Calculating Flood Condition Ranks (Physical Responsiveness Tests) ---")
    print("Test 1: Peak discharge must increase with precipitation rate (0→10→30→50%) for each duration")
    print("Test 2: Time lag must increase with peak duration (1→3→5 days)")
    print(f"Failure in ANY test → Rank {worst_rank} | All tests pass → Rank 1")
    print()

    ranks = {}

    for model in model_list:
        flood_file = f"{base_dir}/{model}/{loc}/{loc_dataset}/peak_discharge/peak_discharge_comparison_{model}.csv"

        if not os.path.exists(flood_file):
            # No valid synthetic-peak predictions (NOT TESTED, not a physical failure)
            print(f"Warning: Flood file not found for {model} -> NOT TESTED (Rank {worst_rank})")
            ranks[model] = worst_rank
            continue
            
        try:
            df = pd.read_csv(flood_file)
            
            # --- TEST 1: PEAK DISCHARGE RESPONSE TO PRECIPITATION RATE ---
            # For each peak duration, check if discharge increases with rate
            test1_fails = []
            
            for peak_duration in [1, 3, 5]:
                # Get median discharge for each rate
                rate_medians = {}
                for rate in [0, 10, 30, 50]:
                    col_name = f'sim_peak_flow_peak{peak_duration}_rate{rate}'
                    if col_name in df.columns:
                        data = df[col_name].dropna().values
                        if len(data) > 0:
                            rate_medians[rate] = np.median(data)
                        else:
                            rate_medians[rate] = np.nan
                    else:
                        rate_medians[rate] = np.nan
                
                # Check monotonic increase: 0 < 10 < 30 < 50
                # Allow for small numerical errors (tolerance of 0.01 mm/d)
                tolerance = 0.01
                
                if not np.isnan(rate_medians[0]) and not np.isnan(rate_medians[10]):
                    if rate_medians[10] < rate_medians[0] - tolerance:
                        test1_fails.append(f"Peak{peak_duration}: Rate10 ({rate_medians[10]:.2f}) < Rate0 ({rate_medians[0]:.2f})")
                
                if not np.isnan(rate_medians[10]) and not np.isnan(rate_medians[30]):
                    if rate_medians[30] < rate_medians[10] - tolerance:
                        test1_fails.append(f"Peak{peak_duration}: Rate30 ({rate_medians[30]:.2f}) < Rate10 ({rate_medians[10]:.2f})")
                
                if not np.isnan(rate_medians[30]) and not np.isnan(rate_medians[50]):
                    if rate_medians[50] < rate_medians[30] - tolerance:
                        test1_fails.append(f"Peak{peak_duration}: Rate50 ({rate_medians[50]:.2f}) < Rate30 ({rate_medians[30]:.2f})")
                
                # Also check for flat/weak response (insufficient increase from 0 to 50)
                # Physically responsive models should show >15% increase in discharge for 50% precip increase
                if not np.isnan(rate_medians[0]) and not np.isnan(rate_medians[50]):
                    increase_pct = ((rate_medians[50] - rate_medians[0]) / max(rate_medians[0], 1e-6)) * 100
                    if increase_pct < 15.0:  # Less than 15% increase is considered physically unresponsive
                        test1_fails.append(f"Peak{peak_duration}: Weak/flat response (only {increase_pct:.1f}% increase from 0% to 50% rate)")
            
            test1_pass = len(test1_fails) == 0
            
            # --- TEST 2: TIME LAG RESPONSE TO PEAK DURATION ---
            # For rate 30%, check if time lag increases with peak duration
            test2_fails = []
            rate = 30
            
            lag_medians = {}
            for peak_duration in [1, 3, 5]:
                lag_col = f'sim_peak_lag_peak{peak_duration}_rate{rate}'
                if lag_col in df.columns:
                    data = df[lag_col].dropna().values
                    if len(data) > 0:
                        lag_medians[peak_duration] = np.median(data)
                    else:
                        lag_medians[peak_duration] = np.nan
                else:
                    lag_medians[peak_duration] = np.nan
            
            # Check that lag increases with duration
            # Allow for small tolerance (0.1 days)
            lag_tolerance = 0.1
            
            if not np.isnan(lag_medians[1]) and not np.isnan(lag_medians[3]):
                if lag_medians[3] < lag_medians[1] - lag_tolerance:
                    test2_fails.append(f"Lag3d ({lag_medians[3]:.2f}) < Lag1d ({lag_medians[1]:.2f})")
            
            if not np.isnan(lag_medians[3]) and not np.isnan(lag_medians[5]):
                if lag_medians[5] < lag_medians[3] - lag_tolerance:
                    test2_fails.append(f"Lag5d ({lag_medians[5]:.2f}) < Lag3d ({lag_medians[3]:.2f})")
            
            # Check for flat lag (no meaningful increase from 1d to 5d)
            if not np.isnan(lag_medians[1]) and not np.isnan(lag_medians[5]):
                lag_increase = lag_medians[5] - lag_medians[1]
                if lag_increase < 0.5:  # Less than 0.5 day increase
                    test2_fails.append(f"Flat lag (only {lag_increase:.2f}d increase from 1d to 5d peak)")
            
            test2_pass = len(test2_fails) == 0
            
            # --- FINAL DECISION ---
            if test1_pass and test2_pass:
                ranks[model] = 1
                print(f"{model} → PASS (Rank 1)")
                print(f"  ✓ Test 1: Peak discharge increases with precip rate")
                print(f"  ✓ Test 2: Time lag increases with peak duration")
            else:
                ranks[model] = worst_rank
                print(f"{model} → FAIL (Rank {worst_rank})")
                if not test1_pass:
                    print(f"  ✗ Test 1 FAILED (Discharge response):")
                    for fail_msg in test1_fails:
                        print(f"      - {fail_msg}")
                else:
                    print(f"  ✓ Test 1: Passed")
                    
                if not test2_pass:
                    print(f"  ✗ Test 2 FAILED (Time lag response):")
                    for fail_msg in test2_fails:
                        print(f"      - {fail_msg}")
                else:
                    print(f"  ✓ Test 2: Passed")
            print()

        except Exception as e:
            print(f"Error processing {model}: {e}")
            import traceback
            traceback.print_exc()
            ranks[model] = worst_rank

    return pd.Series(ranks)

# ==========================================
# MAIN EXECUTION
# ==========================================

# Calculate ranks
mass_ranks = calculate_mass_ranks()
drought_ranks = calculate_drought_ranks()
flood_ranks = calculate_flood_ranks()

# Combine into a single DataFrame
ranking_df = pd.DataFrame({
    'Model': model_list,
    'Mass_Rank': [mass_ranks.get(m, np.nan) for m in model_list],
    'Drought_Rank': [drought_ranks.get(m, np.nan) for m in model_list],
    'Flood_Rank': [flood_ranks.get(m, np.nan) for m in model_list]
})

# Handle NaNs (default to worst rank if something went wrong)
ranking_df = ranking_df.fillna(worst_rank)

print("\n" + "="*40)
print("FINAL AUTOMATED RANKS")
print("="*40)
print(ranking_df)

# Save to CSV
output_csv = f"{output_dir}/automated_plausibility_ranks.csv"
ranking_df.to_csv(output_csv, index=False)
print(f"\nSaved automated ranks to: {output_csv}")