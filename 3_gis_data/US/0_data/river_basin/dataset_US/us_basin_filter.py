import glob
import os
import numpy as np
import pandas as pd

dataset = "camels" # "caravan"

gis_dir = "/data0/funato/hsp2/3_gis_data/US/0_data/river_basin/dataset_US"
input = f"{gis_dir}/us_{dataset}_file_num_full.csv"
output = f"{gis_dir}/us_{dataset}_file_num.csv"

varssim_dir = f"/data0/funato/hsp2/2_MERV/{dataset}/varssim_US"

data_cols = ["Obs flow", "Precip", "Temp", "PET"]

# Metadata columns (lon/lat/area/HUC/region assignments etc.) already worked out
# for the shared 671-basin CONUS CAMELS-US gauge set.
full_columns = ["File_num", "File_num_all", "gauge_id", "lon", "lat", "area",
                 "camels_fil", "HUC", "ARID", "OgRegion", "Region1", "Region"]
region_metadata_source = f"{gis_dir}/us_caravan_file_num_full.csv"
camels_topo_file = "/data0/funato/hsp2/camels_us/camels_topo.txt"


def generate_full_basin_list(dataset, varssim_dir, output_path):
    """Read the dataset's varssim directory (every basin we actually have data
    for) and build the unfiltered 'file_num_full' gauge list, joining in
    lon/lat/area/HUC/region metadata from the existing CONUS gauge tables
    where available."""
    print(f"{output_path} not found - generating full basin list for '{dataset}' "
          f"from {varssim_dir} ...")

    gauge_ids = sorted(
        os.path.basename(f)[len("varssim_"):-len(".csv")]
        for f in glob.glob(f"{varssim_dir}/varssim_*.csv")
    )
    if not gauge_ids:
        raise RuntimeError(f"No varssim_*.csv files found in {varssim_dir}")

    df_meta = pd.read_csv(region_metadata_source)
    df_meta["gauge_id_short"] = df_meta["gauge_id"].str.split("_").str[-1]
    df_meta = df_meta.set_index("gauge_id_short")

    df_topo = pd.read_csv(camels_topo_file, sep=";", dtype={"gauge_id": str})
    df_topo = df_topo.set_index("gauge_id")

    rows = []
    for gauge_id in gauge_ids:
        if gauge_id in df_meta.index:
            row = df_meta.loc[gauge_id, ["lon", "lat", "area", "camels_fil", "HUC",
                                          "ARID", "OgRegion", "Region1", "Region"]].to_dict()
        else:
            row = {col: np.nan for col in ["lon", "lat", "area", "camels_fil", "HUC",
                                            "ARID", "OgRegion", "Region1", "Region"]}
            if gauge_id in df_topo.index:
                row["lon"] = df_topo.loc[gauge_id, "gauge_lon"]
                row["lat"] = df_topo.loc[gauge_id, "gauge_lat"]
                row["area"] = df_topo.loc[gauge_id, "area_gages2"]
        row["gauge_id"] = f"camels_{gauge_id}"
        rows.append(row)

    df_full = pd.DataFrame(rows)
    df_full.insert(0, "File_num", range(1, len(df_full) + 1))
    df_full["File_num_all"] = df_full["File_num"]
    df_full = df_full[full_columns]
    df_full.to_csv(output_path, index=False)
    print(f"Saved full basin list ({len(df_full)} basins) to {output_path}")


if dataset == "camels" and not os.path.exists(input):
    generate_full_basin_list(dataset, varssim_dir, input)

if dataset == "caravan":
    start_date_cal = "1987-01-01"
    end_date_cal = "2000-12-31"
    start_date_eva = "2001-01-01"
    end_date_eva = "2006-12-31"
elif dataset == "camels":
    start_date_cal = "1981-01-01"
    end_date_cal = "1995-12-31"
    start_date_eva = "1996-01-01"
    end_date_eva = "2010-12-31"

date_range_cal = pd.date_range(start_date_cal, end_date_cal, freq="D")
date_range_cal_eva = pd.date_range(start_date_cal, end_date_eva, freq="D")


def has_full_data(df, start, end, expected_dates):
    period = df.loc[start:end, data_cols]
    return len(period) == len(expected_dates) and period.notna().all().all()


df_basins = pd.read_csv(input)

rows_cal_eva = []

for _, row in df_basins.iterrows():
    gauge_id = row["gauge_id"].split("_")[-1]
    varssim_file = f"{varssim_dir}/varssim_{gauge_id}.csv"

    if not os.path.exists(varssim_file):
        print(f"Missing varssim file for {gauge_id}. Skipping.")
        continue

    df = pd.read_csv(varssim_file, parse_dates=["Date"]).set_index("Date")

    if has_full_data(df, start_date_cal, end_date_eva, date_range_cal_eva):
        rows_cal_eva.append(row)

pd.DataFrame(rows_cal_eva).to_csv(output, index=False)

print(f"Basins with full calibration data: {len(rows_cal_eva)} / {len(df_basins)}")
print(f"Basins with full calibration+evaluation data: {len(rows_cal_eva)} / {len(df_basins)}")
