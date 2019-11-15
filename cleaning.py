import numpy as np
import pandas as pd
import sklearn as sk
import scipy as sp
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

drops = ["CRASH_CRN", "DISTRICT", "CRASH_COUNTY", "MUNICIPALITY",
         "POLICE_AGCY", "CRASH_YEAR", "TIME_OF_DAY", "TOTAL_UNITS",
         "LATITUDE", "LONGITUDE", "EST_HRS_CLOSED", "LANE_CLOSED",
         "LN_CLOSE_DIR", "NTFY_HIWY_MAINT", "FLAG_CRN", "VEHICLE_TOWED",
         "PSP_REPORTED", "ROADWAY_CRN", "RDWY_SEQ_NUM", "ADJ_RDWY_SEQ",
         "ACCESS_CTRL", "ROADWAY_COUNTY", "ROAD_OWNER", "ROUTE",
         "SEGMENT", "OFFSET", "TOT_INJ_COUNT", "SCHOOL_BUS_UNIT"]

sev_metric = ["INJURY", "FATAL", "MAJOR_INJURY", "FATAL_COUNT",
              "INJURY_COUNT", "MAJ_INJ_COUNT", "MOD_INJ_COUNT", 
              "MIN_INJ_COUNT", "UNK_INJ_DEG_COUNT", "UNK_INJ_PER_COUNT", 
              "UNB_DEATH_COUNT", "UNB_MAJ_INJ_COUNT", "BELTED_DEATH_COUNT",
              "BELTED_MAJ_INJ_COUNT", "MCYCLE_DEATH_COUNT", 
              "MCYCLE_MAJ_INJ_COUNT", "BICYCLE_DEATH_COUNT", "BICYCLE_MAJ_INJ_COUNT",
              "PED_COUNT", "PED_DEATH_COUNT", "PED_MAJ_INJ_COUNT", "MAX_SEVERITY_LEVEL",
              "FATAL_OR_MAJ_INJ", "INJURY_OR_FATAL", "MINOR_INJURY", "MODERATE_INJURY", 
              "MAJOR_INJURY", "TOT_INJ_COUNT"]


loc_metric = ["DEC_LAT", "DEC_LONG"]

yn_columns = ["SCH_BUS_IND", "SCH_ZONE_IND", "NTFY_HIWY_MAINT", 
              "TFC_DETOUR_IND", "WORK_ZONE_IND"]

direction = {"N": 1, "E": 2, "S": 3, "W": 4, "U": 0, "": 0}

def drop_highly_correlated_features(data):
    # Create correlation matrix
    corr_matrix = data.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.9
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

    for entry in to_drop:
        print("Dropping highly correlated column:", entry)

    # Drop features 
    return data.drop(data[to_drop], axis=1)


def yn_to_bool(data, columns):
    for column in columns:
        if column in data.columns:
            data[column] = pd.Series(np.where(data[column] == 'Y', 1, 0))
    return data


def drop_missing_vals(data):
    percent_missing = data.isnull().mean()
    missing_val_cols = percent_missing[percent_missing > 0.10].index
    for col in missing_val_cols:
        print("Dropping column with missing values:", col)
    return data.drop(data[missing_val_cols], axis=1)

def fix_lat_long(data):
    if "RDWY_ORIENT" in data.columns:
        data["RDWY_ORIENT"] = data["RDWY_ORIENT"].map(direction)
    return data[data["DEC_LONG"] < -79]

def clean(data):
    data = data.drop(data[drops], axis=1) # drop manually choosen columns
    data = drop_missing_vals(data) # drop cols with a lot of missing vals
    data = yn_to_bool(data, yn_columns) # change Y/N to bool values
    data = drop_highly_correlated_features(data) # drop highly corr features
    data = fix_lat_long(data) # drop rows that aren't in pittsburg
    
    data_info(data)
    return data


def get_data(file):
    return pd.read_csv(file)


def plot_accidents(data):
    boundingBox = (data.DEC_LONG.min(),data.DEC_LONG.max(), data.DEC_LAT.min(), data.DEC_LAT.max())
    background = plt.imread('C:\\Users\\Chris\\Dev\\Pummels-in-Pittsburg\\pittsburgh.png')
    fig, ax = plt.subplots(figsize = (8,7))
    ax.scatter(data.DEC_LAT, data.DEC_LAT, zorder=1, alpha= 0.2, c='b', s=10)
    ax.set_title('Plotting Crashes on Pittsburgh')
    ax.set_xlim(boundingBox[0],boundingBox[1])
    ax.set_ylim(boundingBox[2],boundingBox[3])
    ax.imshow(background, zorder=0, extent = boundingBox, aspect= 'equal')


def data_info(data):
    print(data.head())
    print(data.shape)
    plot_accidents(data)


def get_clean_data():
    df = get_data("crash.csv")
    df = clean(df)
    return df

def main():
    get_clean_data()

main()
