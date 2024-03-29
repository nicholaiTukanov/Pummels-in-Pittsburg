import numpy as np
import pandas as pd
import sklearn as sk
import scipy as sp
import matplotlib.pyplot as plt

drops = ["CRASH_CRN", "DISTRICT", "CRASH_COUNTY", "MUNICIPALITY",
         "POLICE_AGCY", "CRASH_YEAR", "TIME_OF_DAY", "TOTAL_UNITS",
         "LATITUDE", "LONGITUDE", "EST_HRS_CLOSED", "LANE_CLOSED",
         "LN_CLOSE_DIR", "NTFY_HIWY_MAINT", "FLAG_CRN", "VEHICLE_TOWED",
         "PSP_REPORTED", "ROADWAY_CRN", "RDWY_SEQ_NUM", "ADJ_RDWY_SEQ",
         "ACCESS_CTRL", "ROADWAY_COUNTY", "ROAD_OWNER", "ROUTE",
         "SEGMENT", "OFFSET", "SCHOOL_BUS_UNIT", "STREET_NAME", "RDWY_ORIENT"]

sev_metric = ["INJURY", "FATAL", "MAJOR_INJURY", "FATAL_COUNT",
              "INJURY_COUNT", "MAJ_INJ_COUNT", "MOD_INJ_COUNT", 
              "MIN_INJ_COUNT", "UNK_INJ_DEG_COUNT", "UNK_INJ_PER_COUNT", 
              "UNB_DEATH_COUNT", "UNB_MAJ_INJ_COUNT", "BELTED_DEATH_COUNT",
              "BELTED_MAJ_INJ_COUNT", "MCYCLE_DEATH_COUNT", 
              "MCYCLE_MAJ_INJ_COUNT", "BICYCLE_DEATH_COUNT", "BICYCLE_MAJ_INJ_COUNT",
              "PED_COUNT", "PED_DEATH_COUNT", "PED_MAJ_INJ_COUNT",
              "FATAL_OR_MAJ_INJ", "INJURY_OR_FATAL", "MINOR_INJURY", "MODERATE_INJURY", 
              "MAJOR_INJURY", "TOT_INJ_COUNT"]

loc_metric = ["DEC_LAT", "DEC_LONG"]

yn_columns = ["SCH_BUS_IND", "SCH_ZONE_IND", "NTFY_HIWY_MAINT", 
              "TFC_DETOUR_IND", "WORK_ZONE_IND"]

categorical_columns = ["DAY_OF_WEEK", "ILLUMINATION", "WEATHER", "RELATION_TO_ROAD", "WOK_ZONE_LOC",
                    "TCD_FUNC_CD", "URBAN_RURAL", "TCD_TYPE", "SPEC_JURIS_CD", "ROAD_CONDITION", 
                    "RDWY_SURF_TYPE_CD", "LOCATION_TYPE", "COLLISION_TYPE", "INTERSECT_TYPE"]

direction = {"N": 1, "E": 2, "S": 3, "W": 4, "U": 0, "": 0}

def drop_highly_correlated_features(data):
    # Create correlation matrix
    corr_matrix = data.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.9
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

    print("Dropping " + str(len(to_drop)) + " highly correlated columns")
    print(to_drop)

    data.style.background_gradient(cmap='coolwarm')

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
    
    print("Dropping columns with missing values > 10%:")
    print(missing_val_cols)

    return data.drop(data[missing_val_cols], axis=1)

def fix_lat_long(data):
    if "RDWY_ORIENT" in data.columns:
        data["RDWY_ORIENT"] = data["RDWY_ORIENT"].map(direction)
    return data[data["DEC_LONG"] < -79]

def get_rid_of_strs(data):
    for col in data.columns:
        if data[col].dtype == 'O':
            print(col, data[col].dtype)
    return data

def drop_rows_by_value(df, column, values):
    for value in values:
        df = df[df[column] != value]
    return df

def clean(data):
    data = data.drop(data[drops], axis=1) # drop manually choosen columns
    data = data.drop(data[sev_metric], axis=1) # drop anything having to do with the severity
    data = drop_missing_vals(data) # drop cols with a lot of missing vals
    data = yn_to_bool(data, yn_columns) # change Y/N to bool values
    data = drop_highly_correlated_features(data) # drop highly corr features
    data = fix_lat_long(data) # drop rows that aren't in pittsburg
    data = get_rid_of_strs(data) # drop cols with strings
    
    return data

def get_data(file):
    return pd.read_csv(file)

def data_info(data):
    print(data.head())
    print(data.shape)

def get_clean_data():
    df = get_data("crash.csv")
    df = clean(df)
    return df