import numpy as np
import pandas as pd
import sklearn as sk
import scipy as sp
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

drops = ["CRASH_CRN", "DISTRICT", "CRASH_COUNTY", "MUNICIPALITY",
         "POLICE_AGCY", "CRASH_YEAR", "TIME_OF_DAY", "TOTAL_UNITS",
         "LATITUDE", "LONGITUDE", "EST HRS_CLOSED", "LANE_CLOSED",
         "LN_CLOSE_DIR", "NTFY_HIWY_MAINT", "FLAG_CRN", "VEHICLE_TOWED"
         "PSP_REPORTED", "ROADWAY_CRN", "RDWY_SEQ_NUM", "ADJ_RDWY_SEQ"
         "ACCESS_CTRL", "ROADWAY_COUNTY", "ROAD_OWNER", "ROUTE"
         "SEGMENT", "OFFSET", "TOTAL_INJ_COUNT", "SCHOOL_BUS_UNIT"]

sev_metric = [""]

loc_metric = ["DEC_LAT", "DEC_LONG"]


def clean(data):

    # loop over columns and drop columns that have a NaN count > 50% of rows

    pass


def get_data(file):
    return pd.read_csv(file)


def main():
    df = get_data("crash.csv")
    print(df)


main()
