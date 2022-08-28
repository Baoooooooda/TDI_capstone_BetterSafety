import pandas as pd
from sodapy import Socrata
import requests
import dill
import numpy as np
from math import pi


def get_pkd():
    df1, df2 = get_ndays(n1 = 7, n2 = 90)
    with open('preprocess_crime.pkd', 'wb') as f:
        dill.dump(df1, f)
    with open('preprocess_time.pkd', 'wb') as f:
        dill.dump(df2, f)
    return None



def download_from_api():
    client = Socrata('data.seattle.gov',
                 'CBIQcKwt7gh7zKhiDNwGfgcMf',
                 username="emmawangbao@gmail.com",
                 password="870616BAObao")
    seattle_gov = client.get("tazs-3rd5", limit = 1000000)
    return seattle_gov


def get_df():
    seattle_gov_df = pd.DataFrame(download_from_api())
    seattle_gov_df.offense_start_datetime = pd.to_datetime(seattle_gov_df.offense_start_datetime)
    seattle_gov_df.longitude = seattle_gov_df.longitude.astype('float')
    seattle_gov_df.latitude = seattle_gov_df.latitude.astype('float')
    return seattle_gov_df


def get_n_days_data(x, n, today = pd.to_datetime('today')):
    mask1 = (x.offense_start_datetime >= pd.to_datetime(-n, unit = 'D', origin = today))\
        & (x.offense_start_datetime < today)
    return x.loc[mask1]


def filter_group(x):
    mask = (x.group_a_b == 'A')\
    | ((x.group_a_b == 'B') & (x.offense_code == '90J')) # only take in Group A and Group B 90J
    return x.loc[mask]

def valid_location(x):
    mask = ~((x.longitude == 0) | (x.latitude == 0))
    return x.loc[mask]

def get_ndays(n1 = 7, n2 = 90):
    df = get_df()
    df_crimeplot = df.pipe(get_n_days_data, n1)\
                       .pipe(filter_group)\
                       .pipe(valid_location)

    df_time = df.pipe(get_n_days_data, n2)\
                       .pipe(filter_group)\
                       .pipe(valid_location)

    return df_crimeplot, df_time

get_pkd()
