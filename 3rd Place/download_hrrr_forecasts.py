import pandas as pd, numpy as np
import os,sys,shutil,gc,re,json,glob,math,time,random,io
from pqdm.processes import pqdm
from datetime import date, datetime, timedelta
import xarray as xr
import cfgrib
import requests
from herbie import Herbie,FastHerbie

NJOBS = 4
short_name = sys.argv[1]
level = sys.argv[2]

# short_name,level = ['PRATE','surface']
# short_name,level = ['PRES','surface']
# short_name,level = ['DPT','2 m above ground']
# short_name,level = ['TMP','2 m above ground']
# short_name,level = ['SPFH','2 m above ground']
# short_name,level = ['GUST','surface']
# short_name,level = ['UGRD','1000 mb']
# short_name,level = ['VGRD','1000 mb']
# short_name,level = ['VIS','surface']

ROOT_DIR = 'data'
DATA_DIR_RAW = f'{ROOT_DIR}/raw'
DATA_DIR_INTERIM = f'{ROOT_DIR}/interim'
out_dir = f'{DATA_DIR_INTERIM}/hrrr/{short_name.lower()}'
os.makedirs(out_dir,exist_ok=True)
META_DIR = f'{DATA_DIR_RAW}/meta'

meta = pd.read_csv(f'{META_DIR}/metadata.csv',parse_dates=['date']).sort_values(by='date')
meta.loc[meta.longitude<0,'longitude'] = meta[meta.longitude<0].longitude+360
meta = meta[meta.date>'2014-09-30'].reset_index(drop=True)
grid_mapping = pd.read_csv(f'{DATA_DIR_INTERIM}/hrrr/uid_grid_mapping.csv',parse_dates=['date'])
meta = pd.merge(meta,grid_mapping)
meta = meta[['date','x','y']].drop_duplicates().reset_index(drop=True)

query_ts = []
dates = []
# for date in list(set(meta.date.tolist())):
for date in meta.date.unique():
    ts = [d for d in pd.date_range(start=date,periods=12,freq="-1H",closed = None)] + [d for d in pd.date_range(start=date,periods=12,freq="1H",closed = None)]
    ts=sorted(list(set(ts)))
    # query_ts[row.date] = ts
    query_ts += ts
    dates +=[date for i in range(len(ts))]

df_qry_dates = pd.DataFrame(dict(date=dates,query_ts=query_ts))
df_qry_dates = df_qry_dates[df_qry_dates.query_ts>='2014-10-01'].drop_duplicates(subset='query_ts').reset_index(drop=True)
dates = df_qry_dates.date.drop_duplicates().tolist()



def extract_data(date):
    query_ts=df_qry_dates[df_qry_dates.date==date].query_ts.tolist()
    FH = FastHerbie(query_ts, model="hrrr", fxx=[0])
    # FH.objects
    ds = FH.xarray(f"{short_name}:{level}", remove_grib=False).drop_vars(['step','gribfile_projection','time','longitude','latitude'])
    ds = ds.to_dataframe().reset_index()
    ds=pd.merge(ds,meta[meta.date==date],on=['x','y'])
    ds.to_csv(f"{out_dir}/{str(date.strftime('%Y-%m-%d'))}.csv",index=False)
    


args = dates
results = pqdm(args, extract_data, n_jobs=NJOBS)
