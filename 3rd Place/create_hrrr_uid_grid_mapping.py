import pandas as pd, numpy as np
import sys,os,shutil,gc,re,json,glob,math,time,random,io
import matplotlib.pyplot as plt
from tqdm import tqdm
from pqdm.threads import pqdm


from datetime import date, datetime, timedelta
import xarray as xr
import cfgrib
import requests
import matplotlib.pyplot as plt
import cmocean
from herbie import Herbie,FastHerbie



ROOT_DIR = 'data'
DATA_DIR_RAW = f'{ROOT_DIR}/raw'
DATA_DIR_INTERIM = f'{ROOT_DIR}/interim'
out_dir = f'{DATA_DIR_INTERIM}/hrrr'
os.makedirs(out_dir,exist_ok=True)
META_DIR = f'{DATA_DIR_RAW}/meta'
meta = pd.read_csv(f'{META_DIR}/metadata.csv',parse_dates=['date']).sort_values(by='date')
meta.loc[meta.longitude<0,'longitude'] = meta[meta.longitude<0].longitude+360
meta = meta[meta.date>'2014-09-30'].reset_index(drop=True)

dates = meta.groupby(meta.date.dt.year).head(1).append(meta.groupby(meta.date.dt.year).tail(1)).date.tolist() #sample dates
FH = FastHerbie(dates, model="hrrr", fxx=[0])
FH.objects
ds = FH.xarray("TMP:2 m", remove_grib=False)
ds = ds.to_dataframe().reset_index()
unique_locs = meta.drop_duplicates(subset=['longitude','latitude'])
unique_locs.shape
lens = []
lons = []
lats = []
xs = []
ys = []
rs = []
for _,row in tqdm(unique_locs.iterrows(),total=len(unique_locs)):
  latitude,longitude,date=row.latitude,row.longitude,row.date
  x=longitude;y=latitude
  r=0.01
  success = False
  while not success:
    minlon = x-r
    maxlon = x+r
    minlat = y-r
    maxlat = y+r
    d=ds[(ds.longitude>=minlon)&(ds.longitude<=maxlon)&(ds.latitude>=minlat)&(ds.latitude<=maxlat)]
    if len(d)>=len(dates):
      lens.append(len(d))
      d = d[d.time==dates[0]]
      lons+=[longitude for i in range(len(d))]
      lats+=[latitude for i in range(len(d))]
      xs+=d.x.tolist()
      ys+=d.y.tolist()
      rs.append(r)
      success=True
    else:
      r+=0.01

df_mapping=pd.DataFrame(dict(longitude=lons,latitude=lats,x=xs,y=ys))
df_mapping.drop_duplicates(subset=['x','y']) #3193 unique grids
df = pd.merge(df_mapping,meta,on=['longitude','latitude'])[['uid','x','y','date']]
df.to_csv(f'{out_dir}/uid_grid_mapping.csv',index=False)
