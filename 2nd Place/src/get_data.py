'''
Functions to get different data sources
and cache them in sqllite DB
'''

import requests
import numpy as np
import pandas as pd
import sqlite3
from io import StringIO
import time
import os
from datetime import datetime, timedelta
from math import sin, cos, tan, acos, atan, atan2, radians, sqrt, pi
import rioxarray
import planetary_computer as pc
import geopy.distance as distance
import odc.stac
#import cv2


db = './data/data.sqlite'
db_con = sqlite3.connect(db)

# having a hard time installing opencv on
# planetary computer, only use this function
# so far

def cv2_norm(matrix):
    amin = matrix.min()
    amax = matrix.max()
    modif = amax/(amax - amin)
    shift_mat = np.round((matrix - amin)*modif)
    return shift_mat.astype(int)


# chunking pandas dataframe
def chunk_pd(df, chunk_size):
    nrows = df.shape[0]
    tot_split = np.ceil(nrows/chunk_size)
    split_rows = np.array_split(np.arange(nrows), tot_split)
    fin_list_df = [df.iloc[s, :] for s in split_rows]
    return fin_list_df

# Get table names
def get_table_names(con=db_con):
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    res = pd.read_sql(query,con)
    return res['name'].tolist()

# Drop table if needed
def drop_table(table,con=db_con):
    dt = f'DROP TABLE {table}'
    con.execute(dt)

# Check if table exists
def tab_exists(table,con=db_con):
    rt = get_table_names(con)
    res = table in rt
    return res

# Add a table with DateTime appended
def add_table(data,tab_name,con=db_con):
    dn = data.copy()
    dn['DateTime'] = pd.to_datetime('now',utc=True)
    dn.to_sql(tab_name,index=False,if_exists='append',con=con)

# Seeing to update data based on old table
def get_update(data,table_name,con=db_con):
    query = f'SELECT DISTINCT uid FROM {table_name};'
    uids = pd.read_sql(query,con=con)['uid'].tolist()
    notin_old = ~data['uid'].isin(uids)
    return data[notin_old].copy()

# Adds in competition meta data
def add_meta_data(con=db_con):
    ret_dat = {}
    name_map = {'meta': './data/metadata.csv',
                'labels': './data/train_labels.csv',
                'format': './data/submission_format.csv'}
    for n,l in name_map.items():
        if tab_exists(n):
            query = f'SELECT * from {n}'
            d = pd.read_sql(query,con=con)
            ret_dat[n] = d
        else:
            d = pd.read_csv(l)
            add_table(d,n,con)
            ret_dat[n] = d
    return ret_dat

# Just to make sure those tables are populated
meta_data = add_meta_data()

###########################################################################################
# VINCENTY FORMULA FOR DISTANCE BETWEEN LAT/LON

# The function below is  based on transcribing and adapting code from
# http://www.movable-type.co.uk/scripts/latlong-vincenty.html
# which includes the note
#I offer these formulae & scripts for free use and adaptation as my contribution to the open-source 
#info-sphere from which I have received so much. You are welcome to re-use these scripts 
#[under a simple attribution license, without any warranty express or implied] 
#provided solely that you retain my copyright notice and a link to this page.[above]
#If you have any queries or find any problems, contact me at ku.oc.epyt-elbavom@oeg-stpircs.
#
#(c) 2002-2011 Chris Veness 

#Another useful source is
#http://search.cpan.org/~bluefeet/GIS-Distance-0.01001/lib/GIS/Distance/Vincenty.pm

# ellipsoid parameters
class Elp:
    a = 6378137.
    b = 6356752.314245
    f = (a-b) / a

def ellipseDist(lat1, lon1, lat2, lon2, inradians=False):
    """Return the distance in meters between the two sets of coordinates  using an ellipsoidal approximation to the earth 
    
    Calculation uses Vincenty inverse ellipsoid formula with WGS-84 ellipsoid parameters.
    
    lat1 and lon1 are the latitude and longitude of the first point and lat2, lon2 of the second.
    if inradians is True, locations are in radians; otherwise they are in degrees.
    If the algorithm fails to converge, the return value is None, which will become SYSMIS when returned to SPSS
    
    The ellipsoid model is more accurrate than the simpler spherical approximation."""
    
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return None
    if not inradians:
        lat1, lon1, lat2, lon2 = [radians(ang) for ang in [lat1, lon1, lat2, lon2]]
    u1, u2 = [atan((1-Elp.f) * tan(x)) for x in [lat1, lat2]]
    sinu1, sinu2 = sin(u1), sin(u2)
    cosu1, cosu2 = cos(u1), cos(u2)
    londif = lon2 - lon1
    
    lam = londif
    lamp = 2 * pi
    numiter = 50  # maximum number of iterations
    
    for i in range(numiter):
        if abs(lam - lamp) <= 1e-12:
            break
        sinLam = sin(lam)
        cosLam = cos(lam)
        sinSigma = sqrt((cosu2 * sinLam) **2 + (cosu1 * sinu2-sinu1 * cosu2 * cosLam)**2)
        if sinSigma == 0:
            return 0
        cosSigma = sinu1 * sinu2 + cosu1 * cosu2 * cosLam
        sigma = atan2(sinSigma, cosSigma)
        sinAlpha = cosu1 * cosu2 * sinLam / sinSigma
        cosSqAlpha = 1 - sinAlpha **2
        try:
            cos2SigmaM = cosSigma - 2 * sinu1 * sinu2 / cosSqAlpha
        except:
            cos2SigmaM = 0 # equatorial line
        c = (Elp.f/16) * cosSqAlpha * (4 + Elp.f * (4 - 3 * cosSqAlpha))
        lamp = lam
        lam = londif + (1 - c)  * Elp.f * sinAlpha\
            * (sigma + c * sinSigma* (cos2SigmaM + c * cosSigma * (-1 + 2 * cos2SigmaM **2)))
    else:
            return None
        
    uSq = cosSqAlpha * (Elp.a * Elp.a - Elp.b * Elp.b) / (Elp.b * Elp.b)
    A = 1 + uSq / 16384 * (4096+uSq * (-768 + uSq * (320-175 * uSq)))
    B = uSq/1024 * (256 + uSq * (-128 + uSq * (74-47 * uSq)))
    deltaSigma = B * sinSigma * (cos2SigmaM + B / 4 * (cosSigma * (-1+2 * cos2SigmaM **2)- \
        B / 6 * cos2SigmaM * (-3 + 4 * sinSigma **2) * (-3 + 4 * cos2SigmaM **2)))
    s = Elp.b * A * (sigma - deltaSigma)
    return s

# need to apply this over a vector, makes it ok with pandas dataframe
# returns in kilometers now
def vector_vincent(x, lat2, lon2):
    return ellipseDist(x[0], x[1], lat2, lon2, inradians=False)/1000

#lat = [35,36,37,38]
#lon = [-72,-71,-72,-71]
#test_df = pd.DataFrame(zip(lat,lon),columns=['lat','lon'])
#dist = test_df.apply(vector_vincent,args=(35.1,-72.1),axis=1)

###########################################################################################


# Average outcome in data from sites within d kilometers (default 300)
# Limit to only those before outcome
def spatial_lag(pred_data,data,distance=[100,300,1000],fields=['severity','logDensity'],lat='latitude',lon='longitude',date='date'):
    res = [] # final list with info
    # prepping data to check
    dc = data[fields + [lat,lon,date]].copy()
    dc[date] = pd.to_datetime(dc[date])
    # now prepping pred_data
    plon_li = pred_data[lon].tolist()
    plat_li = pred_data[lat].tolist()
    pdat_li = pd.to_datetime(pred_data[date]).tolist()
    for plon,plat,pdat in zip(plon_li,plat_li,pdat_li):
        loc_dat = []
        count_dat = []
        sub_dc = dc[dc[date] < pdat].copy()
        km_dist = sub_dc[[lat,lon]].apply(vector_vincent,args=(plat,plon),axis=1)
        for d in distance:
            sub_dist = sub_dc[km_dist < d].copy()
            count_dat.append(sub_dist.shape[0])
            for v in fields:
                loc_dat.append(sub_dist[v].mean())
        res.append(loc_dat + count_dat)
    res_labs = []
    for d in distance:
        for v in fields:
            res_labs.append(f'{v}_{str(d)}')
    res_labs += [f'count_{str(d)}' for d in distance]
    res_pd = pd.DataFrame(res,columns=res_labs,index=pred_data.index)
    return res_pd.fillna(-1)


def get_spatiallag(con=db_con,table_name='spat_lag'):
    # get full metadata table
    full_dat = pd.read_csv('./data/metadata.csv')
    # get metadata with only labels
    labs = pd.read_csv('./data/train_labels.csv')
    only_lab = labs.merge(full_dat,on='uid')
    only_lab['logDensity'] = np.log(only_lab['density'].clip(1))
    # create spatial lag
    lag_df = spatial_lag(full_dat,only_lab)
    lag_df['uid'] = full_dat['uid']
    # save to a new table
    add_table(lag_df,table_name)
    return lag_df


#########################################################################
# helper functions from DataDriven post
# https://drivendata.co/blog/tick-tick-bloom-benchmark


def get_bounding_box(latitude, longitude, meter_buffer=1000):
    """
    Given a latitude, longitude, and buffer in meters, returns a bounding
    box around the point with the buffer on the left, right, top, and bottom.

    Returns a list of [minx, miny, maxx, maxy]
    """
    distance_search = distance.distance(meters=meter_buffer)
    # calculate the lat/long bounds based on ground distance
    # bearings are cardinal directions to move (south, west, north, and east)
    min_lat = distance_search.destination((latitude, longitude), bearing=180)[0]
    min_long = distance_search.destination((latitude, longitude), bearing=270)[1]
    max_lat = distance_search.destination((latitude, longitude), bearing=0)[0]
    max_long = distance_search.destination((latitude, longitude), bearing=90)[1]
    return [min_long, min_lat, max_long, max_lat]


# get our date range to search, and format correctly for query
def get_date_range(date, time_buffer_days=15):
    """Get a date range to search for in the planetary computer based
    on a sample's date. The time range will include the sample date
    and time_buffer_days days prior

    Returns a string"""
    datetime_format = "%Y-%m-%dT"
    range_start = pd.to_datetime(date) - timedelta(days=time_buffer_days)
    date_range = f"{range_start.strftime(datetime_format)}/{pd.to_datetime(date).strftime(datetime_format)}"
    return date_range


def crop_sentinel_image(item, bounding_box):
    """
    Given a STAC item from Sentinel-2 and a bounding box tuple in the format
    (minx, miny, maxx, maxy), return a cropped portion of the item's visual
    imagery in the bounding box.

    Returns the image as a numpy array with dimensions (color band, height, width)
    """
    (minx, miny, maxx, maxy) = bounding_box

    image = rioxarray.open_rasterio(pc.sign(item.assets["visual"].href)).rio.clip_box(
        minx=minx,
        miny=miny,
        maxx=maxx,
        maxy=maxy,
        crs="EPSG:4326",
    )
    # Should I return X/Y as well?
    return cv2_norm(image.to_numpy())


def crop_landsat_image(item, bounding_box):
    """
    Given a STAC item from Landsat and a bounding box tuple in the format
    (minx, miny, maxx, maxy), return a cropped portion of the item's visual
    imagery in the bounding box.

    Returns the image as a numpy array with dimensions (color band, height, width)
    """
    (minx, miny, maxx, maxy) = bounding_box

    image = odc.stac.stac_load(
        [pc.sign(item)], bands=["red", "green", "blue"], bbox=[minx, miny, maxx, maxy]
    ).isel(time=0)
    image_array = image[["red", "green", "blue"]].to_array().to_numpy()

    # normalize to 0 - 255 values
    #image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX)
    image_array = cv2_norm(image_array)

    return image_array

#########################################################################