'''
This script generates the DEM data
run on planetary computer
have trouble running on personal
machine and hitting rate limiting
'''

from src import get_data
from src.get_data import db_con
from src.elevation import elevation_point
from datetime import datetime
import pandas as pd



def add_elevation(data,lat='latitude',lon='longitude',con=db_con,name='elevation_dem'):
    uid = data['uid'].tolist()
    lat_l = data[lat].tolist()
    lon_l = data[lon].tolist()
    iv = 0
    tot_n = data.shape[0]
    res = []
    for u,la,lo in zip(uid,lat_l,lon_l):
        iv += 1
        #print(f'Getting {iv} out of {tot_n} @ {datetime.now()}')
        ele_dat = elevation_point(la,lo,1000)
        ele_dat['uid'] = u
        res.append(ele_dat.copy())
    res_df = pd.DataFrame(res)
    get_data.add_table(res_df,name,con)
    return res_df


def get_elevation(data,con=db_con,name='elevation_dem',chunk=100):
    # If table exists, only worry about getting new information
    if get_data.tab_exists(name,con):
        upd = get_data.get_update(data,name,con)
        if upd.shape[0] > 0:
            print(f'Updating {upd.shape[0]} records')
            upd_chunks = get_data.chunk_pd(upd,chunk)
            print(f'Chunk size is {upd_chunks[0].shape[0]}')
            for i,ud in enumerate(upd_chunks):
                print(f'Getting chunk {i+1} out of {len(upd_chunks)} @ {datetime.now()}')
                ele_data = add_elevation(data=ud,name=name)
        else:
            print('No new records to append to elevation dem table')
    else:
        print('elevation_dem table does not exist, add in stats')
        upd_chunks = get_data.chunk_pd(data,chunk)
        print(f'Chunk size is {upd_chunks[0].shape[0]}')
        for i,ud in enumerate(upd_chunks):
            print(f'Getting chunk {i+1} out of {len(upd_chunks)} @ {datetime.now()}')
            ele_data = add_elevation(data=ud,name=name)
    return None

# sometimes missing data, can delete out
if get_data.tab_exists('elevation_dem'):
    db_con.execute('DELETE FROM elevation_dem WHERE elevation = -99999')

meta = pd.read_csv('./data/metadata.csv')
get_elevation(meta)
res = pd.read_sql('SELECT * FROM elevation_dem',db_con)
res.to_csv('./data/elevation_dem.csv',index=False)