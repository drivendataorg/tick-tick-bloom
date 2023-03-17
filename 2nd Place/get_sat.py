'''
Getting satellite imagery
'''

from src import get_data
from src.sat_fe import get_image_data_ll
import pandas as pd

def loop_image_data(data,con=get_data.db_con,table_name='sat'):
    # if table exists, check to make sure data is not already included
    if get_data.tab_exists(table_name,con):
        dc = get_data.get_update(data,table_name)
        print(f'Updating {dc.shape[0]} records in {table_name}')
    else:
        dc = data.copy()
        print(f'Making {dc.shape[0]} new records in {table_name}')
    uidl = dc['uid'].tolist()
    latl = dc['latitude'].tolist()
    lonl = dc['longitude'].tolist()
    datl = pd.to_datetime(dc['date']).tolist()
    it = 0
    for uid,lat,lon,dat in zip(uidl,latl,lonl,datl):
        it += 1
        print(f'Grabbing image {it} out of {dc.shape[0]} @ {datetime.now()}')
        try:
            res_im = get_image_data_ll(lat,lon,dat)
            res_df = pd.DataFrame([res_im])
            res_df['uid'] = uid
            get_data.add_table(res_df,table_name,con)
        except Exception:
            print(f'Unable to query image data for UID: {uid}')
    res = pd.read_sql(f'SELECT * FROM {table_name}',con)
    return res

#get_data.drop_table('sat')

meta = pd.read_csv('./data/metadata.csv')
res_df = loop_image_data(meta) # meta.sample(100) # to check
print(res_df.shape)
res_df.to_csv('./data/sat_stats.csv',index=False)



