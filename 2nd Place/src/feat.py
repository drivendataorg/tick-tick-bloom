'''
This more easily generates features
based on prepared sqlite db
for testing purposes
'''

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime

def today_str():
    now = datetime.now()
    return now.strftime('%Y_%m_%d')


reg_ord = {'west':4,
           'midwest':3,
           'south':2,
           'northeast':1}

# ordinal encoding for region
def org_reg(data,rstr='region'):
    data[rstr] = data[rstr].replace(reg_ord)

def ord_imtype(data,imstr='imtype'):
    rep_di = {'land_sat':0,
              'sentinel':1}
    data[imstr] = data[imstr].fillna(-1).replace(rep_di)


def filter_landsat(data):
    rep_di = {'land_sat':0,
              'sentinel':1}
    data['imtype'] = data['imtype'].fillna(-1).replace(rep_di)
    im_vars = ['prop_lake_500', 'r_500', 'g_500', 'b_500']
    im_vars += ['prop_lake_1000', 'r_1000', 'g_1000', 'b_1000']
    im_vars += ['prop_lake_2500', 'r_2500', 'g_2500', 'b_2500']
    im_vars += ['imtype']
    landsat = data['imtype'] == 0
    data.loc[landsat,im_vars] = -1

def safesqrt(values):
    return np.sqrt(values.clip(0))

def safelog(x):
    return np.log(x.clip(1))

def strat(values):
    edges = [np.NINF,20000,1e6,1e7,1e8,np.inf]
    labs = [1,2,3,4,5]
    res = pd.cut(values,bins=edges,labels=labs,right=False)
    return res.astype(int)

# Looking at train/test
# there are a few clusters, want
# to make sure to predict these well
def cluster(x):
    lat, lon = x[0], x[1]
    if (lat < 41) & (lon < -116):
        # cali
        return 7
    elif (lat < 41) & (lat > 36.29) & (lon < -92.9) & (lon > -102.2):
        # midwest
        return 6
    elif (lat < 38.14) & (lat > 33.26) & (lon < -74.8) & (lon > -85.52):
        # carolina
        return 2
    elif (lat < 43) & (lat > 38.7) & (lon < -75.4) & (lon > -83.55):
        # erie
        return 3
    elif (lat < 43.1) & (lat > 40.7) & (lon < -69.5) & (lon > -74.6):
        # mass
        return 4
    elif (lat < 49.6) & (lat > 41.5) & (lon < -83.55) & (lon > -104.56):
        # dakota
        return 1
    else:
        # other
        return 5

#                  1   2      2   3    3     4   4    5   5
#te_st = pd.Series([1,20000,30000,1e6,1e6+1,1e7,1e7+1,1e8,1e9])
#print(strat(te_st))

db = './data/data.sqlite'

train_query = """
SELECT 
  m.uid,
  l.region,
  l.severity,
  l.density,
  m.latitude,
  m.longitude,
  m.date,
  e.elevation,
  e.mine,
  e.maxe,
  e.dife,
  e.avge,
  e.stde,
  sl.severity_100,
  sl.logDensity_100,
  sl.count_100,
  sl.severity_300,
  sl.logDensity_300,
  sl.count_300,
  sl.severity_1000,
  sl.logDensity_1000,
  sl.count_1000,
  st.imtype,
  st.prop_lake_500,
  st.r_500,
  st.g_500,
  st.b_500,
  st.prop_lake_1000,
  st.r_1000,
  st.g_1000,
  st.b_1000,
  st.prop_lake_2500,
  st.r_2500,
  st.g_2500,
  st.b_2500
FROM meta AS m
LEFT JOIN elevation_dem AS e
  ON m.uid = e.uid
LEFT JOIN spat_lag AS sl
  ON m.uid = sl.uid
LEFT JOIN sat AS st
  ON m.uid = st.uid
LEFT JOIN labels AS l
  ON m.uid = l.uid
WHERE
  m.split = 'train'
"""

test_query = """
SELECT 
  m.uid,
  l.region,
  m.latitude,
  m.longitude,
  m.date,
  e.elevation,
  e.mine,
  e.maxe,
  e.dife,
  e.avge,
  e.stde,
  sl.severity_100,
  sl.logDensity_100,
  sl.count_100,
  sl.severity_300,
  sl.logDensity_300,
  sl.count_300,
  sl.severity_1000,
  sl.logDensity_1000,
  sl.count_1000,
  st.imtype,
  st.prop_lake_500,
  st.r_500,
  st.g_500,
  st.b_500,
  st.prop_lake_1000,
  st.r_1000,
  st.g_1000,
  st.b_1000,
  st.prop_lake_2500,
  st.r_2500,
  st.g_2500,
  st.b_2500
FROM meta AS m
LEFT JOIN elevation_dem AS e
  ON m.uid = e.uid
LEFT JOIN spat_lag AS sl
  ON m.uid = sl.uid
LEFT JOIN sat AS st
  ON m.uid = st.uid
LEFT JOIN format AS l
  ON m.uid = l.uid
WHERE
  m.split = 'test'
"""

def add_table(data,tab_name,db_str=db):
    db_con = sqlite3.connect(db_str)
    dn = data.copy()
    dn['DateTime'] = pd.to_datetime('now',utc=True)
    dn.to_sql(tab_name,index=False,if_exists='replace',con=db_con)


def get_both(db_str=db,split_pred=False):
    r1 = get_data('train',db_str,split_pred)
    r1['test'] = 0
    r1.drop(columns=['severity','density','logDensity'],inplace=True)
    r2 = get_data('test',db_str,split_pred)
    r2['test'] = 1
    res_df = pd.concat([r1,r2],axis=0)
    return res_df.reset_index(drop=True)

def get_data(data_type='train',db_str=db,split_pred=False):
    db_con = sqlite3.connect(db_str)
    if data_type == 'train':
        sql = train_query
    elif data_type == 'test':
        sql = test_query
    dat = pd.read_sql(sql,con=db_con)
    org_reg(dat) # Region ordinal encode
    # Winning solution used landsat-7 data
    #ord_imtype(dat) # image type landsat/sentinel
    filter_landsat(dat) # filtering mistake landsat-7 info
    dat = dat.fillna(-1) # missing a bit of sat data
    dat['cluster'] = dat[['latitude','longitude']].apply(cluster,axis=1)
    if data_type == 'train':
        dat['logDensity'] = safelog(dat['density'])
    if split_pred:
        pred_test = pd.read_sql('SELECT uid, pred AS split_pred FROM split_pred',con=db_con)
        dat = dat.merge(pred_test,on='uid')
    return dat


# Need logic to take predictions and get them in the right order
def sub_format(data,pred='pred'):
    form = pd.read_csv('./data/submission_format.csv')
    # some logic to transform predictions via Duan
    # smearing
    dp = data[[pred,'uid']].copy()
    dp[pred] = dp[pred].round().astype(int).clip(1,5)
    mf = form.merge(dp,on='uid')
    mf['severity'] = mf['pred']
    return mf[['uid','region','severity']]