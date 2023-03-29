import pandas as pd, numpy as np
import sys,os,shutil,gc,re,json,glob,math,time,random
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
import sklearn.metrics as skm

import lightgbm as lgb
from sklearn import preprocessing 

N_SPLITS = 5
RANDOM_STATE = 41
def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
fix_seed(RANDOM_STATE)


MODEL_DIR = 'model' 
os.makedirs(MODEL_DIR,exist_ok=True)

ROOT_DIR = 'data'
DATA_DIR_RAW = f'{ROOT_DIR}/raw'
DATA_DIR_INTERIM = f'{ROOT_DIR}/interim'
DATA_DIR_PROCESSED = f'{ROOT_DIR}/processed'
META_DIR = f'{DATA_DIR_RAW}/meta'
ss=pd.read_csv(f'{META_DIR}/submission_format.csv')
df_test = pd.read_csv(f'{DATA_DIR_PROCESSED}/test.csv')
df_lsat_test=pd.read_csv(f'{DATA_DIR_PROCESSED}/lsat_test.csv')
df_snel_test=pd.read_csv(f'{DATA_DIR_PROCESSED}/snel_test.csv')
print(df_test.shape, df_lsat_test.shape, df_snel_test.shape)

stats = ['min','mean','max']
hrrr_vars = {'tmp':'t2m','spfh':'sh2'}
forecast_vars=[f'{v}_{s}' for v in hrrr_vars.values() for s in stats]
forecast_vars
uids=df_lsat_test.uid.tolist()+df_snel_test.uid.tolist()
uids=list(set(uids))
print(len(uids))
print(len(df_test[~df_test.uid.isin(uids)]))
#lsat 
landsat_bands = ['coastal','red', 'blue','green','nir08','swir16','swir22','atran', 'cdist', 'drad', 'emis', 'emsd', 'lwir11','qa', 'qa_aerosol', 'qa_pixel', 'qa_radsat','trad', 'urad']

red=landsat_bands.index('red')
blue=landsat_bands.index('blue')
green=landsat_bands.index('green')
nir=landsat_bands.index('nir08')
swir16=landsat_bands.index('swir16')
swir22=landsat_bands.index('swir22')

sentinel_bands = ['AOT', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B11', 'B12', 'B8A', 'SCL', 'WVP']

b1=1
b2=2
b3=3
b4=4
b5=5
b6=6
b7=7
b8=8
b9=9
b11=10
b12=11
b8a=12
scl=13
wvp=14
meta_features = ['month','days_before_sample','longitude'] 
lsat_stats = [f'{b}_{c}' for c in ['mean','min','max','range'] for b in landsat_bands ]
lsat_vis = ['ndvi','ndwi','mndwi']
lsat_features = lsat_stats+meta_features+lsat_vis+forecast_vars
print(len(lsat_features))
snel_stats = [f'{b}_{c}' for c in ['mean','min','max','range'] for b in sentinel_bands ]
snel_vis = ['ndvi','ndvi_re1','ndvi_re2','ndvi_re3']
snel_features = snel_stats+meta_features+snel_vis
print(len(snel_features))
def get_features_lsat(df):
  xs=[]
  ys=[]
  for _,row in tqdm(df.iterrows(),total=len(df)):
    samp=np.load(row.path)['arr_0'].astype(np.float32)
    x=samp.mean(axis=(1,2))
    x1=samp.min(axis=(1,2))
    x2=samp.max(axis=(1,2))
    x3=x2-x1
    #satellite features
    r=np.concatenate((x,x1,x2,x3))
    
    #meta features
    r=np.append(r,[row[var] for var in meta_features])
   
    ndvi = (x[nir] - x[red]) / (x[nir] +x[red] +1)
    ndwi = (x[nir] - x[swir22]) / (x[nir] + x[swir22] +1)
    mndwi = (x[green] - x[swir22]) / (x[green] + x[swir22]+1 )
    r=np.append(r,[ndvi,ndwi,mndwi])

    ##HRRR features
    r=np.append(r, [row[var] for var in forecast_vars])

    xs.append(r)
    ys.append(row.severity)
  
  x=np.array(xs).astype(np.float32)
  y=np.array(ys).astype(np.int8)
  data=pd.DataFrame(x,columns=lsat_features)
  data['severity']=y
  return data

def get_features_snel(df):
  xs=[]
  ys=[]
  for _,row in tqdm(df.iterrows(),total=len(df)):
    samp=np.load(row.path)['arr_0'].astype(np.float32)
    
    x=samp.mean(axis=(1,2))
    x1=samp.min(axis=(1,2))
    x2=samp.max(axis=(1,2))
    x3=x2-x1

    #satellite features
    r=np.concatenate((x,x1,x2,x3))

    #meta features
    r=np.append(r,[row[var] for var in meta_features])

    ndvi=(x[b8]-x[b4])/ (x[b8]+x[b4]+1)
    ndvi_re1 = (x[b8]-x[b5])/ (x[b8]+x[b5]+1) #(nir-re1/nir+re1)
    ndvi_re2 = (x[b8]-x[b6])/ (x[b8]+x[b6]+1) #(nir-re2/nir+re2)
    ndvi_re3 = (x[b8]-x[b7])/ (x[b8]+x[b7]+1) #(nir-re3/nir+re3)
    
    r=np.append(r,[ndvi,ndvi_re1,ndvi_re2,ndvi_re3])

    xs.append(r)
    ys.append(row.severity)
    
  x=np.array(xs).astype(np.float32)
  y=np.array(ys).astype(np.int8)
  data=pd.DataFrame(x,columns=snel_features)
  data['severity']=y
  return data

def eval_model(sat,df_meta,features):
    test_preds = []
    for region,region_meta in df_meta.groupby('region'):
        region_meta=region_meta.reset_index(drop=True)
        if sat=='lsat':
          data = get_features_lsat(region_meta)
        else:
          data = get_features_snel(region_meta)    
        x_test = data.drop(columns='severity')
        for fold in range(N_SPLITS):
            gbm = lgb.Booster(model_file=f"{MODEL_DIR}/{sat}_{region}_{fold}")
            preds = region_meta[['uid','region']].copy()
            preds['severity'] = gbm.predict(x_test)
            test_preds.append(preds)	
    return test_preds
PREDS = []
preds = eval_model(sat='lsat',df_meta=df_lsat_test,features=lsat_features)
PREDS+=preds

preds = eval_model(sat='snel',df_meta=df_snel_test,features=snel_features)
PREDS+=preds

test=pd.concat(PREDS)
test=test.groupby(['uid','region']).mean().round().reset_index()

test_null = df_test[~df_test.uid.isin(test.uid.unique())].copy()[test.columns.tolist()]
print(f'filling {len(test_null)} test samples w/o data with region average')

for region in test_null.region.unique():
  test_null.loc[test_null.region==region,'severity'] = test[test.region==region].severity.mean().round()

test = test.append(test_null)
test['severity']=test.severity.astype(int)
print(test.severity.unique())
test=test.set_index('uid').reindex(ss.uid).reset_index()
assert (test.uid==ss.uid).all()

test.to_csv('solution.csv',index=False)
