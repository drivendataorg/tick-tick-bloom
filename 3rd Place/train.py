import pandas as pd, numpy as np
import sys,os,shutil,gc,re,json,glob,math,time,random,warnings
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
import sklearn.metrics as skm

import lightgbm as lgb
from sklearn import preprocessing

warnings.simplefilter(action='ignore')
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
labels = pd.read_csv(f'{META_DIR}/train_labels.csv')

df_train = pd.read_csv(f'{DATA_DIR_PROCESSED}/train.csv')
df_lsat_train=pd.read_csv(f'{DATA_DIR_PROCESSED}/lsat_train.csv')
df_snel_train=pd.read_csv(f'{DATA_DIR_PROCESSED}/snel_train.csv')
df_train.shape, df_lsat_train.shape, df_snel_train.shape
stats = ['min','mean','max']
hrrr_vars = {'tmp':'t2m','spfh':'sh2'}
forecast_vars=[f'{v}_{s}' for v in hrrr_vars.values() for s in stats]
forecast_vars
uids=df_lsat_train.uid.tolist()+df_snel_train.uid.tolist()
uids=list(set(uids))
print(len(uids))
print(len(df_train[~df_train.uid.isin(uids)]))

def split_data(df):

  kf  = StratifiedGroupKFold(n_splits=N_SPLITS,shuffle=True,random_state=RANDOM_STATE)
  df['fold'] = -1
  for fold_id, (train_index, test_index) in enumerate(kf.split(df,df.region,groups=df.uid)):
      df.loc[test_index,'fold'] = fold_id

  return df

df_lsat_train = split_data(df_lsat_train)
df_snel_train = split_data(df_snel_train)

def get_splits(df,fold):
  df_trn = df[df.fold!=fold].copy()
  df_val = df[df.fold==fold].copy()
  return df_trn,df_val
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

def get_score(df_val):
  scores=[]
  for region in df_val.region.unique():
    d=df_val[df_val.region==region]
    sc=skm.mean_squared_error(d.severity,d.pred.round(),squared=False)
    scores.append(sc)
  return np.mean(scores)


def train_model(sat,df_meta,features,params):
    oofs = []
    for region,region_meta in df_meta.groupby('region'):
        region_meta=region_meta.reset_index(drop=True)
        if sat=='lsat':
          data = get_features_lsat(region_meta)
        else:
          data = get_features_snel(region_meta)

        for fold in range(N_SPLITS):
            df_trn,df_val = get_splits(region_meta,fold)
            ixs_trn = df_trn.index.values
            ixs_val = df_val.index.values
            x_trn,y_trn = data.iloc[ixs_trn].drop(columns='severity').copy(),data.iloc[ixs_trn].severity.values
            x_val,y_val = data.iloc[ixs_val].drop(columns='severity').copy(),data.iloc[ixs_val].severity.values

            lgb_train = lgb.Dataset(x_trn, y_trn,free_raw_data=False)
            lgb_valid = lgb.Dataset(x_val, y_val, reference=lgb_train,free_raw_data=False)
            evals_result = {}
            gbm = lgb.train(params, lgb_train, valid_sets=[lgb_train,lgb_valid], valid_names=['train','valid'],
                            evals_result=evals_result,early_stopping_rounds=100, num_boost_round=100000,verbose_eval=False)

            y_val_pred=gbm.predict(x_val)
            df_val['pred']=y_val_pred
            df_val = df_val[['uid','region','platform','severity','pred']]
            sc = get_score(df_val)
            oofs.append(df_val)

            ##save model
            gbm.save_model(f"{MODEL_DIR}/{sat}_{region}_{fold}")
            print(f'fininshed training region: {region} {fold}, best score: {sc}')
    return oofs
params = {
  "objective" : "regression",
	"metric" : "rmse",
	"max_depth" : -1,
	"num_leaves" : 31,
	"learning_rate" : 0.1,
	"bagging_seed" : RANDOM_STATE,
	"verbosity" : 0,
	"seed": RANDOM_STATE}

OOFS = []


oofs = train_model(sat='lsat',df_meta=df_lsat_train,features=lsat_features,params=params)
OOFS+=oofs

oofs = train_model(sat='snel',df_meta=df_snel_train,features=snel_features,params=params)
OOFS+=oofs

d=pd.concat(OOFS)
d['sat'] = 'snel'
d.loc[d.platform.str.contains('landsat'),'sat'] = 'lsat'
print('lsat score',get_score(d[d.sat=='lsat'].groupby(['uid','region']).mean().reset_index()))
print('snel score',get_score(d[d.sat=='snel'].groupby(['uid','region']).mean().reset_index()))
val=d.groupby(['uid','region']).mean().round().reset_index()
sc=get_score(val.groupby(['uid','region']).mean().reset_index())
print(sc)
val_null = labels[~labels.uid.isin(val.uid.unique())].copy()
for region in val_null.region.unique():
  val_null.loc[val_null.region==region,'pred'] = val[val.region==region].pred.mean().round()

val = val.append(val_null)
print('final score: ',get_score(val))
print(val.pred.round().unique())
# 0.70748584005964 0.711692046347487
