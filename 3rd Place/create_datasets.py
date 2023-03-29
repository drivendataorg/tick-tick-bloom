import pandas as pd, numpy as np
import sys,os,shutil,gc,re,json,glob,math,time,random


ROOT_DIR = 'data'
DATA_DIR_RAW = f'{ROOT_DIR}/raw'
DATA_DIR_INTERIM = f'{ROOT_DIR}/interim'
DATA_DIR_PROCESSED = f'{ROOT_DIR}/processed'
os.makedirs(DATA_DIR_PROCESSED,exist_ok=True)

META_DIR = f'{DATA_DIR_RAW}/meta'
DIR_LSAT_TRAIN = f'{DATA_DIR_INTERIM}/train/lsat'
DIR_SNEL_TRAIN = f'{DATA_DIR_INTERIM}/train/snel'
DIR_LSAT_TEST = f'{DATA_DIR_INTERIM}/test/lsat'
DIR_SNEL_TEST = f'{DATA_DIR_INTERIM}/test/snel'

NSAMPLES_LSAT=1
NSAMPLES_SNEL=15 

meta = pd.read_csv(f'{META_DIR}/metadata.csv')
meta['longitude']=(meta['longitude']/10).round(0) #reduce drift

grid_mapping = pd.read_csv(f'{DATA_DIR_INTERIM}/hrrr/uid_grid_mapping.csv')

hrrr_vars = {'tmp':'t2m','spfh':'sh2'}
for var,var_name in hrrr_vars.items():
  d = [pd.read_csv(p) for p in glob.glob(f'{DATA_DIR_INTERIM}/hrrr/{var}/*')]
  d = pd.concat(d)
  print(len(d))
  d=pd.merge(d,grid_mapping,on=['x','y','date'])
  print(len(d))
  d=d.groupby(['uid'])[[var_name]].agg({var_name:['min','mean','max']})
  d.columns = ['_'.join(col) for col in d.columns.values]
  d=d.reset_index()
  meta=pd.merge(meta,d,on=['uid'],how='left')
  

labels = pd.read_csv(f'{META_DIR}/train_labels.csv')
df_train = labels.merge(meta,how='left',on='uid').sort_values(by='uid').reset_index(drop=True)

ss=pd.read_csv(f'{META_DIR}/submission_format.csv')
df_test = meta[meta.split=='test'].copy()
df_test = ss.merge(df_test,on='uid')
df_test['severity']=-1
stats = ['min','mean','max']
forecast_vars=[f'{v}_{s}' for v in hrrr_vars.values() for s in stats]

def calc_sample_quality(path):
  x=np.load(path)['arr_0']
  if x.mean()<1:
    return 0
  return 1
  
def load_dataset(dataset_path,nsamples,df):
  data=[pd.read_csv(p) for p in glob.glob(f'{dataset_path}/meta*.csv')]
  data=pd.concat(data)
  data=data[data.platform!='landsat-7'] 
  data['path'] = f'{dataset_path}/'+data.uid+'_'+data.datetime+'_'+data.platform+'_'+data.ix.astype(str)+'.npz'
  data['month'] = data.datetime.str.split('-').str[1].astype(int)
  data['quality'] = data.path.apply(calc_sample_quality)
  print(data.shape[0],data.uid.nunique())
  data=data.sort_values(by=['quality','cloud_cover','datetime'],ascending=[False,True,False]).groupby('uid').head(nsamples).reset_index(drop=True)
  data=pd.merge(df,data,on='uid').sort_values(by='uid').reset_index(drop=True)
  data['days_before_sample'] = (data.date.astype(np.datetime64)-data.datetime.str.split('_').str[0].astype(np.datetime64)).dt.days
  return data
df_lsat_train=load_dataset(DIR_LSAT_TRAIN,NSAMPLES_LSAT,df_train)
df_snel_train=load_dataset(DIR_SNEL_TRAIN,NSAMPLES_SNEL,df_train)
print('lsat: ',df_lsat_train.shape, df_snel_train.shape)
# uids=df_lsat_train.uid.tolist()+df_snel_train.uid.tolist()
# uids=list(set(uids))
# print(len(uids))
# df_train[~df_train.uid.isin(uids)]
df_lsat_test=load_dataset(DIR_LSAT_TEST,NSAMPLES_LSAT,df_test)
df_snel_test=load_dataset(DIR_SNEL_TEST,NSAMPLES_SNEL,df_test)
print('snel: ', df_lsat_test.shape, df_snel_test.shape)
# uids=df_lsat_test.uid.tolist()+df_snel_test.uid.tolist()
# uids=list(set(uids))
# print(len(uids))
# df_test[~df_test.uid.isin(uids)]
df_lsat_train.to_csv(f'{DATA_DIR_PROCESSED}/lsat_train.csv',index=False)
df_snel_train.to_csv(f'{DATA_DIR_PROCESSED}/snel_train.csv',index=False)

df_lsat_test.to_csv(f'{DATA_DIR_PROCESSED}/lsat_test.csv',index=False)
df_snel_test.to_csv(f'{DATA_DIR_PROCESSED}/snel_test.csv',index=False)

df_train.to_csv(f'{DATA_DIR_PROCESSED}/train.csv',index=False)
df_test.to_csv(f'{DATA_DIR_PROCESSED}/test.csv',index=False)
