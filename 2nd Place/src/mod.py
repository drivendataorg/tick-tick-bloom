'''
Model function helpers
'''


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler
import numpy as np
import os
import pandas as pd
import pickle
from xgboost import XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMRegressor, Dataset
from sklearn.ensemble import RandomForestRegressor
#from sklearn.linear_model import ElasticNet

# Setting the global seed
np.random.seed(10)

# Just easier function to reset indices
def split(data,test_size=1200,random_state=10):
    train, test = train_test_split(data,test_size=test_size,random_state=random_state)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    return train, test

def split_weight(data,test_size=1200,weight='pred_split',random_state=10):
    test = data.sample(test_size,weights='split_pred', random_state=random_state)
    train = data[~data['uid'].isin(test['uid'])].reset_index(drop=True)
    test.reset_index(drop=True,inplace=True)
    return train, test

# Just a wrapper around sklearn
# so it returns a pandas dataframe with named
# columns
class DumEnc():
    def __init__(self,dtype=int):
        self.OHE = OneHotEncoder(dtype=dtype,
                                 handle_unknown='ignore',
                                 sparse=False)
        self.var_names = None
    def fit(self, X):
        self.OHE.fit(X)
        cats = self.OHE.categories_
        var = list(X)
        vn = []
        for v,ca in zip(var,cats):
            for c in ca:
                vn.append(f'{v}_{c}')
        self.var_names = vn
    def transform(self,X):
        res = pd.DataFrame(self.OHE.transform(X),
                           columns=self.var_names,
                           index=X.index)
        return res

def dummy_stats(values,begin_date):
    vdate = pd.to_datetime(values,errors='ignore')
    year = vdate.dt.year
    month = vdate.dt.month
    week_day = vdate.dt.dayofweek
    diff_days = (vdate - begin_date).dt.days
    # if binary, turn week/month into dummy variables
    return diff_days, week_day, month, year

def circle_stats(values,begin_date):
    vdate = pd.to_datetime(values,errors='ignore')
    within_year = vdate.dt.dayofyear
    week_day = vdate.dt.dayofweek
    # calculate sine/cosine for within year
    year_cos = np.cos(within_year*(2*np.pi/365))
    year_sin = np.sin(within_year*(2*np.pi/365))
    # calculate sine/cosine for within week
    week_cos = np.cos(week_day*(2*np.pi/7))
    week_sin = np.sin(week_day*(2*np.pi/7))
    diff_days = (vdate - begin_date).dt.days
    return diff_days, year_cos, year_sin, week_cos, week_sin


class DateEnc():
    def __init__(self,
                 begin = '1/1/2015',
                 dummy = True,
                 dum_types=['days','weekday','month']):
        self.begin = pd.to_datetime(begin)
        self.dummy = dummy
        # 'days','weekday','month','year'
        self.dum_types = dum_types
        self.cat_vars = []
        # Setting categorical variables
    def fit(self,X):
        # These are just fixed functions
        pass
    def transform(self,X):
        vars = list(X)
        res = []
        res_labs = []
        cat_labs = []
        if self.dummy:
            for v in vars:
                dd, week_day, month, year = dummy_stats(X[v],self.begin)
                if 'days' in self.dum_types:
                    res.append(dd) # this is not likely to be categorical
                    res_labs.append(f'days_{v}')
                if 'weekday' in self.dum_types:
                    res.append(week_day)
                    res_labs.append(f'weekday_{v}')
                    cat_labs.append(f'weekday_{v}')
                if 'month' in self.dum_types:
                    res.append(month)
                    res_labs.append(f'month_{v}')
                    cat_labs.append(f'weekday_{v}')
                if 'year' in self.dum_types:
                    res.append(year)
                    res_labs.append(f'year_{v}')
                    cat_labs.append(f'year_{v}')
            self.cat_vars = cat_labs
        else:
             for v in vars:
                dd, year_cos, year_sin, week_cos, week_sin = circle_stats(X[v],self.begin)
                res += [dd, year_cos, year_sin, week_cos, week_sin]
                res_labs += [f'days_{v}',f'yearcos_{v}',f'yearsin_{v}',f'weekcos_{v}',f'weeksin_{v}']
        res_df = pd.concat(res,axis=1)
        res_df.index = X.index
        res_df.columns = res_labs
        return res_df


# Spline encoding
# defaults to regular knots
# or specified locations
class SplEnc():
    def __init__(self):
        pass
    def fit(self,X):
        pass
    def transform(self,X):
        pass


class IdentEnc():
    def __init__(self):
        self.note = None
    def fit(self,X):
        pass
    def transform(self,X):
        return X.copy()


class SimpleOrdEnc():
    def __init__(self,
                 dtype=int,
                 unknown_value=-1,
                 lim_k=None,
                 lim_count=None):
        self.unknown_value = unknown_value
        self.dtype = dtype
        self.lim_k = lim_k
        self.lim_count = lim_count
        self.vars = None
        self.soe = None
    def fit(self, X):
        self.vars = list(X)
        # Now creating fit for each variable
        res_oe = {}
        for v in list(X):
            res_oe[v] = OrdinalEncoder(dtype=self.dtype,
                handle_unknown='use_encoded_value',
                        unknown_value=self.unknown_value)
            # Get unique values minus missing
            xc = X[v].value_counts().reset_index()
            xc.columns = [v, "Freq"]
            # If lim_k, only taking top K value
            if self.lim_k:
                top_k = self.lim_k - 1
                un_vals = xc.loc[0:top_k,:]
            # If count, using that to filter
            elif self.lim_count:
                un_vals = xc[xc["Freq"] >= self.lim_count].copy()
            # If neither
            else:
                un_vals = xc
            # Now fitting the encoder for one variable
            res_oe[v].fit(un_vals[[v]])
        # Appending back to the big class
        self.soe = res_oe
    # Defining transform/inverse_transform classes
    def transform(self, X):
        xcop = X[self.vars].copy()
        for v in self.vars:
            xcop[v] = self.soe[v].transform( X[[v]].fillna(self.unknown_value) )
        return xcop
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)
    def inverse_transform(self, X):
        xcop = X[self.vars].copy()
        for v in self.vars:
            xcop[v] = self.soe[v].inverse_transform( X[[v]].fillna(self.unknown_value) )
        return xcop

# You can compose multiple feature engines together
# Such as DateEnc() and DumEnc()
class ComposeFE():
    def __init__(self,
                mods):
        # expects a dictionary
        # of models
        self.mods = mods
    def fit(self,X):
        trans = X.copy()
        # goes left to right
        for k,m in self.mods.items():
            m.fit(trans)
            trans = m.transform(trans)
    def transform(self,X):
        trans = X.copy()
        for k,m in self.mods.items():
            trans = m.transform(trans)
        return trans


class FeatureEngine():
    def __init__(self,
                 ord_vars = None,
                 dum_vars = None,
                 spl_vars = None,
                 dat_vars = None,
                 ide_vars = None,
                 scale = None):
        self.fin_vars = None
        self.enc_dict = {}
        self.ord_vars = ord_vars
        self.dum_vars = dum_vars
        self.spl_vars = spl_vars
        self.dat_vars = dat_vars
        self.ide_vars = ide_vars
        self.cat_vars = None
        self.ord = SimpleOrdEnc()
        self.dum = DumEnc()
        self.dat = DateEnc()
        self.spl = SplEnc()
        self.ide = IdentEnc()
        self.scale = scale
        enc_vars = [ord_vars,dum_vars,spl_vars,dat_vars,ide_vars]
        enc_mods = [self.ord, self.dum, self.spl, self.dat, self.ide]
        for v,m in zip(enc_vars,enc_mods):
            if v is not None:
                self.enc_dict[tuple(v)] = m
    def fit(self, X):
        res = []
        for v,m in self.enc_dict.items():
            m.fit(X[list(v)])
            rf = m.transform(X[list(v)])
            res.append(rf)
        # doing a transform to know the variable names in the end
        res_df = pd.concat(res,axis=1)
        if self.scale is not None:
            self.scale.fit(res_df)
            res_df = pd.DataFrame(self.scale.transform(res_df),columns=list(res_df))
        self.fin_vars = list(res_df)
        # Adding categorical variables back in
        cat_vars = []
        if self.dum_vars is not None:
            cat_vars += self.dum.var_names
        if self.dat_vars is not None:
            cat_vars += self.dat.cat_vars
        if self.ord_vars is not None:
            cat_vars += self.ord_vars
        self.cat_vars = cat_vars
        return res_df
    def transform(self, X):
        res = []
        for v,m in self.enc_dict.items():
            res.append(m.transform(X[list(v)]))
        res_df = pd.concat(res,axis=1)
        if self.scale is not None:
            res_df = pd.DataFrame(self.scale.transform(res_df),columns=list(res_df))
        return res_df

def safelog(x):
    return np.log(x.clip(1))

def strat(values):
    edges = [np.NINF,20000,1e6,1e7,1e8,np.inf]
    labs = [1,2,3,4,5]
    res = pd.cut(values,bins=edges,labels=labs,right=False)
    return res.astype(int)


# Class to insert different types of
# regression models
class RegMod():
    def __init__(self,
                 ord_vars = None,
                 dum_vars = None,
                 dat_vars = None,
                 ide_vars = None,
                        y = None,
                transform = None,
                inv_trans = None,
                   weight = None,
                  scale_x = None,
                      mod = XGBRegressor(n_estimators=100, max_depth=3)):
        self.fe = FeatureEngine(ord_vars=ord_vars,
                           dat_vars=dat_vars,
                           dum_vars=dum_vars,
                           ide_vars=ide_vars,
                           scale = scale_x)
        self.transform = transform
        self.inv_trans = inv_trans
        self.mod = mod
        self.y = y
        self.resids = None
        self.cat_vars = None
        self.weight = weight
        self.metrics = None
        self.fit_cat = False
    def fit(self, X, weight=True, cat=True):
        if (self.weight is not None) & weight:
            sw = X[self.weight]
        else:
            sw = None
            #print('NOT using Weights in fit')
        y_dat = X[self.y].copy()
        if self.transform:
            y_dat = self.transform(y_dat)
        X_dat = self.fe.fit(X)
        self.cat_vars = self.fe.cat_vars
        # If catboost or lightgbm, pass in categories
        if (type(self.mod) == CatBoostRegressor) & cat:
            vt = list(X_dat)
            if self.cat_vars is not None:
                ci = [vt.index(c) for c in self.cat_vars]
            else:
                ci = None
            self.mod.fit(X_dat,y_dat,sample_weight=sw,cat_features=ci)
            self.fit_cat = True
        elif (type(self.mod) == LGBMRegressor) & cat:
            self.fit_cat = True
            for v in self.cat_vars:
                X_dat[v] = X_dat[v].astype('category')
            self.mod.fit(X_dat, y_dat, sample_weight=sw)
        else:
            self.mod.fit(X_dat,y_dat,sample_weight=sw)
        pred = self.mod.predict(X_dat)
        self.resids = pd.Series(y_dat - pred)
    def predict(self,X,duan=True):
        X_dat = self.fe.transform(X)
        if self.fit_cat & (type(self.mod) == LGBMRegressor):
            for v in self.cat_vars:
                X_dat[v] = X_dat[v].astype('category')
        pred = pd.Series(self.mod.predict(X_dat), X.index)
        resids = self.resids
        # if transform, do Duans smearing
        if (self.transform is not None) & duan:
            resids = resids.values.reshape(1,resids.shape[0])
            dp = self.inv_trans(pred.values.reshape(X.shape[0],1) + resids)
            pred = pd.Series(dp.mean(axis=1), X.index)
        return pred
    def predict_int(self,X):
        pred = self.predict(X)
        if self.transform:
            pred = strat(pred)
        pred = pred.clip(1,5).round().astype(int)
        return pred
    def feat_import(self):
        var_li = self.fe.fin_vars
        mod_fi = self.mod.feature_importances_
        res_df = pd.DataFrame(zip(var_li,mod_fi),columns = ['Var','FI'])
        res_df.sort_values('FI',ascending=False,inplace=True,ignore_index=True)
        # Normalize to sum to 1
        res_df['FI'] = res_df['FI']/res_df['FI'].sum()
        return res_df
    def met_eval(self, data, weight=True, cat=True, full_train=False,
                 split_tt='weighted', test_size=2000, test_splits=10, 
                 ret=False, pr=False):
        dc = data.copy()
        seeds = np.random.randint(1,1e6,test_splits)
        metrics = []
        for s in seeds:
            if split_tt == 'weighted':
                if self.weight is not None:
                    wv = self.weight
                else:
                    wv = 'pred_split'
                train, test = split_weight(dc,test_size,wv,s)
            else:
                train, test = split(dc,test_size,s)
            self.fit(train, weight, cat)
            test['pred'] = self.predict_int(test)
            met_di = rmse_region(test,ret=True)
            met_di['seed'] = s
            metrics.append(met_di.copy())
        mpd = pd.DataFrame(metrics)
        if full_train:
            self.fit(data)
        if self.metrics is None:
            self.metrics = mpd
        else:
            self.metrics = pd.concat([self.metrics,mpd],axis=0)
        if pr:
            print(mpd[['AvgError','midwest','northeast','south','west']].describe().T)
        if ret:
            return mpd['AvgError'].mean()

class CatMod():
    def __init__(self,
                 ord_vars = None,
                 dum_vars = None,
                 dat_vars = None,
                 ide_vars = None,
                        y = None,
                transform = None,
                inv_trans = None,
                  scale_x = None,
                      mod = CatBoostClassifier(iterations=100,depth=5,allow_writing_files=False,verbose=False)
                      ):
        self.fe = FeatureEngine(ord_vars=ord_vars,
                           dat_vars=dat_vars,
                           dum_vars=dum_vars,
                           ide_vars=ide_vars,
                           scale = scale_x)
        self.mod = mod
        self.y = y
        self.cat_vars = None
    def fit(self, X, sample_weight=None):
        y_dat = X[self.y].copy().astype(int)
        X_dat = self.fe.fit(X)
        self.mod.fit(X_dat,y_dat,sample_weight=sample_weight)
    def predict_proba(self,X):
        X_dat = self.fe.transform(X)
        pred_probs = self.mod.predict_proba(X_dat)
        # Turning into nicer dataframe
        cols = [f'P{str(i)}' for i in range(pred_probs.shape[1])]
        pred_probs = pd.DataFrame(pred_probs,index=X.index, columns=cols)
        return pred_probs
    def predict(self,X):
        # returns predicted probability
        pred = self.predict_proba(X)
        return pred["P1"]
    def feat_import(self):
        var_li = self.fe.fin_vars
        mod_fi = self.mod.feature_importances_
        res_df = pd.DataFrame(zip(var_li,mod_fi),columns = ['Var','FI'])
        res_df.sort_values('FI',ascending=False,inplace=True,ignore_index=True)
        # Normalize to sum to 1
        res_df['FI'] = res_df['FI']/res_df['FI'].sum()
        return res_df

# If you pass in multiple models
# this will ensemble them, presumes regressor models
class EnsMod():
    def __init__(self, mods, av_func = 'mean'):
        self.mods = mods #should be dict
        self.av_func = av_func
    def fit(self,X,weight=True,cat=True):
        for key,mod in self.mods.items():
            mod.fit(X,weight=weight,cat=cat)
    def predict(self,X):
        res = []
        for key,mod in self.mods.items():
            res.append(mod.predict(X))
        res_df = pd.concat(res,axis=1)
        if self.av_func == 'mean':
            pred = res_df.mean(axis=1)
        return pred
    def predict_int(self,X):
        pred = self.predict(X)
        pred = pred.clip(1,5).round().astype(int)
        return pred

def rmse_region(data,pred='pred',true='severity',region='region',scale=False,ret=False):
    dc = data[[region,true,pred]].copy()
    if scale:
        dc[pred] = strat(dc[pred])
    dc[pred] = dc[pred].round().astype(int).clip(1,5)
    dc[region] = dc[region].replace({1:'northeast',
                                     2:'south',
                                     3:'midwest',
                                     4:'west'})
    dc['sq_error'] = (dc[true] - dc[pred])**2
    gr_val = dc.groupby(region,as_index=False)['sq_error'].mean()
    gr_val['root_mse'] = np.sqrt(gr_val['sq_error'])
    avg_error = gr_val['root_mse'].mean()
    if ret:
        regions = gr_val['region'].tolist()
        regions.append('AvgError')
        vals = gr_val['root_mse'].tolist()
        vals.append(avg_error)
        gr_di = {r:v for r,v in zip(regions,vals)}
        return gr_di
    else:
        print(f'\nAverage error {avg_error:.4f}')
        print('\nRegion Error')
        print(gr_val[['region','root_mse']])

def save_model(mod,name):
    fname = f'./models/{name}.pkl'
    outfile = open(fname,"wb")
    pickle.dump(mod,outfile)
    outfile.close()

def load_model(name):
    fname = f'./models/{name}.pkl'
    infile = open(fname, "rb")
    mod = pickle.load(infile)
    infile.close()
    return mod


# function to check if similar to any past submissions
def check_similar(current):
    files = os.listdir("./submissions")
    for fi in files:
        old = pd.read_csv(f"./submissions/{fi}")
        dif = np.abs(current['severity'] - old['severity']).sum()
        if dif == 0:
            print(f'Date {fi} same as current')


def check_day(current,day="sub_2023_01_31.csv"):
    old = pd.read_csv(fr"./submissions/{day}")
    dstr = f'dif_{day[4:-4]}'
    current[dstr] = old['severity'] - current['severity']
    print(current[dstr].value_counts())
