import pandas as pd
import seaborn as sns
import numpy as np

import glob
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KDTree
import tqdm
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.decomposition import PCA
import os

import pickle
import warnings
warnings.filterwarnings('ignore')


def pseudo_round(x):
    if x < 1.65:
        return 1
    elif x < 2.55:
        return 2
    elif x < 3.5:
        return 3
    elif x < 4.5:
        return 4
    else:
        return 5


lgb_params = {'application': 'regression',
          'boosting': 'gbdt',
          'metric': 'rmse',
          'learning_rate': 0.005,
          'bagging_fraction': 0.3,
          'feature_fraction': 0.3,
          'min_split_gain': 0.1,
          'verbosity': -1,
          'data_random_seed': 2023
}



if __name__ == '__main__': 
    # load input and nrrr data
    dfs = []
    files = glob.glob('../downloaded_data/nrrr/*.csv')
    for f in files:
        dfs.append(pd.read_csv(f, index_col=0))
    temp = pd.concat(dfs)

    temp = temp[~temp.id.duplicated()]

    df = pd.read_csv('../inputs/metadata.csv')
    train_label = pd.read_csv('../inputs/train_labels.csv')

    train_df = df[df.split == 'train'].copy()
    train = train_df.merge(train_label, on='uid', how='inner')

    train['month'] = pd.DatetimeIndex(train.date).month
    train['year'] = pd.DatetimeIndex(train.date).year

    train = train.merge(temp, left_on='uid', right_on='id', how='inner').drop(labels='id', axis=1)

    test_df = df[df.split == 'test'].copy()
    test_meta = pd.read_csv('../inputs/submission_format.csv')

    test_df = test_df.merge(test_meta, on='uid')

    test_df['month'] = pd.DatetimeIndex(test_df.date).month
    test_df['year'] = pd.DatetimeIndex(test_df.date).year

    test = test_df.merge(temp, left_on='uid', right_on='id', how='inner').drop(labels='id', axis=1)
    
    # load water color data
    water_color = pd.read_csv('../outputs/Sentinels_available_features.csv')

    train = train.merge(water_color[
        ['uid', 'R', 'G', 'B', 'GMax', 'GMin', 'G_R', 'G_B', 'R_B', 'GMax_B', 'GMin_B']
    ], how='left', on='uid')

    test = test.merge(water_color[
        ['uid', 'R', 'G', 'B', 'GMax', 'GMin', 'G_R', 'G_B', 'R_B', 'GMax_B', 'GMin_B']
    ], how='left', on='uid')

    train_full = train.copy()
    
    
    if not os.path.exists('../outputs/weights'):
        os.makedirs('../outputs/weights')

    test_pred_list = []
    rmses = []
    models = []
    for i in range(100):
        # northeast', 'midwest
        train = train_full[train_full.region.isin(['northeast', 'midwest'])].copy()
        km = KMeans(n_clusters=100)
        train['cluster'] = km.fit_predict(train[['longitude', 'latitude']].values).astype(str)
        train['fold'] = -1
        gkf = GroupKFold(n_splits=5)
        for idx, (trn, val) in enumerate(gkf.split(train, groups=train.cluster)):
            train.iloc[val, -1] = idx
        verbose_eval = 100000
        num_rounds = 30000
        early_stop = 500
        test_preds = []

        oofs = []
        for f in range(5):
            trn_data = train[(train.fold != f)].drop(
                labels=['latitude', 'longitude', 'year', 'cluster', 'fold', 'uid', 'date', 'split', 'severity', 'density'], axis=1).copy()
            trn_label = train[(train.fold != f)].severity
            val_data = train[(train.fold == f)].drop(
                labels=['latitude', 'longitude', 'year', 'cluster', 'fold', 'uid', 'date', 'split', 'density'], axis=1).copy()
            test_data = test.drop(['latitude', 'longitude', 'year', 'uid', 'date', 'split', 'severity'], 1).copy()

            trn_data['region'] = trn_data['region'].map({
                'midwest': 0,
                'south': 1,
                'northeast': 2,
                'west': 3
            })

            val_data['region'] = val_data['region'].map({
                'midwest': 0,
                'south': 1,
                'northeast': 2,
                'west': 3
            })

            test_data['region'] = test_data['region'].map({
                'midwest': 0,
                'south': 1,
                'northeast': 2,
                'west': 3
            })


            d_train = lgb.Dataset(trn_data, label=trn_label.values, categorical_feature=['region'])
            d_valid = lgb.Dataset(val_data.drop(labels='severity', axis=1),
                                  label=val_data.severity, categorical_feature=['region'])


            model = lgb.train(lgb_params, d_train, num_boost_round=num_rounds, valid_sets=d_valid,
                                 early_stopping_rounds=early_stop, verbose_eval=verbose_eval)

            val_pred = model.predict(val_data.drop(labels='severity', axis=1))
            val_data['pred'] = np.round(val_pred).astype(np.int)
            val_data['raw_pred'] = val_pred

            test_pred = model.predict(test_data)
            test_preds.append(test_pred)
            oofs.append(val_data)

            model.save_model(f'../outputs/weights/model_i{i}_f{f}.bin')

        oof = pd.concat(oofs)
        rmses.append(np.sqrt(mean_squared_error(oof.severity, oof.pred)))
        test_pred_list.append(test_preds)
        
    test_pred_flat = []
    for e in test_pred_list:
        test_pred_flat.extend(e)

    test['severity'] = np.stack(test_pred_flat, -1).mean(1)#).astype(int)
    lgb_test = test[['uid', 'region', 'severity']].copy()

    print(np.mean(rmses))
    
    kdt = KDTree(train_full[['latitude', 'longitude']].values, leaf_size=30, metric='euclidean')
    distance, matches = kdt.query(test_df[['latitude', 'longitude']].values, k=100, return_distance=True)
    pred = []
    for i, x in enumerate(matches):
        pred.append((train_full.iloc[x].severity * (1 / distance[i])).sum() / (1 / distance[i]).sum())
    test_df['severity'] = pred
    knn_test = test[['uid', 'region', 'severity']].copy()
    
    with open('../outputs/weights/kdt.bin', 'wb') as fp:
        pickle.dump([kdt, train_full], fp)
        
    mg_test = pd.concat([lgb_test[lgb_test.region.isin(['midwest', 'northeast'])],
                   knn_test[~knn_test.region.isin(['midwest', 'northeast'])]])
    
    
    mg_test['severity'] = mg_test.apply(lambda x: pseudo_round(x.severity), 1)
    test_df['severity'] = test_df.apply(lambda x: int(np.round(x.severity)), 1)
    
    sub = pd.concat([
    test_df[~test_df.uid.isin(mg_test[mg_test.region.isin(['northeast', 'midwest'])].uid)][
        ['uid', 'region', 'severity']], mg_test[mg_test.region.isin(['northeast', 'midwest'])]])
    
    sub.set_index('uid').loc[test_df.uid].to_csv('../outputs/submission.csv')