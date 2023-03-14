import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import tqdm
import cv2
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from scipy.stats import kruskal


if __name__ == '__main__': 
    # read competition related files
    meta = pd.read_csv('../inputs/metadata.csv')
    train = meta.copy()
    train.head(2)

    train_label = pd.read_csv('../inputs/train_labels.csv')
    test_pslabel = pd.read_csv('../inputs/submission_format.csv')

    water_files = set(glob.glob('../downloaded_data/Sentinel/W_*.png'))
    all_files = set(glob.glob('../downloaded_data/Sentinel/*.png'))
    imgs = list(all_files - water_files)

    print('Downloaded Sentinels files: {}, water mask: {}, img: {}'.format(
        len(all_files), len(water_files), len(imgs)))

    df = pd.DataFrame({'files': list(water_files)})
    df['uid'] = df.files.str.split('_', expand=True)[2]

    lbls = pd.concat([test_pslabel, train_label])

    train = train.merge(df, on='uid', how='inner').merge(lbls, on='uid', how='inner')
    train['path'] = train.files.apply(lambda x: x.replace('W_', ''), 1)


    interested = train.copy()
    
    r,g,b,gmax,gmin = [], [], [], [], []
    for i, row in interested.iterrows():
        wm = imread(row.files)
        img = imread(row.path)
        water_scaled = np.stack([cv2.resize(wm, (img.shape[1], img.shape[0]))] * 3, -1) == 6
        if water_scaled.sum() == 0:
            r.append(np.nan)
            g.append(np.nan)
            b.append(np.nan)
            gmax.append(np.nan)
            gmin.append(np.nan)
        else:
            r.append(img[:, :, 0][water_scaled[:, :, 0]].mean())
            g.append(img[:, :, 1][water_scaled[:, :, 1]].mean())
            b.append(img[:, :, 2][water_scaled[:, :, 2]].mean())
            gmax.append(np.percentile(img[:, :, 1][water_scaled[:, :, 1]], 95))
            gmin.append(np.percentile(img[:, :, 1][water_scaled[:, :, 1]], 5))

    interested['R'] = r
    interested['G'] = g
    interested['B'] = b
    interested['GMax'] = gmax
    interested['GMin'] = gmin


    interested['G_R'] = interested.G / interested.R
    interested['G_B'] = interested.G / interested.B
    interested['R_B'] = interested.R / interested.B
    interested['GMax_B'] = interested.GMax / interested.B
    interested['GMin_B'] = interested.GMin / interested.B
    
    interested.to_csv('../outputs/Sentinels_available_features.csv', index=False)