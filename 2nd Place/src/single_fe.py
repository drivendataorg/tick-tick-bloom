'''
This is just intended as illustration
To generate the features used in the
model based on a single input
from other functions
'''

from .elevation import elevation_point
from .sat_fe import get_image_data_ll
from .feat import cluster, reg_ord
from . import mod
import pandas as pd

def get_features(lat,lon,date,region):
    fin_dat = {}
    ele_dat = elevation_point(lat,lon)
    img_dat = get_image_data_ll(lat,lon,date)
    # if landsat-7, filter to all missing
    # need to change imtype = 1 for sentinel
    # -1 for everything else
    if img_dat['imtype'] == 'land_sat':
        im2 = {k:-1 for k in img_dat.keys()}
    else:
        img2 = img_dat.copy()
        img2['imtype'] = 1
    clus = {'cluster': cluster([lat,lon])}
    fin_dat.update(ele_dat)
    fin_dat.update(img2)
    fin_dat.update(clus)
    fin_dat['region'] = reg_ord[region]
    fin_dat['latitude'] = lat
    fin_dat['longitude'] = lon
    fin_dat['date'] = date
    return fin_dat


# UID aabm
#res = get_features(39.080319,-86.430867,'2018-05-14','midwest')
#print(res)

# No default model, need to load in yourself
#best_model = 'mod_2023_02_06'
#best_model = mod.load_model(best_model)

# Function to predict entirely out of sample
# loads in the model object inside function
# not super-efficient
def pred_out(lat,lon,date,region,model):
    di = get_features(lat,lon,date,region)
    pdat = pd.DataFrame([di])
    pred = model.predict_int(pdat)
    return pred[0]

# UID aabm
#pred = pred_out(39.080319,-86.430867,'2018-05-14','midwest')
#print(pred)