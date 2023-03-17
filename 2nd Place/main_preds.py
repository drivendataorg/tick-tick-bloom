'''
Main script to fit model
and generate predictions

Andy Wheeler
'''

from src import feat, mod

today = feat.today_str()
train_dat = feat.get_data(split_pred=True)

###################################
# Example just predicting severity directly

sat_500 = ['prop_lake_500', 'r_500', 'g_500', 'b_500']
sat_1000 = ['prop_lake_1000', 'r_1000', 'g_1000', 'b_1000']
sat_2500 = ['prop_lake_2500', 'r_2500', 'g_2500', 'b_2500']
sat_1025 = ['prop_lake_2500', 'r_2500', 'g_2500', 'b_2500', 
           'prop_lake_1000', 'r_1000', 'g_1000', 'b_1000']

cat = mod.RegMod(ord_vars=['region','cluster'],
                dat_vars=['date'],
                ide_vars=['latitude','longitude','maxe','dife'],
                y='severity',
                mod = mod.CatBoostRegressor(iterations=380,depth=6,
                   allow_writing_files=False,verbose=False)
                )
cat.fit(train_dat,weight=False,cat=False)


lig = mod.RegMod(ord_vars=['region','cluster','imtype'],
                dat_vars=['date'],
                ide_vars=['latitude','longitude','elevation','dife'] + sat_1025,
                y='severity',
                mod = mod.LGBMRegressor(n_estimators=470,max_depth=8)
                )
lig.fit(train_dat,weight=False,cat=True)


xgb = mod.RegMod(ord_vars=['region','cluster'],
                 dat_vars=['date'],
                 y='severity',
                 mod = mod.XGBRegressor(n_estimators=70, max_depth=2))
xgb.fit(train_dat,weight=False,cat=False)


rm = mod.EnsMod(mods={'xgb': xgb, 'cat': cat, 'lig': lig})


# Now getting files for out of sample data
test = feat.get_data(data_type='test')
test['pred'] = rm.predict_int(test)

form_dat = feat.sub_format(test)
print(form_dat['severity'].value_counts())

# function to check if similar to any past submissions
mod.check_similar(form_dat)

# Checking to see differences compared to best submission so far
current = form_dat.copy()
mod.check_day(current,day="sub_2023_02_16.csv")
current.groupby('region',as_index=False)['dif_2023_02_16'].value_counts()

# Saving the data and model
form_dat.to_csv(f'sub_BESTRESULTS_0216.csv',index=False)
mod.save_model(rm,f'mod_BESTRESULTS_0216')
###################################