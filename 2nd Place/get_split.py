'''
This script evaluates
the differences in the train/test
set
'''

from src import feat, mod
from sklearn.model_selection import train_test_split

today = feat.today_str()
all_dat = feat.get_both()
train, test = mod.split(all_dat,test_size=3000)


cv2 = ['latitude','longitude','maxe','dife']


cm = mod.CatMod(ord_vars=['region'],
                dum_vars=None,
                dat_vars=None,
                ide_vars=cv2,
                y='test')


cm.fit(train)
cm.feat_import()

train['pred'] = cm.predict(train)
test['pred'] = cm.predict(test)

mod.rmse_region(train,'pred','test')
mod.rmse_region(test,'pred','test')

cm.fit(all_dat)
all_dat['pred'] = cm.predict(all_dat)

#feat.add_table(all_dat[['uid','pred']],'split_pred')

# Saving the model
mod.save_model(cm,f'weight_{today}')