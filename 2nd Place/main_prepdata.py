'''
So you can either do it two ways

One is to run each of the files
individually, e.g.


cd ./algaebloom
python get_dem.py
python get_lag.py
python get_sat.py

These take along time, and dem/sat can
fail periodically

They are not idempotent, so incrementally save 
the values to a SQLite database (in the data folder)

The easier way though to replicate the SQLite database
though is to run this script, it presumes you have the files

algaebloom/data
       - metadata.csv           # from competition
       - train_labels.csv       # from competition
       - submission_format.csv  # from competition
       - elevation_dem.csv      # generated via get_dem.py
       - spat_lag.csv           # generated via get_lag.py
       - sat.csv                # generated via get_sat.py
       - split_pred.csv         # generated via get_split.py

The final solution does not use the lag values, but have
included to replicate my prior tests. Get split needs to be
run AFTER the elevation stats are prepped, but otherwise 
the order does not matter

any questions? Feel free to email me, 

apwheele@gmail.com
Andy Wheeler
'''


import pandas as pd
import sqlite3

db = './data/data.sqlite'
tab_names = ['elevation_dem','split_pred','spat_lag','sat']
meta_names = ['meta','labels', 'format']
meta_csv = ['metadata','train_labels','submission_format']

fd = {c:t for c,t in zip(meta_csv,meta_names)}
ft = {t:t for t in tab_names}
fd.update(ft)

# Function to save out csv files
def save_csv(db=db):
    db_con = sqlite3.connect(db)
    for t in tab_names:
        res = pd.read_sql(f'SELECT * FROM {t}',db_con)
        res.to_csv(f'./data/{t}.csv',index=False)

# Function to prep SQLite DB
def prep_sql(db=db):
    db_con = sqlite3.connect(db)
    for csv,tab_name in fd.items():
        res = pd.read_csv(f'./data/{csv}.csv')
        res.to_sql(tab_name,index=False,if_exists='replace',con=db_con)
    # Showing resulting table names
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    rt = pd.read_sql(query,db_con)
    print('Resulting table names in sqlite')
    print(rt['name'].tolist())

if __name__ == "__main__":
    print('Executing script to prep data in sqlite')
    print('This expects the csv files listed at front')
    print('of script to be in the data folder')
    prep_sql()



