import pandas as pd, numpy as np
import sys,os,shutil,gc,re,json,glob,math,time,random,logging,warnings
from pqdm.processes import pqdm
import sklearn.metrics as skm
from datetime import timedelta
from odc import stac
import planetary_computer as pc
from pystac_client import Client
from pystac.extensions.eo import EOExtension as eo
import geopy.distance as distance

warnings.filterwarnings("ignore")

split=sys.argv[1] #'train','test'
satellite=sys.argv[2] #'lsat','snel'

ROOT_DIR = 'data'
DATA_DIR_RAW = f'{ROOT_DIR}/raw'
DATA_DIR_INTERIM = f'{ROOT_DIR}/interim'
out_dir = f'{DATA_DIR_INTERIM}/{split}/{satellite}'
os.makedirs(out_dir,exist_ok=True)
META_DIR = f'{DATA_DIR_RAW}/meta'

print(f'downloading {satellite} {split} data to {out_dir}')

QUERY_RADIUS = 5000
EXPORT_RADIUS = 200
MAX_DAYS = 15

meta = pd.read_csv(f'{META_DIR}/metadata.csv')
df=meta[meta.split==split]
print(f'#{len(df)} split samples')

def get_catalog():
    # Establish a connection to the STAC API
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    return catalog

catalog = get_catalog()

# get our bounding box to search latitude and longitude coordinates
def get_bounding_box(latitude, longitude, meter_buffer):
    """
    Given a latitude, longitude, and buffer in meters, returns a bounding
    box around the point with the buffer on the left, right, top, and bottom.

    Returns a list of [minx, miny, maxx, maxy]
    """
    distance_search = distance.distance(meters=meter_buffer)

    # calculate the lat/long bounds based on ground distance
    # bearings are cardinal directions to move (south, west, north, and east)
    min_lat = distance_search.destination((latitude, longitude), bearing=180)[0]
    min_long = distance_search.destination((latitude, longitude), bearing=270)[1]
    max_lat = distance_search.destination((latitude, longitude), bearing=0)[0]
    max_long = distance_search.destination((latitude, longitude), bearing=90)[1]

    return [min_long, min_lat, max_long, max_lat]


# get our date range to search, and format correctly for query
def get_date_range(date, time_buffer_days=15):
    """Get a date range to search for in the planetary computer based
    on a sample's date. The time range will include the sample date
    and time_buffer_days days prior

    Returns a string"""
    datetime_format = "%Y-%m-%d"
    range_start = pd.to_datetime(date) - timedelta(days=time_buffer_days)
    date_range = f"{range_start.strftime(datetime_format)}/{pd.to_datetime(date).strftime(datetime_format)}"
    return date_range

def query_item(sample,date_buffer):
    bbox = get_bounding_box(sample['latitude'], sample['longitude'], meter_buffer=QUERY_RADIUS)
    date_range = get_date_range(sample['date'],date_buffer)
    if satellite=='lsat':
        search = catalog.search(
            collections=["landsat-c2-l2"], bbox=bbox, datetime=date_range,
            query={"platform": {"in": ["landsat-8", "landsat-9"]},},
        )
    else:
        search = catalog.search(
        collections=["sentinel-2-l2a"], bbox=bbox, datetime=date_range,
    )

    # see how many items were returned
    items = [item for item in search.get_all_items()]

    # get details of all of the items returned
    item_details = pd.DataFrame(
        [
            {
                "datetime": item.datetime.strftime("%Y-%m-%d_%H-%M-%S"),
                "platform": item.properties["platform"],
                "min_long": item.bbox[0],
                "max_long": item.bbox[2],
                "min_lat": item.bbox[1],
                "max_lat": item.bbox[3],
                "bbox": item.bbox,
                "cloud_cover":eo.ext(item).cloud_cover,
                "item_obj": item,
            }
            for item in items
        ]
    )
    
    if len(item_details)>0:
        # check which rows actually contain the sample location
        item_details["contains_sample_point"] = (
            (item_details.min_lat < sample['latitude'])
            & (item_details.max_lat > sample['latitude'])
            & (item_details.min_long < sample['longitude'])
            & (item_details.max_long > sample['longitude'])
        )


        item_details = item_details[item_details["contains_sample_point"]]
        item_details['uid'] = sample['uid']
    
    return item_details


landsat_bands = ['coastal','red', 'blue','green','nir08','swir16','swir22'] 
landsat_bands += ['atran', 'cdist', 'drad', 'emis', 'emsd', 'lwir11','qa', 'qa_aerosol', 'qa_pixel', 'qa_radsat','trad', 'urad']
sentinel_bands = ['AOT', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B11', 'B12', 'B8A', 'SCL', 'WVP']

def crop_image(item, bounding_box, satellite, bands):
    """
    Given a STAC item from Landsat and a bounding box tuple in the format
    (minx, miny, maxx, maxy), return a cropped portion of the item's visual
    imagery in the bounding box.

    Returns numpy array 
    """
    (minx, miny, maxx, maxy) = bounding_box
    image = stac.stac_load(
            [item], bands=bands, bbox=[minx, miny, maxx, maxy]
        ).isel(time=0)
    if satellite=='lsat':
        arr = image[bands].to_array().to_numpy()
    else:
        arr=[]
        for b in bands:
            x=image[b].to_numpy()
            arr.append(x)
        arr=np.array(arr)
    return arr


def export_item(sample,item_details,out_dir):
    """
    Download  satellite image and export as a compressed numpy array
    """
    minx, miny, maxx, maxy = get_bounding_box(
            sample['latitude'], sample['longitude'], meter_buffer=EXPORT_RADIUS
        )
    bbox = (minx, miny, maxx, maxy)
    export_failed = False
    for _,row in item_details.iterrows():
        item = row.item_obj
        try:
            if row.platform in ['Sentinel-2A','Sentinel-2B']:
                x = crop_image(item,bbox,'snel', sentinel_bands)                
            else:
                x = crop_image(item,bbox,'lsat', landsat_bands)
            fname = row.uid+'_'+row.datetime+'_'+row.platform+'_'+str(row.ix)
            np.savez(f'{out_dir}/{fname}',x)
        except Exception as e:
            export_failed=True
            print(e)

    return export_failed

def get_item(sample,sleep=0.1):
    success=False
    while not success:
        try:
            item_details = query_item(sample,date_buffer=MAX_DAYS)
            success=True
        except Exception as e:
            print(f'ERROR {e}. Waiting for {sleep}s \n')
            time.sleep(sleep)
            return get_item(sample,sleep*2)
    return item_details


def extract_uid_data(uid):
    sample = df[df.uid==uid].to_dict(orient='records')[0]
    export_failed=True
    while export_failed:
        item_details = get_item(sample)
        if len(item_details)==0:
            export_failed=False
        else:
            item_details['ix'] = range(len(item_details))
            export_failed = export_item(sample,item_details,out_dir)
            if not export_failed:
                return item_details.drop(columns='item_obj')
    

args = df.uid.values
results = pqdm(args, extract_uid_data, n_jobs=16)
meta=pd.concat(results).reset_index(drop=True)
meta.to_csv(f'{out_dir}/meta.csv',index=False)
