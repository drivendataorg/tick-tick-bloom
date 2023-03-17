'''
Function to download
lat/lon data
'''

import pystac_client
import rioxarray
import planetary_computer

from . import get_data


catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

def elevation_point(lat,lon,box=1000):
    # returns point elevation, min/max/dif/mean within box meters
    try:
        ll = [lon,lat]
        search = catalog.search(
           collections=["cop-dem-glo-30"],
           intersects={"type": "Point", "coordinates": ll},)
        items = list(search.get_items())
        if len(items) == 0:
            md = -99999
            dat = {'latitude': lat, 'longitude': lon, 'box': box, 'elevation':md, 'mine':md, 'maxe':md, 'dife':md, 'avge': md, 'stde': md}
            return dat
        signed_asset = planetary_computer.sign(items[0].assets["data"])
        ro = rioxarray.open_rasterio(signed_asset.href)
        #ro.x.values
        #ro.y.values
        #ro.values # need to flatten
        ele = ro.sel(x=lon, y=lat, method="nearest").values[0]
        bbox = get_data.get_bounding_box(lat,lon,box)
        ro_clip = ro.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3])
        min_ele = ro_clip.values.min()
        max_ele = ro_clip.values.max()
        dif_ele = max_ele - min_ele
        avg_ele = ro_clip.values.mean()
        std_ele = ro_clip.values.std()
        dat = {'latitude': lat,
               'longitude': lon,
               'box': box,
               'elevation': ele, 
               'mine': min_ele, 
               'maxe': max_ele, 
               'dife': dif_ele, 
               'avge': avg_ele, 
               'stde': std_ele}
        ro.close()
        return dat
    except Exception:
        print(f'Query failed for {lat},{lon}')
        md = -99999
        dat = {'latitude': lat, 'longitude': lon, 'box': box, 'elevation':md, 'mine':md, 'maxe':md, 'dife':md, 'avge': md, 'stde': md}
        time.sleep(10)
        return dat