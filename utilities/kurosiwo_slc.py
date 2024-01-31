import datetime
import os
import pickle as pkl
import subprocess
import zipfile
from pathlib import Path

import adlfs
import asf_search as asf
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rio
import xarray as xr
from shapely import wkt
from shapely.geometry import Polygon, box
from tqdm import tqdm
import time

def download_from_asf(geometry,start,end,storage_path='slc_data/'):
    print('Searching for polygon: ',geometry, '\nFrom: ',start,' till: ',end)
    opts = {
        'platform': asf.PLATFORM.SENTINEL1,
        'start':str(start),
        'end': str(end),
        'processingLevel': 'SLC'

    }
    results = asf.geo_search(intersectsWith=geometry, **opts)
    info = results[0].geojson()
    write_path = os.path.join(storage_path,info['properties']['fileName'])
    folder = os.path.join(storage_path,info['properties']['fileID'])
    if os.path.isfile(write_path):
        print('Product already downloaded. Skipping download.')
        return write_path, folder
    print('Downloading product: ',info['properties']['fileName'])
    session = asf.ASFSession().auth_with_creds('ngearth', 'shQradFCJ2j93DB')
    results.download(path=storage_path, session=session)   
    return write_path,  folder


def merge_rasters(raster_list,save_path):
    """
    Merges a list of rasters into a single raster.
    Args:
        raster_list (list): List of rasters
    Returns:
        merged_raster (xarray.DataArray): Merged raster
    """
   
    #gdal_command = ["gdal_merge.py", "-o", save_path, "-n", "0" ,  *raster_list]
    #status = subprocess.run(gdal_command)    
    # Build vrt
    #gdal Warp first 
    warped_list =[]
    vrt_path = Path(save_path[:-4]).parent
    vrt_path = vrt_path / Path(vrt_path.name + '-vrt') 
    vrt_path.mkdir(parents=True, exist_ok=True)
    vrt_path = vrt_path / Path(save_path.split('/')[-1][:-4] + '.vrt')
    for i in raster_list:
        if not os.path.isfile(i[:-5] + "_Warped.tif"):
            warp_command = ["gdalwarp", "-of", "GTiff","-t_srs", "EPSG:3857",i, i[:-5] + "_Warped.tif"]
            subprocess.run(warp_command,check=True)   
        else:
            print('File exists. Skipping file: ',i[:-5] + "_Warped.tif")
        warped_list.append( i[:-5] + "_Warped.tif")

    if not os.path.isfile(vrt_path):
        gdal_vrt_command = ["gdalbuildvrt", vrt_path , *warped_list]
        status = subprocess.run(gdal_vrt_command,check=True)   
        print(status)
    else:
        print('File exists. Skipping file: ',vrt_path)
    if not os.path.isfile(save_path):
        # Translate vrt to tif
        gdal_translate_command = ["gdal_translate", "-of", "GTiff",vrt_path, save_path]   
        status = subprocess.run(gdal_translate_command,check=True) 
        print(status)  
   
    return save_path


def merge_slc_frames(measurement_path, save_path,folder):

    files = os.listdir(measurement_path)

    vv_files = [os.path.join(measurement_path,f) for f in files if 'vv' in f]
    vh_files = [os.path.join(measurement_path,f) for f in files if 'vh' in f]
    if len(vv_files) == 0 or len(vh_files) == 0:
        return None
    vv_save_path = os.path.join(save_path, folder + '_vv.tif')
    vh_save_path = os.path.join(save_path, folder + '_vh.tif')
    if not os.path.isfile(vv_save_path):
        merge_rasters(vv_files,vv_save_path)
    if not os.path.isfile(vh_save_path):
        merge_rasters(vh_files,vh_save_path)
    return vv_save_path, vh_save_path

def create_frame(slc_path,save_path):
    '''
        Merges all vv (vh) tifs into a single tif
        @param slc_path: Path to the SLC product
        @param save_path: Path to store the merged tifs 
    '''

    unzip_path = slc_path[:-4] + '.SAFE'
    folder = slc_path[:-4].split('/')[-1]
    if not os.path.isdir(unzip_path):
        if os.path.isfile(slc_path):
            print('Unzipping file: ',slc_path)
            with zipfile.ZipFile(slc_path, 'r') as zip_ref:
                slc_root_path = slc_path.split('/')[-2]
                zip_ref.extractall(slc_root_path)
        else:
            print('No SLC found in: ',slc_path,'. Skipping tile.')
    measurement_path = os.path.join(unzip_path,'measurement')
    tile_folder = Path(os.path.join(save_path,folder))
    tile_folder.mkdir(parents=True, exist_ok=True)
    merge_paths = merge_slc_frames(measurement_path,str(tile_folder),folder)
    if merge_paths is None:
        return None
    vv_save_path, vh_save_path = merge_paths
    return vv_save_path, vh_save_path

    
def search(slc_frames_storage_path):
    tiles = pd.read_csv('slc_geometries_full.csv')
    for row in tqdm(tiles.iterrows()):
        print(row[1]['geometry'])
        print(row[1]['source_date'])
        source_date = datetime.datetime.strptime(row[1]['source_date'], '%Y-%m-%d')
        start_date = (source_date - datetime.timedelta(days=1)) + datetime.timedelta(hours=23)
        end_date = source_date +datetime.timedelta(hours=23)
        
        #d = (datetime.datetime(row[1]['source_date']) + datetime.datetime.timedelta(days=1))
        print(start_date)
        print(end_date)
        try:
            slc_file, folder = download_from_asf(row[1]['geometry'],start=start_date, end=end_date)
        except:
            print('Download failed. Retrying after 30seconds.')
            time.wait(280)

        create_frame(row[1]['geometry'],slc_file,slc_frames_storage_path)
        break

def create_tile(reference_tile,kuro_siwo_root_path, source,output_tile_storage_path,slc_vv_path,slc_vh_path):
    '''
    Create a SLC tile from a reference tile
    @param reference_tile: Reference tile
    @param output_tile_storage_path: Path to store the output tile
    @param slc_frames_storage_path: Path to get the SLC frames
    @return: SLC tile path
    '''
    out_vv_path = os.path.join(output_tile_storage_path,reference_tile, source+'_IVV.tif')
    out_vh_path = os.path.join(output_tile_storage_path,reference_tile, source+'_IVH.tif')
    #Load reference tile
    reference_tile = os.path.join(kuro_siwo_root_path,reference_tile)
    reference_files = os.listdir(reference_tile)
    vv_tile = [f for f in reference_files if f.startswith(source+'_IVV')][0]
    vh_tile = [f for f in reference_files if f.startswith(source+'_IVH')][0]

    vv_tile = rio.open_rasterio(os.path.join(reference_tile,vv_tile))
    vh_tile = rio.open_rasterio(os.path.join(reference_tile,vh_tile))
    extent = vv_tile.rio.bounds()

    # Load SLC frame
    slc_vv = rio.open_rasterio(slc_vv_path)
    # Change crs if needed
    if slc_vv.rio.crs != vv_tile.rio.crs:
        slc_vv = slc_vv.rio.reproject(vv_tile.rio.crs)
    slc_extent = slc_vv.rio.bounds()
    
    overlap = not (slc_extent[2] < extent[0] or
               slc_extent[0] > extent[2] or
               slc_extent[3] < extent[1] or
               slc_extent[1] > extent[3])

    if overlap:
        print("Rasters spatially overlap.")
    else:
        print("Rasters do not spatially overlap. Unable to crop.")
        print(slc_extent)
        print(extent)
        return None

    slc_vv = slc_vv.rio.clip_box(*extent)
    slc_vh = rio.open_rasterio(slc_vh_path)
    
    if slc_vh.rio.crs != vv_tile.rio.crs:
        slc_vh = slc_vh.rio.reproject(vv_tile.rio.crs)

    slc_vh = slc_vh.rio.clip_box(*extent)
    #Write to disk
    file_path = Path(out_vv_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    #Split to real and imaginary parts
    arr_vv = np.array([ np.real(slc_vv.values), np.imag(slc_vv.values)]).squeeze()
    slc_array_vv = xr.DataArray(arr_vv, dims=('channel', 'y', 'x'), coords={'channel': ['real', 'imaginary'], 'y': slc_vv.coords['y'], 'x': slc_vv.coords['x']})
    slc_array_vv.rio.set_crs(slc_vv.rio.crs)
    slc_array_vv = slc_array_vv.rio.reproject_match(vv_tile)
    slc_array_vv.rio.to_raster(out_vv_path)

    # For the vh
    arr_vh = np.array([ np.real(slc_vh.values), np.imag(slc_vh.values)]).squeeze()
    slc_array_vh = xr.DataArray(arr_vh, dims=('channel', 'y', 'x'), coords={'channel': ['real', 'imaginary'], 'y': slc_vh.coords['y'], 'x': slc_vh.coords['x']})
    slc_array_vh.rio.set_crs(slc_vh.rio.crs)
    slc_array_vh = slc_array_vh.rio.reproject_match(vh_tile)
    slc_array_vh.rio.to_raster(out_vh_path)

    return out_vv_path, out_vh_path

def get_slc_tile(source_date,kuro_siwo_root_path,source, polygon, reference_tile,slc_product_storage_path,slc_frames_storage_path,output_tile_storage_path):
    '''
    Get the respective SLC tile for a given reference tile
    @param source_date: Date of the reference tile
    @param polygon: Polygon of the reference tile
    @param reference_tile: Reference tile
    '''
    out_vv_path = os.path.join(output_tile_storage_path,reference_tile, source+'_IVV.tif')
    out_vh_path = os.path.join(output_tile_storage_path,reference_tile, source+'_IVH.tif')
    if os.path.isfile(out_vv_path) and os.path.isfile(out_vh_path):
        print('SLC tile already exists')
        return 0
    source_date = datetime.datetime.strptime(source_date, '%Y-%m-%d')
    start_date = (source_date - datetime.timedelta(days=1)) + datetime.timedelta(hours=23,minutes=59,seconds=59)
    end_date = source_date +datetime.timedelta(hours=23,minutes=59,seconds=59)
    try:
        slc_file, folder = download_from_asf(polygon,start=start_date, end=end_date,storage_path=slc_product_storage_path)
    except:
        print('Download failed. Retrying after 5 seconds')
        time.sleep(5)
        d_status = False
        for tries in range(0,4):
            try:
                slc_file, folder = download_from_asf(polygon,start=start_date, end=end_date,storage_path=slc_product_storage_path)
                d_status = True
                break
            except:
                print('Download failed. Retrying after 5 seconds')
                time.sleep(5)
            if tries==3:
                print('Download failed. Continuing to next tile')
                return 1
    frame_paths = create_frame(slc_file,slc_frames_storage_path)
    if frame_paths is None:
        print('Frame was not complete. Continuing to next tile')
        return 1
    vv_save_path, vh_save_path = frame_paths
    tile_status = create_tile(reference_tile=reference_tile,kuro_siwo_root_path=kuro_siwo_root_path, source=source,output_tile_storage_path=output_tile_storage_path,slc_vv_path=vv_save_path,slc_vh_path=vh_save_path)
    if tile_status is not None:
        print('Tile created successfully')
        return 0
    else:
        print('Tile creation failed. Continuing to next tile')
        return 1

def create_slc_pairs(pickle_path,kuro_siwo_root_path, slc_product_storage_path, slc_frames_storage_path, output_tile_storage_path):
    #Read samples filepath pickle
    with open(pickle_path, 'rb') as handle:
        grid_dict = pkl.load(handle)
    import pprint
    if not os.path.isdir(output_tile_storage_path):
        Path(output_tile_storage_path).mkdir(parents=True, exist_ok=True)
    if not os.path.isdir(slc_frames_storage_path):
        Path(slc_frames_storage_path).mkdir(parents=True, exist_ok=True)
    if not os.path.isdir(slc_product_storage_path):
        Path(slc_product_storage_path).mkdir(parents=True, exist_ok=True)

    for key in grid_dict:
        polygon =  wkt.loads( grid_dict[key]['info']['geom'])
        gdf = gpd.GeoDataFrame(geometry=[polygon])
        gdf.set_crs('epsg:3857', inplace=True)
        gdf = gdf.to_crs(epsg=4326)
        record = {}
        polygon = gdf.to_wkt().at[0,'geometry']
        record['kuro_siwo_root_path'] = kuro_siwo_root_path
        record["path"] = grid_dict[key]["path"]
        record['geometry'] = polygon
        record['MS1_date'] = grid_dict[key]['info']['sources']['MS1']['source_date']
        record['SL1_date'] = grid_dict[key]['info']['sources']['SL1']['source_date']
        record['SL2_date'] = grid_dict[key]['info']['sources']['SL2']['source_date']
        record['flood_date'] = grid_dict[key]["info"]['flood_date']
        
        for source in ['MS1','SL1','SL2']:
            status = get_slc_tile(source_date=record[source+'_date'],kuro_siwo_root_path=kuro_siwo_root_path,source=source, polygon=str(record['geometry']), reference_tile=record['path'],slc_product_storage_path=slc_product_storage_path,slc_frames_storage_path=slc_frames_storage_path,output_tile_storage_path=output_tile_storage_path)


if __name__ == '__main__':
    #Example usage
    create_slc_pairs(pickle_path='YOUR PICKLE PATH',kuro_siwo_root_path='KURO SIWO ROOT',slc_product_storage_path='RAW PRODUCT STORAGE PATH',slc_frames_storage_path='MERGED TIFS STORAGE PATH',output_tile_storage_path='SLC DATASET STORAGE PATH')
