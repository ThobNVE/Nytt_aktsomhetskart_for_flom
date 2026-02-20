from pathlib import Path
from tempfile import TemporaryDirectory

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pysheds.grid import Grid
from rasterio.enums import Resampling
from shapely.geometry import box
import geojson

import hoydedata_api

#from utils import geometry_utils, raster_processing
#from utils.path_utils import ensure_output_directory, validate_and_convert_path

#logger = get_module_logger(__name__)


MAX_AREA = 100_000

def compute_flow_acc(dem_path, path_out) -> tuple:
    """
    Computes the flow accumulation of a digital elevation model (DEM).

    Args:
        dem_path (str): The file path to the DEM.
        path_out (str): The file path to save the flow accumulation result.

    Returns:
        tuple: A tuple containing the Grid object, the flow accumulation and direction grid, and direction map.

    """

    grid = Grid.from_raster(str(dem_path), data_name='dem', nodata=-9999)
    dem = grid.read_raster(str(dem_path), data_name='dem', nodata=-9999)
    pit_filled_dem = grid.fill_pits(dem)
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    inflated_dem = grid.resolve_flats(flooded_dem)

    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    fdir = grid.flowdir(inflated_dem, dirmap=dirmap)
    acc = grid.accumulation(fdir, dirmap=dirmap)

    if path_out is not None:
        grid.to_raster(acc, path_out)

    return grid, acc, fdir, dirmap

def compute_rivers(dem_path, path_out, threshold, rivers_out, flowdir_out) -> tuple:
    """
    Computes the flow accumulation of a digital elevation model (DEM).

    Args:
        dem_path (str): The file path to the DEM.
        path_out (str): The file path to save the flow accumulation result.

    Returns:
        tuple: A tuple containing the Grid object, the flow accumulation and direction grid, and direction map.

    """
    
    grid = Grid.from_raster(str(dem_path), data_name='dem', nodata=-9999)
    dem = grid.read_raster(str(dem_path), data_name='dem', nodata=-9999)
    pit_filled_dem = grid.fill_pits(dem)
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    inflated_dem = grid.resolve_flats(flooded_dem)

    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    fdir = grid.flowdir(inflated_dem, dirmap=dirmap)
    acc = grid.accumulation(fdir, dirmap=dirmap)
    print("accumulation calculated")
    rivers = grid.extract_river_network(fdir, acc > threshold)
    print("river network extracted, saving outputs")

    if flowdir_out is not None:
        grid.to_raster(fdir, flowdir_out)
        print(f"saved flow direction raster to {flowdir_out}")

    if path_out is not None:
        grid.to_raster(acc, path_out)
        print(f"saved flow accumulation raster to {path_out}")

    if rivers_out is not None:
         with open(rivers_out, 'w') as f:
             geojson.dump(rivers, f)
         print(f"saved river network raster to {rivers_out}")

    return grid, acc, fdir, dirmap, rivers



