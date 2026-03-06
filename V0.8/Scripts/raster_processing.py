from typing import Union

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm


def resample_raster(filename, res, out_path=None):
    """
    Resample a raster to a given resolution
    Args:
        filename: path to the raster
        res: desired resolution

    Returns:
        data: numpy array of the resampled raster
        transform: rasterio transform of the resampled raster
    """
    with rasterio.open(filename) as dataset:
        if dataset.res[0] != dataset.res[1]:
            print("Warning: different resolutions x, y")
        scale_factor = dataset.res[0] / res
        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * scale_factor),
                int(dataset.width * scale_factor)
            ),
            resampling=Resampling.bilinear
        )

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )
        crs = dataset.crs
        dtypes = dataset.dtypes
    data = data[0]

    if out_path is not None:
        with rasterio.open(out_path, "w", driver='Gtiff', count=1,
                       height=data.shape[0], width=data.shape[1],
                       transform=transform, crs=crs,
                       dtype=dtypes[0]) as dest:
            dest.write(data, 1)
            return

    return data, transform


def read_overlapping_rasters(ras1, ras2) -> tuple:
    """
    Read two overlapping rasters and return the overlapping area as numpy arrays
    Args:
        ras1: rasterio raster
        ras2: rasterio raster

    Returns:
        overlap_1: numpy array of the overlapping area of ras1
        overlap_2: numpy array of the overlapping area of ras2
        profile: profile of the overlapping area (based on the profile of ras1)
    """
    ext1 = box(*ras1.bounds)
    ext2 = box(*ras2.bounds)
    intersection = ext1.intersection(ext2)
    win1 = rasterio.windows.from_bounds(*intersection.bounds, ras1.transform)
    win2 = rasterio.windows.from_bounds(*intersection.bounds, ras2.transform)

    overlap_1 = ras1.read(1, window=win1)
    overlap_2 = ras2.read(1, window=win2)

    if overlap_1.shape == overlap_2.shape:
        height, width = overlap_1.shape
    else:
        raise ValueError("rasters with different shapes")

    transform = rasterio.transform.from_bounds(*intersection.bounds, width, height)
    profile = ras1.profile.copy()
    profile.update(transform=transform, width=width, height=height)

    return overlap_1, overlap_2, profile


def save_raster(arr, profile, filename) -> None:
    """
    Save a numpy array as a raster
    Args:
        arr: numpy array
        profile: rasterio profile
        filename: path to save the raster
    Returns:
        None
    """
    with rasterio.open(filename, "w", **profile) as dst:
        dst.write(arr, 1)


def save_without_profile(arr, bounds, crs, filename) -> None:
    """
    Save a numpy array as a raster without a profile
    Args:
        arr: numpy array
        bounds: bounds of the raster
        crs: crs of the raster
        filename: path to save the raster
    Returns:
        None
    """
    transform = rasterio.transform.from_bounds(*bounds, width=arr.shape[1], height=arr.shape[0])

    with rasterio.open(filename, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype=arr.dtype, transform=transform, crs=crs) as dst:
        dst.write(arr, 1)


def read_two_rasters(raster1, raster2, res) -> list:
    """
    Read two overlapping rasters and resample them to the desired resolution.
    Returns each raster's array of the overlapped area and its profiles

    Args:
        raster1: path to raster 1
        raster2: path to raster 2
        res: desired resolution

    Returns:
        output: list of tuples (array, profile) for each raster
    """
    output = []
    with rasterio.open(raster1) as src1, rasterio.open(raster2) as src2:
        ext1 = box(*src1.bounds)
        ext2 = box(*src2.bounds)
        intersection = ext1.intersection(ext2)
        win1 = rasterio.windows.from_bounds(*intersection.bounds, src1.transform)
        win2 = rasterio.windows.from_bounds(*intersection.bounds, src2.transform)

        height = int((intersection.bounds[3] - intersection.bounds[1]) / res)
        width = int((intersection.bounds[2] - intersection.bounds[0]) / res)

        out_shape = (1, width, height)

        for dataset, window in zip([src1, src2], [win1, win2]):
            # resample data to target shape
            data = dataset.read(
                window=window,
                out_shape=out_shape,
                resampling=Resampling.bilinear
            )

            transform = rasterio.transform.from_bounds(*intersection.bounds, width, height)

            # scale image transform
            # transform = dataset.transform * dataset.transform.scale(
            #     (dataset.width / data.shape[-1]),
            #     (dataset.height / data.shape[-2])
            # )
            profile = dataset.profile.copy()
            profile.update(transform=transform, width=width, height=height)
            output.append((data, profile))
    return output


def merge_rasters(raster_list, save_path) -> None:
    """
    Merge a list of rasters into a single raster
    Args:
        raster_list: list of paths to the rasters to merge
        save_path: path to save the merged raster

    Returns:
        None
    """

    from rasterio.merge import merge
    mosaic, transform = merge(raster_list)

    with rasterio.open(raster_list[0]) as src:
        meta = src.meta.copy()  # I don't know why it doesn't work with profile but it does with meta.
    meta.update(
        {"driver": "GTiff",
         "height": mosaic.shape[1],
         "width": mosaic.shape[2],
         "transform": transform,
         }
    )
    with rasterio.open(save_path, "w", **meta) as m:
        m.write(mosaic)


def rasterize_overlapping_shapes(shapes: gpd.GeoDataFrame, field: str, res: float,
                                 fill_value: int = 0, aggr_funct: np.ufunc = np.maximum, save_as: str = None):
    """
    Rasterize overlapping polygons and reduces value of each pixel based on a aggregation function
    Args:
        shapes: geopandas dataframe with polygons
        field: field to use for rasterization
        res: resolution of the raster
        fill_value: value to fill the raster with
        aggr_funct: aggregation function to use
        save_as: path to save the raster

    Returns:
        max_values_raster: numpy array of the rasterized shapes
    """
    assert aggr_funct in [np.maximum, np.add], "Invalid reduce function"
    x_res = res  # Width of a cell in your raster
    y_res = res  # Height of a cell in your raster

    # Create a bounding box around your polygons
    bounds = shapes.total_bounds
    x_min, y_min, x_max, y_max = bounds

    # Calculate the dimensions of your raster
    width = int((x_max - x_min) / x_res)
    height = int((y_max - y_min) / y_res)

    # Initialize an array to keep track of the maximum values
    max_values_raster = np.zeros((height, width), dtype=np.float32)

    # Rasterize each polygon separately and update the maximum values raster
    for value, geom in tqdm(zip(shapes[field], shapes.geometry), total=len(shapes)):
        # Rasterize the polygon
        rasterized = rasterize(
            [(geom, value)],
            out_shape=(height, width),
            transform=rasterio.transform.from_bounds(x_min, y_min, x_max, y_max, width, height),
            fill=fill_value,  # Initial fill value
            all_touched=True
        )

        # Update the max_values_raster with the maximum values
        max_values_raster = aggr_funct(max_values_raster, rasterized)

    if save_as is not None:
        # Define the raster metadata
        raster_meta = {
            'driver': 'GTiff',
            'height': max_values_raster.shape[0],
            'width': max_values_raster.shape[1],
            'count': 1,
            'dtype': max_values_raster.dtype,
            'crs': shapes.crs,
            'transform': rasterio.transform.from_bounds(x_min, y_min, x_max, y_max, width, height)
        }

        # Write the raster to a TIFF file
        with rasterio.open(save_as, 'w', **raster_meta) as dst:
            dst.write(max_values_raster, 1)

    else:
        return max_values_raster


def clip_raster_to_extents(
        raster_path, 
        extents, 
        output_path,
        crs: int = 25833) -> None:
    """
    Clip a raster to the given extents and save the result.

    Args:
        raster_path (PathLike): Path to the input raster file.
        extents (ExtentsInput): The geographical extents to clip to.
        output_path (PathLike): Path to save the clipped raster.
    """
    if isinstance(extents, (list, tuple, np.ndarray)):
        extents = gpd.GeoDataFrame(geometry=[box(*extents)], crs=crs)

    bounds = extents.total_bounds

    with rasterio.open(raster_path) as src:
        window = rasterio.windows.from_bounds(*bounds, src.transform)
        clipped_data = src.read(window=window)

        # Update transform for the clipped area
        clipped_transform = rasterio.windows.transform(window, src.transform)

        # Update profile
        profile = src.profile.copy()
        profile.update({
            'height': clipped_data.shape[1],
            'width': clipped_data.shape[2],
            'transform': clipped_transform
        })

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(clipped_data)



def polygonize_array(result_array: np.ndarray, dem_profile: rasterio.profiles.Profile,
                       field="value", threshold_value=1):
    """
    Polygonize an array whose values are greater than the given threshold and save the results as a 
    geopandas dataframe with the given field name.

    Args:
        result_array: array with the results
        dem_profile: profile of the dem
        field: field name to be used for the polygonized results
        threshold_value: value to be used as threshold for the polygonization

    Returns:
        gpd_polygonized_raster: geopandas dataframe with the polygonized results
    """

    raster_transform = dem_profile['transform']
    raster_crs = dem_profile['crs']

    results_binary = (result_array >= threshold_value).astype("int16")

    results = ({"properties": {"id": i, field: int(v)}, "geometry": s}
               for i, (s, v) in enumerate(shapes(results_binary, mask=None, transform=raster_transform)))
    geoms = list(results)
    gpd_polygonized_raster = gpd.GeoDataFrame.from_features(geoms)
    gpd_polygonized_raster = gpd_polygonized_raster[gpd_polygonized_raster[field] >= threshold_value]
    gpd_polygonized_raster = gpd_polygonized_raster.set_crs(raster_crs)
    return gpd_polygonized_raster


def rasterize_shape(shapes:Union[gpd.GeoDataFrame, list[BaseGeometry]], dem_profile: rasterio.profiles.Profile) -> np.ndarray:
    """
    Rasterize the release area
    Args:
        shapes: shapes as a geopandas dataframe or a shapely geometry
        dem_profile: profile of the dem

    Returns:
        rasterized: rasterized release area as a numpy array
    """
    dem_height = dem_profile['height']
    dem_width = dem_profile['width']
    dem_transform = dem_profile['transform']
    if isinstance(shapes, gpd.GeoDataFrame):
        geom = [shapes_ii for shapes_ii in shapes.geometry]
    elif all(isinstance(shape, BaseGeometry) for shape in shapes):
        geom = shapes
    else:
        raise ValueError("shapes must be either a GeoDataFrame or a list with geometries")

    rasterized = rasterize(geom,
                           out_shape=(dem_height, dem_width),
                           fill=0,
                           out=None,
                           transform=dem_transform,
                           all_touched=True,
                           default_value=1,
                           dtype=None)

    return rasterized

def get_data_mask(raster, nodata_value=0):
    with rasterio.open(raster) as src:
        profile = src.profile
        nodata = src.nodata
        if nodata is None:
            nodata = nodata_value
        data = src.read(1)
        data = (data != nodata).astype(np.uint8)
    mask = polygonize_array(data, profile, field="value", threshold_value=1)
    return mask[["geometry"]]

def get_data_mask_two_rasters(raster1, raster2, nodata_value=0, crs=25833):
    masks = []
    for raster in [raster1, raster2]:
        mask = get_data_mask(raster, nodata_value=nodata_value)
        mask = mask.to_crs(crs)
        masks.append(mask)
    combined_mask = gpd.overlay(masks[0], masks[1], how="intersection")
    return combined_mask
