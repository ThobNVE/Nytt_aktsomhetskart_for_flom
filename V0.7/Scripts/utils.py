import re
from typing import Union
from uuid import uuid4

import geopandas as gpd
import pandas as pd
from shapely.geometry import box


def clean_url(url: str) -> str:
    cleaned_url = re.sub(r"(?<!:)/{2,}", "/", url)
    return cleaned_url

def transform_bounds(bounds:list[float], crs_in:int=25833, crs_out:int=4386) -> list[float]:
    bbox = gpd.GeoDataFrame(geometry=[box(*bounds)], crs=crs_in).to_crs(crs_out).total_bounds
    return bbox


def split_bbox(bbox:Union[gpd.GeoDataFrame, list[float], tuple[float]], n_rows:int, n_cols:int, crs=25833) -> gpd.GeoDataFrame:
    """
    
    """
    if isinstance(bbox, (list, tuple)):
        minx, miny, maxx, maxy = bbox
    elif isinstance(bbox, gpd.GeoDataFrame):
        minx, miny, maxx, maxy = bbox.total_bounds
        if bbox.crs.to_epsg() != crs:
            bbox = bbox.to_crs(crs)
    else:
        raise TypeError("Invalid bbox type")
    n_rows = int(n_rows)
    n_cols = int(n_cols)
    width = (maxx - minx) / n_cols
    height = (maxy - miny) / n_rows
    sub_boxes = []
    for i in range(n_cols):
        for j in range(n_rows):
            sub_minx = minx + i * width
            sub_miny = miny + j * height
            sub_maxx = sub_minx + width
            sub_maxy = sub_miny + height
            sub_boxes.append(box(sub_minx, sub_miny, sub_maxx, sub_maxy))
    subgrid = gpd.GeoDataFrame(geometry=sub_boxes, crs = crs)
    subgrid["id"] = range(1, len(subgrid)+1)
    return subgrid

def bounds_area(bounds):
    """
    returns area in km2
    """
    xmin, ymin, xmax, ymax = bounds
    return ((xmax - xmin)/1000) * ((ymax - ymin)/1000)


def assign_unique_id(gdf: pd.DataFrame|gpd.GeoDataFrame, id_field:str="lokalid") -> pd.DataFrame|gpd.GeoDataFrame:
    """
    Assign unique identifiers to each row in a GeoDataFrame.
    If the specified ID field doesn't exist, creates it and populates with UUIDs.
    If duplicate IDs are found, appends incremental suffixes to make them unique.
    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame to process.
        id_field (str, optional): Name of the field to use for IDs. Defaults to "lokalid".
    Returns:
        gpd.GeoDataFrame: The GeoDataFrame with unique IDs assigned.
    Examples:
        >>> import geopandas as gpd
        >>> gdf = gpd.GeoDataFrame({'geometry': [Point(0, 0), Point(1, 1)]})
        >>> gdf_with_ids = assign_unique_id(gdf)
        >>> gdf_with_ids['lokalid'].is_unique
        True
        >>> gdf_duplicates = gpd.GeoDataFrame({'lokalid': ['id1', 'id1', 'id2']})
        >>> result = assign_unique_id(gdf_duplicates)
        >>> result['lokalid'].tolist()
        ['id1-1', 'id1-2', 'id2']
    """
    
    if id_field not in gdf.columns:
        gdf[id_field] = ""
        gdf[id_field] = gdf[id_field].apply(lambda _: str(uuid4()))

    duplicated_ids = gdf[gdf.duplicated(subset=[id_field], keep=False)][id_field].unique()
    
    for original_id in duplicated_ids:
        mask = gdf[id_field] == original_id
        indices = gdf[mask].index
        
        # Asignar sufijos incrementales
        for i, idx in enumerate(indices):
            gdf.at[idx, id_field] = f"{original_id}-{i+1}"
    
    return gdf