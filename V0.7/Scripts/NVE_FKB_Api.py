import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml
from shapely.wkt import loads
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SAWarning
from sqlalchemy.orm import sessionmaker

#from config.logging import get_module_logger
#from config.settings import settings

#logger = get_module_logger(__name__)


def get_dwh_databases() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "dwh.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        db_dict = yaml.safe_load(f)
        return db_dict


def get_selected_databases(db_names: list[str]) -> dict[str,dict]:
    dbs = get_dwh_databases()
    selected = {k: dbs[k] for k in db_names if k in dbs}
    return selected


# DWH connection
def get_sql_db(bounds, dbname, table, fields=None, server='gis-sql04', clip=True, crs=25833, limit=None):
#def get_sql_db(bounds, dbname, table, fields=None, clip=True, crs=25833, limit=None):
    """
    Connects to the database and retrieves water area data within the specified bounds.
    example:
    dbname = kart
    table = sk.FKB50_VANN_GRENSE
    """
    #logger.info(f"Connecting to database '{dbname}' on server '{server}' for table '{table}'")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SAWarning)
        try:
            connection_string = f'mssql+pyodbc://{server}/{dbname}?driver=ODBC+Driver+17+for+SQL+Server&Trusted_Connection=yes'
            engine = create_engine(connection_string)
            inspector = inspect(engine)
            #logger.debug("Database engine created and inspector initialized.")
        except Exception as e:
            
            print(f"Failed to connect or initialize inspector: {e}")
            #logger.error(f"Failed to connect or initialize inspector: {e}")
            # Try to log table info if possible
        try:
            schema = table.split(".")[0] if "." in table else None
            table_name = table.split(".")[-1]
            if inspector.has_table(table_name, schema=schema):
                columns = inspector.get_columns(table_name, schema=schema)
                #logger.info(f"Table '{table}' exists. Columns: {[col['name'] for col in columns]}")
            else:
                logger.warning(f"Table '{table}' does not exist in database '{dbname}'.")
                existing_tables = inspector.get_table_names(schema=schema) if schema else inspector.get_table_names()
                #logger.info(f"Existing tables in database '{dbname}': {existing_tables}")
        except Exception as inner_e:
            
            print(f"Could not inspect table '{table}': {inner_e}")
            

        try:
            table_columns = inspector.get_columns(*table.split(".")[::-1])
            table_fields = [col["name"] for col in table_columns if col["name"].lower() != "shape"]
            #logger.debug(f"Retrieved columns for table '{table}': {table_fields}")
        except Exception as e:
            #logger.error(f"Failed to retrieve columns for table '{table}': {e}")
            raise
        
        if bounds is not None:
            xmin, ymin, xmax, ymax = bounds
            #logger.info(f"Using bounds: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
            spatial_query = f"WHERE SHAPE.STIntersects(geometry::STGeomFromText('POLYGON(({xmin} {ymin}, {xmax} {ymin}, {xmax} {ymax}, {xmin} {ymax}, {xmin} {ymin}))', {crs})) = 1"
        else:
            spatial_query = ""
        
        if limit is not None:
            query_limit = f"TOP {limit} "
        else:
            query_limit = ""

        if fields is None:
            fields = table_fields
            #logger.debug("No fields specified, using all table fields except 'shape'.")

        query = f"""
        SELECT {query_limit}{', '.join(fields)}, Shape.STAsText() as shape 
        FROM {table} 
        {spatial_query}
        """

        #logger.debug(f"Executing query: {query}")
        try:
            with sessionmaker(bind=engine)() as session:
                result = session.execute(text(query)).fetchall()
                #logger.info(f"Query executed successfully, retrieved {len(result)} rows.")
        except Exception as e:
            #logger.error(f"Query execution failed: {e}")
            raise

    try:
        data = pd.DataFrame(result, columns=fields + ['shapes'])
        #logger.debug("DataFrame created from query result.")
        data["geometry"] = data.shapes.apply(lambda x: loads(x))
        data.drop(columns=["shapes"], inplace=True)
        gdf = gpd.GeoDataFrame(data, geometry="geometry", crs=crs)
        #logger.info("GeoDataFrame created.")
        if clip and bounds is not None:
            gdf = gdf.clip(bounds)
            #logger.info("GeoDataFrame clipped to bounds.")
        return gdf
    except Exception as e:
        #logger.error(f"Error processing query results: {e}")
        raise