"""Trollflow2 plugin to push point data data to Postgis databases."""

import datetime as dt
import logging
import os
from contextlib import contextmanager

import numpy as np
import psycopg
from satpy.area import get_area_def

logger = logging.getLogger(__name__)


def to_int16(data):
    """Convert the data to int16."""
    return data.astype(np.int16)


def to_uint16(data):
    """Convert the data to uint16."""
    return data.astype(np.uint16)


def divide_by_million(data):
    """Divide the data by a million."""
    return data / 1e6


def datetime64_to_str(times):
    """Convert datetime64 array to a list of PostGIS-recognized strings."""
    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    return [str(dt.datetime.fromtimestamp((t - unix_epoch) / one_second, dt.timezone.utc)) for
            t in times]

def datetime_to_str(times):
    """Convert datetimes to PostGIS-recognized strings."""
    if not isinstance(times, (list, tuple)):
        times = [times]
    return [str(t) for t in times]

CONVERSIONS = {
    "to_int16": to_int16,
    "to_uint16": to_uint16,
    "divide_by_million": divide_by_million,
    "datetime64_to_str": datetime64_to_str,
    "datetime_to_str": datetime_to_str,
}


def points2postgis(job):
    """Store point data from Satpy Scene to PostGIS database."""
    # Data insertion related settings
    insert_str = job["product_list"]["product_list"]["postgis"]["insert_str"]
    conversions = job["product_list"]["product_list"]["postgis"]["conversions"]

    # Database connection. Credentials are read from environment variables
    host = job["product_list"]["product_list"]["postgis"]["host"]
    port = job["product_list"]["product_list"]["postgis"]["port"]
    dbname = job["product_list"]["product_list"]["postgis"]["database_name"]

    area_def_name = list(job["product_list"]["product_list"]["areas"].keys())[0]
    if area_def_name == "None":
        area_def = None
    else:
        area_def = get_area_def(area_def_name)

    data = _convert_and_compute_data_from_scene(job["scene"], conversions)

    with get_database_connection(host, port, dbname) as conn:
        with conn.cursor() as cur:
            _store_data(cur, data, insert_str, area_def)
        conn.commit()


def _convert_and_compute_data_from_scene(scn, dataset_conversions):
    scn2 = scn.compute()
    data = {}
    for dset, conversions in dataset_conversions.items():
        if dset in scn2:
            val = scn2[dset].data
        elif dset in ("nominal_time", "start_time"):
            val = scn2.start_time
        else:
            raise AttributeError(f"No dataset {dset}.")
        data[dset] = _apply_conversions(val, conversions)
    return data


@contextmanager
def get_database_connection(host, port, dbname):
    """Create a connection to PostGIS database."""
    user = os.environ["POSTGIS_USER"]
    password = os.environ["POSTGIS_PASSWORD"]

    connstr = f"host={host} port={port} dbname={dbname} user={user} password={password}"

    conn = psycopg.connect(connstr)
    try:
        yield conn
    finally:
        conn.close()


def _store_data(cur, data, insert_str, area_def):
    num_stored = 0
    num_received = len(data["longitude"])
    for i in range(num_received):
        if _data_outside_area(data["longitude"][i], data["latitude"][i], area_def):
            continue
        values = []
        for key in data.keys():
            try:
                val = data[key][i]
            except IndexError:
                val = data[key][0]
            values.append(val)
        cur.execute(insert_str, values)
        num_stored += 1

    logger.info(f"Stored {num_stored} of {num_received} observations.")


def _data_outside_area(longitude, latitude, area_def):
    if area_def is None:
        return False
    if (longitude, latitude) in area_def:
        return False
    return True


def _apply_conversions(data, conversions):
    for conversion in conversions:
        data = CONVERSIONS[conversion](data)
    return data
