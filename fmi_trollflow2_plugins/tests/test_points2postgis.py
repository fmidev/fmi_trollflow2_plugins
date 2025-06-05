"""Test the points2postgis plugin."""

import os
from unittest import mock

import dask.array as da
import numpy as np
import pytest
import xarray as xr
import yaml
from satpy import Scene

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def read_config(fname):
    """Read yaml config file from the bundled etc/ directory."""
    full_path = os.path.join(THIS_DIR, "etc", fname)
    with open(full_path) as fid:
        config = yaml.safe_load(fid)

    return config


def create_li_lfl_scene():
    """Create a Satpy Scene with simulated LI LFL data."""
    scn = Scene()
    # First two points are within "euro4" area, the other two outside
    scn["latitude"] = xr.DataArray(
        da.array([60., 42., -60, -42], dtype=np.float32)
    )
    scn["longitude"] = xr.DataArray(
        da.array([25., 10., 0, 42], dtype=np.float32)
    )
    scn["flash_time"] = xr.DataArray(
        da.array(np.array([
            "2025-01-30T10:50:08.405699968", "2025-01-30T10:50:08.489699968",
            "2025-01-30T10:50:08.540699904", "2025-01-30T10:50:08.587699968"],
            dtype="datetime64[ns]")
        )
    )
    scn["radiance"] = xr.DataArray(
        da.array([10., 100., 1000., 2000.], dtype=np.float32)
    )
    scn["flash_footprint"] = xr.DataArray(
        da.array([2., 18., 5., 19.], dtype=np.float32)
    )
    scn["flash_duration"] = xr.DataArray(
        da.array([300000000, 606000000, 228000000, 602000000],
        dtype="timedelta64[ns]")
    )

    return scn


def create_job(fname, scene):
    """Create a Trollflow2 job."""
    job = {
        "product_list": read_config(fname),
        "scene": scene,
    }
    return job


EXPECTED_LI_AREA_NONE = (
    ["2025-01-30 10:50:08.405700+00:00",
     np.float32(10.0), np.int16(2), np.uint16(300), np.float32(60.0), np.float32(25.0)],
    ["2025-01-30 10:50:08.489700+00:00",
     np.float32(100.0), np.int16(18), np.uint16(606), np.float32(42.0), np.float32(10.0)],
    ["2025-01-30 10:50:08.540700+00:00",
     np.float32(1000.0), np.int16(5), np.uint16(228), np.float32(-60.0), np.float32(0.0)],
    ["2025-01-30 10:50:08.587700+00:00",
     np.float32(2000.0), np.int16(19), np.uint16(602), np.float32(-42.0), np.float32(42.0)]
)
EXPECTED_LI_AREA_EURO4 = (
    ["2025-01-30 10:50:08.405700+00:00",
     np.float32(10.0), np.int16(2), np.uint16(300), np.float32(60.0), np.float32(25.0)],
    ["2025-01-30 10:50:08.489700+00:00",
     np.float32(100.0), np.int16(18), np.uint16(606), np.float32(42.0), np.float32(10.0)],
)

@pytest.mark.parametrize(("config_fname", "expected_values"),
                         [("trollflow2_points2postgis_li_area_none.yaml", EXPECTED_LI_AREA_NONE),
                          ("trollflow2_points2postgis_li_area_euro4.yaml", EXPECTED_LI_AREA_EURO4)])
def test_points2postgis_li(config_fname, expected_values):
    """Test points2postgis plugin with LI data."""
    from fmi_trollflow2_plugins import points2postgis

    scene = create_li_lfl_scene()
    job = create_job(config_fname, scene)

    conn = mock.MagicMock()
    cur = mock.MagicMock()
    # conn.__enter__.cursor.return_value.__enter__.return_value = cur
    conn.__enter__.return_value = cur
    with mock.patch("fmi_trollflow2_plugins.points2postgis.get_database_connection") as gdc:
        gdc.return_value = conn
        points2postgis(job)
    execute_calls = cur.cursor.return_value.__enter__.return_value.execute.mock_calls

    insert_str = job["product_list"]["product_list"]["postgis"]["insert_str"]
    for i, call in enumerate(execute_calls):
        assert call.args == (insert_str, expected_values[i])
