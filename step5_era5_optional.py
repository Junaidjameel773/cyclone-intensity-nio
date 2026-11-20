"""
STEP 5 (OPTIONAL - do this LAST, only if steps 1-4 already work)

ERA5 gives you the environment the storm sat in: sea surface temperature and
vertical wind shear are the two strongest physical predictors of whether a
cyclone will intensify. Adding them is the honest "multi-modal" part of the
project - but it is also the slowest part, so do not start here.

Before running:
  1. Make a free account at https://cds.climate.copernicus.eu
  2. Open https://cds.climate.copernicus.eu/how-to-api and copy your token
  3. Create ~/.cdsapirc containing:
         url: https://cds.climate.copernicus.eu/api
         key: <YOUR-PERSONAL-ACCESS-TOKEN>
  4. Open the ERA5 single-levels and pressure-levels dataset pages and click
     "Accept terms" at the bottom of the download form. Requests fail with a
     403 until you do this, and the error message is not obvious.
  5. pip install cdsapi

Requests sit in a queue - a month of data can take minutes to hours. Ask for
ONE month at a time, only the months that actually contain cyclones, and only
the North Indian Ocean box.

Run:  python step5_era5_optional.py --year 2007 --month 6
"""

import argparse
import config

AREA = [30, 50, 0, 100]     # North, West, South, East - the NI basin box
SINGLE_VARS = ["sea_surface_temperature", "10m_u_component_of_wind",
               "10m_v_component_of_wind", "mean_sea_level_pressure"]
LEVEL_VARS = ["u_component_of_wind", "v_component_of_wind"]
LEVELS = ["200", "850"]     # shear = wind at 200 hPa minus wind at 850 hPa


def main(year: int, month: int):
    import cdsapi
    c = cdsapi.Client()
    out = config.RAW / "era5"
    out.mkdir(exist_ok=True)
    ym = f"{year}{month:02d}"
    days = [f"{d:02d}" for d in range(1, 32)]
    times = ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"]

    c.retrieve("reanalysis-era5-single-levels", {
        "product_type": ["reanalysis"], "variable": SINGLE_VARS,
        "year": [str(year)], "month": [f"{month:02d}"], "day": days,
        "time": times, "area": AREA, "data_format": "netcdf",
    }, str(out / f"era5_single_{ym}.nc"))

    c.retrieve("reanalysis-era5-pressure-levels", {
        "product_type": ["reanalysis"], "variable": LEVEL_VARS,
        "pressure_level": LEVELS,
        "year": [str(year)], "month": [f"{month:02d}"], "day": days,
        "time": times, "area": AREA, "data_format": "netcdf",
    }, str(out / f"era5_levels_{ym}.nc"))

    print(f"Saved ERA5 for {ym} in {out}")
    print("""
Next: sample these grids at each storm's lat/lon/time and add the columns
to the manifest, then add their names to config.TAB_COLS. Sketch:

    import xarray as xr, pandas as pd, config
    man = pd.read_csv(config.MANIFEST, parse_dates=["ISO_TIME"])
    sl  = xr.open_dataset("data/raw/era5/era5_single_200706.nc")
    lv  = xr.open_dataset("data/raw/era5/era5_levels_200706.nc")

    pts = dict(latitude=xr.DataArray(man.LAT, dims="i"),
               longitude=xr.DataArray(man.LON, dims="i"),
               valid_time=xr.DataArray(man.ISO_TIME, dims="i"))

    man["SST"] = sl.sst.sel(**pts, method="nearest").values - 273.15
    u = lv.u.sel(**pts, method="nearest"); v = lv.v.sel(**pts, method="nearest")
    man["SHEAR"] = ((u.sel(pressure_level=200) - u.sel(pressure_level=850))**2 +
                    (v.sel(pressure_level=200) - v.sel(pressure_level=850))**2) ** 0.5

Check the coordinate names first - recent ERA5 files use `valid_time` and
`pressure_level`, older ones use `time` and `level`.
""")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--month", type=int, required=True)
    a = p.parse_args()
    main(a.year, a.month)
