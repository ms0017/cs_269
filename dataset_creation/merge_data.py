import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime
import numpy as np

def coarsen_data(ds, latitude_step, longitude_step):
    # Define a function to round the value to the nearest whole number (for integer alignment)
    def round_to_integer(value):
        return np.floor(value)

    # Create latitude bins from the minimum to the maximum latitude, ensuring the bins are aligned to integer degrees
    lat_min = round_to_integer(ds.g0_lat_1.min())
    lat_max = round_to_integer(ds.g0_lat_1.max()) + latitude_step  # Ensure max is included

    # Create longitude bins from the minimum to the maximum longitude, ensuring the bins are aligned to integer degrees
    lon_min = round_to_integer(ds.g0_lon_2.min())
    lon_max = round_to_integer(ds.g0_lon_2.max()) + longitude_step  # Ensure max is included

    lat_bins = np.arange(lat_min, lat_max, latitude_step)
    lon_bins = np.arange(lon_min, lon_max, longitude_step)

    # First group by latitude bins and compute mean
    ds_lat_coarsened = ds.groupby_bins('g0_lat_1', lat_bins).mean(dim='g0_lat_1')

    # Then group by longitude bins and compute mean
    ds = ds_lat_coarsened.groupby_bins('g0_lon_2', lon_bins).mean(dim='g0_lon_2')

    return ds

def print_timestep(ds, variable, time):
    ds = ds.get(variable)
    ds = ds.sel(initial_time0_hours=time)
        
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.coastlines(resolution="10m")
    plot = ds.plot(
        cmap=plt.cm.coolwarm, transform=ccrs.PlateCarree(), cbar_kwargs={"shrink": 0.6}
    )

    time = datetime.strptime(str(ds.initial_time0_hours.data)[:26], "%Y-%m-%dT%H:%M:%S.%f")
    
    fig.title("ERA5 - Africa " + ds.long_name + ' ' + time.strftime("%B %d, %Y at %I:%M %p"))
    
    fig.savefig("era5_2m_temperature.png")

era5land = xr.open_dataset("data.grib", engine="pynio")
era5land = coarsen_data(era5land, 1, 1)

print_timestep(era5land, "STL1_GDS0_DBLY", "2019-01-01T00:00:00")