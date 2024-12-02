import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime
import numpy as np
import pandas as pd
import os
import datetime
import rasterio


# Bin_Size
coarsen_factor = 5

long_bins = np.arange(-20, 55, coarsen_factor)
lat_bins = np.arange(-40, 40, coarsen_factor)

print(len(long_bins), len(lat_bins), len(long_bins) * len(lat_bins))


def coarsen_data(ds):
    if isinstance(ds, xr.Dataset):
        ds_lat_coarsened = ds.groupby_bins('latitude', lat_bins).mean(dim='latitude')
        return ds_lat_coarsened.groupby_bins('longitude', long_bins).mean(dim='longitude')
    else:
        ds['longitude_bin'] = pd.cut(ds['longitude'], bins=long_bins, include_lowest=True)
        ds['latitude_bin'] = pd.cut(ds['latitude'], bins=lat_bins, include_lowest=True)

        # Group by the bins and compute the mean for each bin
        return ds.groupby(['longitude_bin', 'latitude_bin']).agg({
            'value': 'mean'
        }).reset_index()

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

def gather_era5_data(file_name):
    era5land = xr.open_dataset("data.grib", engine="pynio")
    era5land = era5land.rename({
        'initial_time0_hours': 'time',
        'g0_lat_1': 'latitude',
        'g0_lon_2': 'longitude'
    })
    
    return era5land

def gather_land_station_data(start_date, end_date, periods):
    # Define the date range and periods
    delta = datetime.timedelta(days=1)
    
    columns = ['name', 'wigosid', 'country code', 'in OSCAR', 'longitude', 'latitude',
           '#received', '#expected', 'default schedule', 'color code',
           'description', 'variable', 'date', 'center']

    # Create an empty DataFrame to store the results
    combined_df = pd.DataFrame(columns=columns)

    # Generate the dates list
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += delta

    # Loop through each date and period
    for date in dates:
        for period in periods:
            filename = os.path.join('12hour', 'data_' + date.strftime('%Y-%m-%d') + f'_{period}' + '.csv')
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                
                # Ensure the necessary columns exist
                if 'longitude' in df.columns and 'latitude' in df.columns and 'color code' in df.columns:
                    # Extract stations in Africa based on longitude and latitude ranges
                    df_africa = df[(df['longitude'] >= -20) & (df['longitude'] <= 55) & 
                                (df['latitude'] >= -40) & (df['latitude'] <= 40)]
                    
                    # Add the date and period columns to the DataFrame
                    datetime_str = f"{date.strftime('%Y-%m-%d')}T{int(period):02d}:00:00"
                    df_africa.loc[:, 'date'] = datetime_str

                    # Append the filtered data to the combined DataFrame
                    combined_df = pd.concat([combined_df, df_africa[columns]])

    return combined_df

def gather_population(filename):
    with rasterio.open(filename) as dataset:
        # Read the data from all bands
        data = dataset.read()  # Read all bands (multi-dimensional array)
        
        # Get the metadata
        transform = dataset.transform  # Affine transform for coordinates
        width = dataset.width
        height = dataset.height
        
        # Create a 2D array of pixel coordinates (row, col)
        rows, cols = np.indices((height, width))
        
        # Convert pixel indices to geographic coordinates
        lon, lat = rasterio.transform.xy(transform, rows, cols)
        
        # Flatten the arrays to make a 1D DataFrame
        lon = np.array(lon).flatten()
        lat = np.array(lat).flatten()
        
        # Flatten the data array for each band and stack them into columns
        bands = [data[i].flatten() for i in range(data.shape[0])]
        band_names = [f'band_{i+1}' for i in range(data.shape[0])]  # Create band names
        
        # Combine the band data into a DataFrame
        df = pd.DataFrame({
            'longitude': lon,
            'latitude': lat,
            'value': bands[0],
            #**{band_names[i]: bands[i] for i in range(len(bands))}
        })
    df_africa = df[(df['longitude'] >= -20) & (df['longitude'] <= 55) & 
                (df['latitude'] >= -40) & (df['latitude'] <= 40)]
    return df_africa

population = gather_population('ppp_2019_1km_Aggregated.tif')
print(len(population))
population = coarsen_data(population)
print(len(population))

era5land = gather_era5_data("data.grib")
era5land = coarsen_data(era5land)

land_df = gather_land_station_data(datetime.date(2019, 1, 1), datetime.date(2019, 1, 31), ['06', '18'])
#print_timestep(era5land, "STL1_GDS0_DBLY", "2019-01-01T00:00:00")