import pandas as pd
from os import makedirs

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime

import numpy as np

coarsen_factor = 5
long_bins = np.arange(-20, 55, coarsen_factor)
lat_bins = np.arange(-40, 40, coarsen_factor)

def coarsen_data(df, coarsen_factor=5):
    """
    Coarsen the DataFrame's latitude and longitude into bins of the specified coarsen factor.

    Parameters:
    - df: pandas DataFrame containing 'longitude', 'latitude' columns.
    - coarsen_factor: int, the factor by which to coarsen the grid (default is 5 for 5°x5° bins).

    Returns:
    - DataFrame with binned latitude and longitude.
    """
    # Define bins for latitude and longitude
    long_bins = np.arange(-20, 55, coarsen_factor)
    lat_bins = np.arange(-40, 40, coarsen_factor)
    
    # Bin the latitude and longitude columns
    df['longitude_bin'] = pd.cut(df['longitude'], bins=long_bins, include_lowest=True)
    df['latitude_bin'] = pd.cut(df['latitude'], bins=lat_bins, include_lowest=True)
    
    return df, long_bins, lat_bins

def plot_5deg_grid(df, variable, time):
    """
    Plot a specific timestep from a DataFrame with latitude and longitude representing top-right corners of 5°x5° grid cells.
    
    Parameters:
    - df: pandas DataFrame containing 'longitude', 'latitude', 'date', and the variable column.
    - variable: str, the variable name to plot.
    - time: datetime-like, the time to plot.
    """
    # Coarsen the data
    df, long_bins, lat_bins = coarsen_data(df)
    
    # Filter DataFrame for the specified time
    df_filtered = df[df['date'] == pd.Timestamp(time)]

    if df_filtered.empty:
        print(f"No data available for {time}.")
        return

    # Pivot table to create a grid for plotting
    grid = df_filtered.pivot_table(
        index='latitude_bin', columns='longitude_bin', values=variable, fill_value=np.nan
    )

    # Extract the bin boundaries from the intervals
    lon_bins = [interval.mid for interval in grid.columns]
    lat_bins = [interval.mid for interval in grid.index]
    values_grid = grid.values

    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.coastlines(resolution="10m")

    # Create meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(lon_bins, lat_bins)

    mesh = ax.pcolormesh(
        lon_grid, lat_grid, values_grid,
        cmap=plt.cm.coolwarm, transform=ccrs.PlateCarree(), shading='auto'
    )
    
    ax.set_extent([-20, 55, -40, 40], crs=ccrs.PlateCarree())

    plt.colorbar(mesh, ax=ax, shrink=0.6, label=variable)
    formatted_time = time.strftime("%B %d, %Y at %I:%M %p")
    plt.title(f"ERA5 - Africa {variable} on {formatted_time}")
    fig.savefig(f"{variable}_{formatted_time.replace(' ', '_').replace(',', '')}.png")
    plt.show()

# Read CSV file
df = pd.read_csv('dataset.csv')  # Replace 'file.csv' with your file's path
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['date'] = pd.to_datetime(df['date'])
print(df.columns, df['date'][0])


makedirs('figures', exist_ok=True)
plot_5deg_grid(df, 'population', df['date'][0])
#print_timestep(df, 'STL1_GDS0_DBLY', df['date'][0])
