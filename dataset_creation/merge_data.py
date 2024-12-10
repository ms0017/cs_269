import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime
import numpy as np
import pandas as pd
import os
import datetime
import rasterio
from concurrent.futures import ThreadPoolExecutor


# Bin_Size
coarsen_factor = 2

long_bins = np.arange(-20, 55, coarsen_factor)
lat_bins = np.arange(-40, 40, coarsen_factor)

print(len(long_bins), len(lat_bins), len(long_bins) * len(lat_bins))


def coarsen_data(ds, population=False):
    if isinstance(ds, xr.Dataset):
        # Group by latitude bins and calculate mean
        ds_lat_coarsened = ds.groupby_bins('latitude', lat_bins).mean(dim='latitude')
        
        # Group by longitude bins and calculate mean
        ds_coarsened = ds_lat_coarsened.groupby_bins('longitude', long_bins).mean(dim='longitude')
        
        # Extract upper-right corners of bins
        upper_latitudes = [bin.right for bin in ds_coarsened['latitude_bins'].values]
        upper_longitudes = [bin.right for bin in ds_coarsened['longitude_bins'].values]
        
        # Assign new coordinates to the coarsened Dataset
        ds_coarsened = ds_coarsened.assign_coords({
            'latitude': ('latitude_bins', upper_latitudes),
            'longitude': ('longitude_bins', upper_longitudes)
        })
        
        # Drop the bins and keep only coordinates and data
        ds_coarsened = ds_coarsened.drop_vars(['latitude_bins', 'longitude_bins'])
        
        return ds_coarsened
    else:
        ds['latitude'] = pd.cut(ds['latitude'], bins=lat_bins, include_lowest=True, right=True).apply(lambda x: x.right)
        ds['longitude'] = pd.cut(ds['longitude'], bins=long_bins, include_lowest=True, right=True).apply(lambda x: x.right)

        if population:
            # Group by the bins and compute the mean for each bin
            return ds.groupby(['longitude', 'latitude']).agg({
                'population': 'sum'
            }).reset_index()
        else:
            return ds.groupby(['longitude', 'latitude']).agg({
                'score': 'sum',
                'elevation_ft': 'sum',
            }).reset_index()
        
def density_based_grid(
    data, value_cols, distance_threshold=0.03, sigma=None
):
    """
    Distributes values in a DataFrame based on a density algorithm across a geographic grid for multiple value columns.
    
    Parameters:
    - data (pd.DataFrame): Input DataFrame containing longitude, latitude, and value columns.
    - value_cols (list): List of column names for values to be distributed.
    - distance_threshold (float): Maximum distance (in degrees) for influence of a point.
    - sigma (float, optional): Standard deviation for Gaussian weighting. Defaults to `distance_threshold / 3`.
    
    Returns:
    - result_df (pd.DataFrame): DataFrame with bins, aggregated density values for each value column,
      longitude, latitude representing the top-right corner of each bin, and associated dates.
    """
    
    # Validate value_cols input
    if not isinstance(value_cols, list) or len(value_cols) == 0:
        raise ValueError("`value_cols` must be a non-empty list of column names.")
    
    # Initialize records list
    grid_records = []
    
    # Set default sigma if not provided
    if sigma is None:
        sigma = distance_threshold / 3

    # Define Gaussian weight function
    def gaussian_weight(distance, sigma):
        return np.exp(-0.5 * (distance / sigma) ** 2)

    # Iterate through each row in the data
    for _, row in data.iterrows():
        lon, lat = row['longitude'], row['latitude']
        
        # Iterate over all bins
        for i in range(len(long_bins) - 1):
            for j in range(len(lat_bins) - 1):
                # Bin top-right corner
                lon_bin_top_right = long_bins[i + 1]
                lat_bin_top_right = lat_bins[j + 1]

                # Calculate distance from the point to the bin center
                bin_center_lon = (long_bins[i] + lon_bin_top_right) / 2
                bin_center_lat = (lat_bins[j] + lat_bin_top_right) / 2
                distance = np.sqrt((lon - bin_center_lon) ** 2 + (lat - bin_center_lat) ** 2)
                
                if distance <= distance_threshold:
                    # Apply Gaussian weight
                    weight = gaussian_weight(distance, sigma)
                    
                    # Prepare the record with all value_cols and date
                    record = {
                        "longitude": lon_bin_top_right,
                        "latitude": lat_bin_top_right,
                    }
                    record.update({value: row[value] * weight for value in value_cols})
                    
                    # Append to grid records
                    grid_records.append(record)

    # Create DataFrame from the records
    grid_df = pd.DataFrame(grid_records)

    # Aggregate results into a single DataFrame
    agg_columns = {value: "sum" for value in value_cols}
    result_df = grid_df.groupby(["longitude", "latitude"], as_index=False).agg(agg_columns)
    
    return result_df

def count_grid(df, target, target_values, value_names):
    """
    Count occurrences of target values within specified latitude and longitude bins.
    
    Parameters:
    df (pd.DataFrame): DataFrame with 'latitude', 'longitude', and target columns.
    target (str): Column to filter by specific values.
    target_values (list): List of values to count.
    value_names (list): Descriptive names for each target value.
    lat_bins (array-like): Latitude bin boundaries.
    long_bins (array-like): Longitude bin boundaries.
    
    Returns:
    pd.DataFrame: A DataFrame with counts of target values per bin.
    """
    assert len(target_values) == len(value_names), "Target values and value names lengths must match."

    df['latitude'] = pd.cut(df['latitude'], bins=lat_bins, include_lowest=True, right=True).apply(lambda x: x.right)
    df['longitude'] = pd.cut(df['longitude'], bins=long_bins, include_lowest=True, right=True).apply(lambda x: x.right)

    result_df = None

    for value, name in zip(target_values, value_names):
        filtered_df = df[df[target] == value]
        count_grid = (
            filtered_df.groupby(['latitude', 'longitude', 'date'])
            .size()
            .reset_index(name=name)
        )

        if result_df is None:
            result_df = count_grid
        else:
            result_df = pd.merge(result_df, count_grid, how="outer", on=['latitude', 'longitude', 'date'])

    return result_df

def density_based_count(
    df, target, target_values, value_names, distance_threshold=0.03, sigma=None
):
    """
    Calculate density-based counts of target values within specified latitude and longitude bins using multi-threading.
    
    Parameters:
    - df (pd.DataFrame): DataFrame with 'latitude', 'longitude', and target columns.
    - target (str): Column to filter by specific target values.
    - target_values (list): List of target values to count.
    - value_names (list): Descriptive names for each target value.
    - long_bins (array-like): Longitude bin boundaries.
    - lat_bins (array-like): Latitude bin boundaries.
    - distance_threshold (float): Maximum distance (in degrees) for influence of a point.
    - sigma (float, optional): Standard deviation for Gaussian weighting. Defaults to `distance_threshold / 3`.

    Returns:
    - result_df (pd.DataFrame): DataFrame with density-based counts per bin for each target value.
    """
    assert len(target_values) == len(value_names), "Target values and value names lengths must match."

    # Set default sigma if not provided
    if sigma is None:
        sigma = distance_threshold / 3

    # Define Gaussian weight function
    def gaussian_weight(distance, sigma):
        return np.exp(-0.5 * (distance / sigma) ** 2)

    # Function to process a single target value
    def process_target(value, name):
        filtered_df = df[df[target] == value]
        grid_records = []

        for _, row in filtered_df.iterrows():
            lon, lat, date = row['longitude'], row['latitude'], row['date']

            # Iterate over all bins
            for i in range(len(long_bins) - 1):
                for j in range(len(lat_bins) - 1):
                    # Bin top-right corner
                    lon_bin_top_right = long_bins[i + 1]
                    lat_bin_top_right = lat_bins[j + 1]

                    # Calculate distance from the point to the bin center
                    bin_center_lon = (long_bins[i] + lon_bin_top_right) / 2
                    bin_center_lat = (lat_bins[j] + lat_bin_top_right) / 2
                    distance = np.sqrt((lon - bin_center_lon) ** 2 + (lat - bin_center_lat) ** 2)

                    if distance <= distance_threshold:
                        # Apply Gaussian weight
                        weight = gaussian_weight(distance, sigma)

                        # Append the density-based count record
                        grid_records.append({
                            "longitude": lon_bin_top_right,
                            "latitude": lat_bin_top_right,
                            "date": date,
                            name: weight
                        })

        # Create DataFrame for the current target value
        grid_df = pd.DataFrame(grid_records)

        # Aggregate density-based counts
        return grid_df.groupby(["longitude", "latitude", "date"], as_index=False).agg({name: "sum"})

    # Use ThreadPoolExecutor for multi-threading
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_target, value, name)
            for value, name in zip(target_values, value_names)
        ]
        results = [future.result() for future in futures]

    # Merge results from all threads
    result_df = pd.concat(results, ignore_index=True)

    # Aggregate across all values (optional, depends on whether duplicate bins are possible)
    result_df = result_df.groupby(["longitude", "latitude", "date"], as_index=False).sum()

    return result_df

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
        'initial_time0_hours': 'date',
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
        
        # Combine the band data into a DataFrame
        df = pd.DataFrame({
            'longitude': lon,
            'latitude': lat,
            'population': bands[0],
        })
    df_africa = df[(df['longitude'] >= -20) & (df['longitude'] <= 55) & 
                (df['latitude'] >= -40) & (df['latitude'] <= 40)]
    return df_africa

def gather_flights(filename):
    # Corrected line, without the 'with' statement
    dataset = pd.read_csv(filename, on_bad_lines='skip', sep=',')
    dataset.rename(columns={'latitude_deg': 'latitude', 'longitude_deg': 'longitude'}, inplace=True)
    
    # Continue processing your dataset
    return dataset

#flights = gather_flights('flights/af-airports.csv')
#flights = density_based_grid(flights, ['score'], distance_threshold=coarsen_factor * 3.5)
#flights = flights.drop_duplicates(subset=['longitude', 'latitude']).dropna()
#print('finished flights')

population = gather_population('ppp_2019_1km_Aggregated.tif')
population = coarsen_data(population, population=True)
population = population.drop_duplicates(subset=['longitude', 'latitude']).dropna()
print('finished population')

era5land = gather_era5_data("data.grib")
era5land = coarsen_data(era5land).to_dataframe().reset_index()
era5land['date'] = pd.to_datetime(era5land['date'])
era5land = era5land.drop_duplicates(subset=['date', 'longitude', 'latitude'])
era5land = era5land.dropna().drop(columns=['longitude_bins', 'latitude_bins', 'initial_time3_hours', 'forecast_time4', 'initial_time3_encoded', 'initial_time0_encoded'])
#print(era5land.columns, population.columns)
print('finished era5')

# yellow gray purple, what do those mean
land_stations = gather_land_station_data(datetime.date(2019, 1, 1), datetime.date(2019, 1, 31), ['06', '18'])
#land_station = count_grid(land_stations, 'color code', ['black', 'orange', 'red', 'green'], ['not_recieved', 'low_availability', 'high_availability', 'complete'])
land_station = density_based_count(land_stations, 'color code', ['black', 'orange', 'red', 'green'], ['not_recieved', 'low_availability', 'high_availability', 'complete'], distance_threshold=coarsen_factor * 3.5)
land_station['date'] = pd.to_datetime(land_station['date'])
land_station = land_station.drop_duplicates(subset=['date', 'longitude', 'latitude'])
land_station = land_station.dropna()
print('finished land stations')
print(land_station)
print(land_station.columns, era5land.columns, population.columns)

merged_land = pd.merge(land_station, era5land, how='outer', on=['latitude', 'longitude', 'date'])
merged = pd.merge(merged_land, population, how='outer', on=['latitude', 'longitude'])
#merged = pd.merge(merged, flights, how='outer', on=['latitude', 'longitude'])

print(merged.columns, len(merged), len(merged['date'].unique()), len(merged['latitude'].unique()), len(merged['longitude'].unique()))
merged.to_csv('dataset.csv', index=False)

#print_timestep(era5land, 'population', '2019-12-31 18:00:00')