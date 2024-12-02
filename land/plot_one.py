import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches

# Load the CSV file into a DataFrame
# Replace 'your_file.csv' with the actual path to your CSV file
df = pd.read_csv('wdqms_gbon_synop_availability_daily_temperature_2024-11-04.csv')

# Define the geographical boundaries for Africa
# Latitude: from -40 to 40, Longitude: from -20 to 55
africa_lat_min, africa_lat_max = -40, 40
africa_lon_min, africa_lon_max = -20, 55

# Filter stations within Africa
africa_stations = df[(df['latitude'] >= africa_lat_min) & (df['latitude'] <= africa_lat_max) &
                     (df['longitude'] >= africa_lon_min) & (df['longitude'] <= africa_lon_max)]

# Check the first few rows to confirm data loading (optional)
print(africa_stations.head())

# Set up the map with Cartopy
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# Set the extent to focus on Africa
ax.set_extent([africa_lon_min, africa_lon_max, africa_lat_min, africa_lat_max], crs=ccrs.PlateCarree())

# Add geographic features
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.gridlines(draw_labels=True)

# Plot each climate station
for index, row in africa_stations.iterrows():
    # station = row['Station']
    color_code = row['color code']
    lat = row['latitude']
    lon = row['longitude']
    
    # Plot each station with its color code
    ax.plot(lon, lat, marker='o', color=color_code, markersize=5, transform=ccrs.PlateCarree())
    # ax.text(lon + 0.5, lat, station, transform=ccrs.PlateCarree(), fontsize=8)

# Create a legend based on unique color codes and descriptions
legend_items = africa_stations[['color code', 'description']].drop_duplicates()
legend_patches = [mpatches.Patch(color=row['color code'], label=row['description']) for _, row in legend_items.iterrows()]
plt.legend(handles=legend_patches, loc='upper right', title="Station Description")

plt.title("Climate Stations in Africa")
plt.show()