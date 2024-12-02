import pandas as pd
import datetime
import matplotlib.pyplot as plt
import imageio
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Create 'frames' directory if it doesn't exist
if not os.path.exists('frames'):
    os.makedirs('frames')

# Define the date range
start_date = datetime.date(2019, 1, 1)
end_date = datetime.date(2019, 3, 11)
delta = datetime.timedelta(days=1)

dates = []
current_date = start_date
while current_date <= end_date:
    dates.append(current_date)
    current_date += delta

images = []

for date in dates:
    filename = os.path.join('daily', 'data_' + date.strftime('%Y-%m-%d') + '.csv')
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        
        # Ensure the necessary columns exist
        if 'longitude' in df.columns and 'latitude' in df.columns and 'color code' in df.columns:
        
            # Extract stations in Africa based on longitude and latitude ranges
            df_africa = df[(df['longitude'] >= -20) & (df['longitude'] <= 55) & 
                           (df['latitude'] >= -40) & (df['latitude'] <= 40)]
            
            # Create the map plot
            fig = plt.figure(figsize=(10,10))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND)
            ax.add_feature(cfeature.OCEAN)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.add_feature(cfeature.COASTLINE)
            ax.set_extent([-20, 55, -40, 40])  # Set the extent to cover Africa
            
            # Plot the stations with their color codes
            scatter = ax.scatter(df_africa['longitude'], df_africa['latitude'], 
                                 c=df_africa['color code'], s=10, transform=ccrs.PlateCarree())
            plt.title('Data Availability on ' + date.strftime('%Y-%m-%d'))
            
            # Save the figure to the 'frames' directory
            frame_filename = 'frames/frame_' + date.strftime('%Y-%m-%d') + '.png'
            plt.savefig(frame_filename)
            plt.close()
            
            images.append(frame_filename)
        else:
            print(f"Columns 'longitude', 'latitude', or 'color code' not found in {filename}")
    else:
        print(f"File {filename} does not exist.")

# Create the GIF from the saved frames
with imageio.get_writer('data_availability.gif', mode='I', duration=0.5) as writer:
    for filename in images:
        image = imageio.imread(filename)
        writer.append_data(image)

print("GIF creation complete. Saved as 'data_availability.gif'.")