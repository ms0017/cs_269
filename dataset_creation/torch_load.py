import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class ERA5_With_Observations(Dataset):
    def __init__(self, csv_file, label_columns, date_column, lat_column, lon_column, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with data.
            label_columns (list of strings): List of column names for the labels/targets.
            date_column (string): The column name for the date.
            lat_column (string): The column name for latitude.
            lon_column (string): The column name for longitude.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.label_columns = label_columns  # Multiple label columns
        self.date_column = date_column
        self.lat_column = lat_column
        self.lon_column = lon_column
        self.transform = transform
        
        # Convert the date column to datetime and then to a timestamp (e.g., seconds since epoch)
        self.data[self.date_column] = pd.to_datetime(self.data[self.date_column], errors='coerce')
        self.data['date_timestamp'] = self.data[self.date_column].astype(int) / 10**9  # Convert to seconds since epoch
        
        # Group data by date and merge latitudes and longitudes as arrays
        self.date_groups = self.data.groupby('date_timestamp').agg({
            'latitude': list,
            'longitude': list,
            **{label: list for label in self.label_columns}  # Keep all label values as arrays
        }).reset_index()

    def __len__(self):
        return len(self.date_groups)

    def __getitem__(self, idx):
        # Extract the grouped row by index
        row = self.date_groups.iloc[idx]
        
        # Extract the date, latitude and longitude arrays, and labels arrays
        date = row['date_timestamp']
        latitude = np.array(row['latitude'])
        longitude = np.array(row['longitude'])
        
        # Convert lat/long arrays to a 2D array (if desired)
        location = np.column_stack((latitude, longitude))
        
        # Extract label arrays
        labels = {label: torch.tensor(row[label], dtype=torch.float32) for label in self.label_columns}
        
        # Combine into a sample dictionary
        sample = {
            'date': torch.tensor(date, dtype=torch.float32),
            'location': torch.tensor(location, dtype=torch.float32),  # Array of lat/long pairs
            'labels': labels  # Labels as arrays
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

# Define file path and columns
csv_file = 'dataset.csv'  # Replace with your file path
label_columns = ['not_recieved', 'low_availability',
       'high_availability', 'complete', 'STL1_GDS0_DBLY', '2T_GDS0_SFC',
       '2D_GDS0_SFC', 'STL2_GDS0_DBLY', 'STL3_GDS0_DBLY', 'SKT_GDS0_SFC',
       'STL4_GDS0_DBLY', 'population']
date_column = 'date'  # The column with dates
lat_column = 'latitude'  # The column with latitude
lon_column = 'longitude'  # The column with longitude

# Create Dataset from CSV
dataset = ERA5_With_Observations(csv_file=csv_file, 
                                label_columns=label_columns, 
                                date_column=date_column,
                                lat_column=lat_column,
                                lon_column=lon_column)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

print(len(dataset), len(dataloader), dataset[0])
