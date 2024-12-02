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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract row by index
        row = self.data.iloc[idx]
        
        # Separate features and labels
        features = row.drop(self.label_columns)  # Drop all label columns
        labels = {label: row[label] for label in self.label_columns}  # Extract multiple labels
        
        # Extract date, latitude, and longitude as separate indices (or as features)
        date = row[self.date_column]
        latitude = row[self.lat_column]
        longitude = row[self.lon_column]

        # Combine features and indices into a single dictionary
        sample = {
            'date': torch.tensor(date, dtype=torch.float32),  # Convert to tensor
            'latitude': torch.tensor(latitude, dtype=torch.float32),
            'longitude': torch.tensor(longitude, dtype=torch.float32),
            'features': torch.tensor(features.drop([self.date_column, self.lat_column, self.lon_column]).values, dtype=torch.float32),
            'labels': {key: torch.tensor(value, dtype=torch.float32) for key, value in labels.items()}
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

# Define file path and columns
csv_file = 'dataset.csv'  # Replace with your file path
label_columns = ['not recieved', 'low_availability',
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
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print(len(dataset), len(dataloader))
