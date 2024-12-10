# Counteracting Data Availability Bias in Learning-based Climate Models

## Installation

## Running Models

```python
python3 dataset_creation/download_era5.py
python3 dataset_creation/dowload_land_station.py
python3 dataset_creation/download_population.py
```

Then, unzip the era5 and make sure file names match within merge_data

In terms of options, you can set the coarsening factor and whether to use density-based or summation-based grid method.

```python
python3 merge_data.py
```

This should create a dataset.csv file (assuming that all inputs are specified and downloaded). This can be directly plugged into the torch models.