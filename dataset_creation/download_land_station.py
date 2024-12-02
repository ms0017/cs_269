import requests
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# Define the start and end dates
start_date = datetime(2019, 1, 1)
end_date = datetime(2019, 12, 31)
# end_date = datetime(2024, 12, 1)

# Define the periods
# periods = ["00", "06", "12", "18"]
periods = ["06", "18"]

# Calculate the number of days between start and end dates
delta = end_date - start_date

# Ensure the directory exists
os.makedirs('./12hour', exist_ok=True)

# Define the function that will handle the download for each date and period
def download_data(date, period):
    date_str = date.strftime('%Y-%m-%d')
    url = (
        'https://wdqms.wmo.int/wdqmsapi/v1/download/nwp/synop/six_hour/availability/'
        f'?date={date_str}&period={period}'
        '&variable=temperature&centers=DWD,ECMWF,JMA,NCEP&baseline=OSCAR'
    )

    # Send the GET request
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Determine the file extension based on the content type
        content_type = response.headers.get('content-type')
        if 'application/json' in content_type:
            ext = 'json'
        elif 'text/csv' in content_type:
            ext = 'csv'
        else:
            ext = 'txt'
        
        # Save the content to a file
        filename = f"data_{date_str}_{period}.{ext}"
        with open(os.path.join('./12hour', filename), 'wb') as file:
            file.write(response.content)
        print(f"Downloaded data for {date_str}_{period}")
    else:
        print(f"Failed to download data for {date_str}_{period}")

# Create a list of all date-period combinations to download
date_period_combinations = [
    (start_date + timedelta(days=i), period)
    for i in range(delta.days + 1)
    for period in periods
]

# Use ThreadPoolExecutor to download data concurrently
with ThreadPoolExecutor() as executor:
    executor.map(lambda dp: download_data(dp[0], dp[1]), date_period_combinations)
