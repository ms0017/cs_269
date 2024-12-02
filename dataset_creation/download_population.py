import subprocess

# Define the URL and the output file name
url = "https://data.worldpop.org/GIS/Population/Global_2000_2020/2019/0_Mosaicked/ppp_2019_1km_Aggregated.tif"
output_filename = "ppp_2019_1km_Aggregated.tif"

# Run the wget command using subprocess
subprocess.run(["wget", url, "-O", output_filename])
