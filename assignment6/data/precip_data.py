
import os
import pandas as pd

# Path to the folder containing Excel files
folder_path = "precip/"

# List to store individual DataFrames
dfs = []

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        # Read each Excel file into a DataFrame, skipping the first two rows
        df = pd.read_csv(file_path, skiprows=9)
        # df = df[["Month", "Area Affected"]]
        # Append the DataFrame to the list
        dfs.append(df)

# Combine all DataFrames into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)
monthly_avg = combined_df.groupby(['LAT', 'LON', 'MO'])['PRECTOTCORR'].mean().reset_index()


from geopy.geocoders import Nominatim

# Create a geolocator object
geolocator = Nominatim(user_agent="my_geocoder")

def get_us_state(lat, lon):
    geolocator = Nominatim(user_agent="get_us_state")
    location = geolocator.reverse((lat, lon), language='en')

    if location and 'address' in location.raw:
        address = location.raw['address']
        if 'state' in address:
            print(address['state'])
            return address['state']
        elif 'state_district' in address:
            return address['state_district']

    return None

# Apply the function to create a new 'STATE' column
# monthly_avg['STATE'] = monthly_avg.apply(lambda row: get_us_state(row['LAT'], row['LON']), axis=1)

#%%

import geopandas as gpd
from shapely.geometry import Point

# Assuming your DataFrame is named monthly_avg
geometry = [Point(lon, lat) for lon, lat in zip(monthly_avg['LON'], monthly_avg['LAT'])]

# Create a GeoDataFrame
geo_df = gpd.GeoDataFrame(monthly_avg, geometry=geometry)

# Now, you can drop the original 'LAT' and 'LON' columns if you want
geo_df = geo_df.drop(['LAT', 'LON'], axis=1)

# Print the GeoDataFrame
print(geo_df)



df = pd.read_csv('plants_inblocks_eq_fire.csv')



#%%

valid_states = ['Maine', 'New Hampshire', 'Vermont', 'Massachusetts', 'Rhode Island', 'Connecticut', 'New York', 'New Jersey', 'Pennsylvania']
monthly_avg = monthly_avg[combined_df['STATE'].isin(valid_states)]

# Replace state names with short state names
state_name_mapping = {
    'Maine': 'ME',
    'New Hampshire': 'NH',
    'Vermont': 'VT',
    'Massachusetts': 'MA',
    'Rhode Island': 'RI',
    'Connecticut': 'CT',
    'New York': 'NY',
    'New Jersey': 'NJ',
    'Pennsylvania': 'PA'
}
monthly_avg['STATE'] = monthly_avg['STATE'].map(state_name_mapping)

state_month_avg_precip= monthly_avg.groupby(['STATE', 'MO'])['PRECTOTCORR'].mean().reset_index()

loc_avg_precip= monthly_avg.groupby(['LAT', 'LON'])['PRECTOTCORR'].mean().reset_index()












