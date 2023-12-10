from geopy.geocoders import Nominatim
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

def generate_squares(min_lat, max_lat, min_lon, max_lon, step):
    squares = []
    for lat in range(int(min_lat * 100), int(max_lat * 100), int(step * 100)):
        for lon in range(int(min_lon * 100), int(max_lon * 100), int(step * 100)):
            square = Polygon([(lon / 100, lat / 100),
                              ((lon / 100)+step, lat / 100),
                              ((lon/ 100) + step, (lat/ 100)+ step),
                              (lon / 100, (lat / 100)+ step)])
            squares.append(square)
    return squares

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

# Define bounding box for specific Northeast USA states
min_latitude = 37.0
max_latitude = 47.0
min_longitude = -81.0
max_longitude = -66.8

# Specify the step size for latitude and longitude (0.1 degree)
step_size = 0.5

# Generate squares
squares = generate_squares(min_latitude, max_latitude, min_longitude, max_longitude, step_size)

# Create a GeoDataFrame from the squares
gdf = gpd.GeoDataFrame(geometry=squares)
# Print the GeoDataFrame
print(gdf)

# Add a column to designate the state
gdf['state'] = 'NE'  # Default value for all squares

# Extract state information using the get_us_state function
gdf['state'] = gdf.apply(lambda row: get_us_state(row.geometry.centroid.y, row.geometry.centroid.x), axis=1)

# Print the GeoDataFrame
print(gdf)


# Optionally, you can save the GeoDataFrame to a shapefile or other formats
# gdf.to_file("squares_northeast_usa_with_states.shp")
#%%

# Filter out unwanted states
valid_states = ['Maine', 'New Hampshire', 'Vermont', 'Massachusetts', 'Rhode Island', 'Connecticut', 'New York', 'New Jersey', 'Pennsylvania']
gdf = gdf[gdf['state'].isin(valid_states)]

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
gdf['state'] = gdf['state'].map(state_name_mapping)

#%%

def read_power_plants_csv(file_path):
    try:
        # Read the CSV file into a DataFrame, keeping only the specified columns
        columns_to_keep = ['STATE', 'TYPE', 'LATITUDE', 'LONGITUDE']
        df = pd.read_csv(file_path, usecols=columns_to_keep)

        # Display the DataFrame
        return (df)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Replace 'power_plants.csv' with the actual file path if it's in a different location
file_path = 'power_plants.csv'
df=read_power_plants_csv(file_path)

#%%
states_to_keep = ['NY', 'NJ', 'PA', 'CT', 'RI', 'MA', 'ME', 'NH', 'VT']

# Filter the DataFrame
powerplant_df = df[df['STATE'].isin(states_to_keep)]

# Print the resulting DataFrame
print(powerplant_df)
#%%
from collections import Counter
power_plants_gdf = gpd.GeoDataFrame(powerplant_df, geometry=gpd.points_from_xy(powerplant_df.LONGITUDE, powerplant_df.LATITUDE))
joined_data = gpd.sjoin(power_plants_gdf, gdf, op='within')

# Function to find majority TYPE for each polygon
def majority_type(types):
    counter = Counter(types)
    return counter.most_common(1)[0][0]

# Group by state and apply the majority_type function
majority_types = joined_data.groupby('index_right')['TYPE'].agg(majority_type).reset_index()

majority_types.index=majority_types.index_right
# # Merge the result back to gdf
gdf = gdf.merge(majority_types, left_index=True, right_index=True, how='left')

# # Rename the column to indicate the majority TYPE
gdf.rename(columns={'TYPE': 'Majority_TYPE'}, inplace=True)

# # Print or display the final GeoDataFrame
gdf.dropna()
gdf.drop('index_right', axis=1, inplace=True)
print(gdf)
gdf.to_csv('plants_inblocks.csv', index=False)


#%%

import os
import pandas as pd

# Path to the folder containing Excel files
folder_path = "disruption/"

# List to store individual DataFrames
dfs = []

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".xls") or filename.endswith(".xlsx"):
        file_path = os.path.join(folder_path, filename)
        # Read each Excel file into a DataFrame, skipping the first two rows
        df = pd.read_excel(file_path, skiprows=1)
        df = df[["Month", "Area Affected"]]
        # Append the DataFrame to the list
        dfs.append(df)

# Combine all DataFrames into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)
    

combined_df['Area Affected'] = combined_df['Area Affected'].str.split(':').str[0]



valid_states = ['Maine', 'New Hampshire', 'Vermont', 'Massachusetts', 'Rhode Island', 'Connecticut', 'New York', 'New Jersey', 'Pennsylvania']
combined_df = combined_df[combined_df['Area Affected'].isin(valid_states)]

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
combined_df['Area Affected'] = combined_df['Area Affected'].map(state_name_mapping)

# Display the combined DataFrame
print(combined_df)
breakdown_df=pd.DataFrame()
breakdown_df['Month']=combined_df['Month']
breakdown_df['State']=combined_df['Area Affected']
breakdown_df.to_csv('breakdowns.csv', index=False)

#%%
from shapely.geometry import Point

# Assuming your CSV file is in the same directory as your script
file_path = "earthquakes.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Create Shapely Point geometries from earthquake_data coordinates
earthquake_points = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]

# Create a GeoDataFrame from the earthquake_points
earthquake_gdf = gpd.GeoDataFrame(geometry=earthquake_points)

# Spatial join to count earthquakes within each polygon
gdf_with_earthquake_count = gpd.sjoin(gdf, earthquake_gdf, how='left', op='contains')

gdf_with_earthquake_count['index_right'].fillna(0, inplace=True)
gdf_with_earthquake_count['index_right'] = gdf_with_earthquake_count['index_right'].apply(lambda x: 1 if x != 0 else 0)
gdf_with_earthquake_count = gdf_with_earthquake_count.rename(columns={'index_right': 'earthquake_count'})

# # Group by polygon and count the number of earthquakes
# earthquake_count_per_polygon = gdf_with_earthquake_count.groupby('index_right').size()
# eqcount=pd.DataFrame()
# eqcount['count']=earthquake_count_per_polygon

# # Merge dataframes based on index
# merged_df = gdf.merge(eqcount, left_index=True, right_index=True, how='left')

# # Fill NaN values in 'count' column with 0
# merged_df['count'].fillna(0, inplace=True)

# # Rename the 'count' column to 'eq_count'
# merged_df.rename(columns={'count': 'eq_count'}, inplace=True)

# # Display the resulting dataframe
# print(merged_df)

# # Merge the earthquake count back to the original GeoDataFrame
# gdf = gdf.merge(eqcount, how='left', left_index=True, right_index=True)

# # Rename the column to something meaningful
# gdf = gdf.rename(columns={'count': 'earthquake_count'})

# # Fill NaN values with 0 (indicating no earthquakes)
# gdf['earthquake_count'] = gdf['earthquake_count'].fillna(0).astype(int)

# Print or use the GeoDataFrame with the added column
print(gdf_with_earthquake_count)
gdf_with_earthquake_count.to_csv('plants_inblocks_eqcount.csv', index=False)
#%%
# Function to read DataFrame and create Shapely Points
def create_shapely_points_from_dataframe(dataframe):
    points = []

    # Assuming 'X' and 'Y' are column names in the DataFrame
    x_column = 'X'
    y_column = 'Y'

    for index, row in dataframe.iterrows():
        x = row[x_column]
        y = row[y_column]
        point = Point(x, y)
        points.append(point)

    return points

# Example usage
file_path = 'InFORM_Fire_Occurrence_Data_Records.csv'  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

shapely_points = create_shapely_points_from_dataframe(df)
#%%
# Now, shapely_points contains a list of Shapely Point objects created from the 'X' and 'Y' columns in the DataFrame

# Create a GeoDataFrame from the list of Shapely Points
points_gdf = gpd.GeoDataFrame(geometry=shapely_points, crs=gdf.crs)

# Perform a spatial join to count points within each polygon
spatial_join = gpd.sjoin(points_gdf, gdf_with_earthquake_count, op='within', how='left')

# Group by polygon and count the number of points within each polygon
points_count = spatial_join.groupby('index_right').size().reset_index(name='fire_count')
points_count.index=points_count['index_right']
points_count.drop('index_right', axis=1, inplace=True)
# Merge the count back to the original GeoDataFrame
gdf_eq_fire = gdf_with_earthquake_count.merge(points_count, how='left', left_index=True, right_on='index_right')

# Fill NaN values with 0 (for polygons without points)
gdf_eq_fire['fire_count'] = gdf_eq_fire['fire_count'].fillna(0).astype(int)

# Print or use gdf_with_count as needed
print(gdf_eq_fire)
gdf_eq_fire.to_csv('plants_inblocks_eq_fire.csv', index=False)
#%%

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

result = gpd.sjoin(geo_df, gdf, how='left', op='within')

result = result.drop(columns=['Majority_TYPE'])

state_month_avg_precip= result.groupby(['state', 'MO'])['PRECTOTCORR'].mean().reset_index()

#%%

import pandas as pd
from shapely.geometry import Point


# Example usage
file_path = 'FemaWebDisasterDeclarations.csv'  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

# Now, shapely_points contains a list of Shapely Point objects created from the 'X' and 'Y' columns in the DataFrame
df['declarationDate'] = pd.to_datetime(df['declarationDate'])

# Add a new column 'month' based on the 'declarationDate'
df['month'] = df['declarationDate'].dt.month

# If you want the month name instead of the month number, you can use the following:
# df['month'] = df['declarationDate'].dt.strftime('%B')

# Display the DataFrame with the new 'month' column
print(df)
# Assuming df is your DataFrame
columns_to_keep = ['month', 'stateCode', 'incidentType']

# Select only the specified columns
df_subset = df[columns_to_keep]

# Display the DataFrame with the selected columns
print(df_subset)



import os
import pandas as pd

# Path to the folder containing Excel files
folder_path = "disruption/"

# List to store individual DataFrames
dfs = []

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".xls") or filename.endswith(".xlsx"):
        file_path = os.path.join(folder_path, filename)
        # Read each Excel file into a DataFrame, skipping the first two rows
        df = pd.read_excel(file_path, skiprows=1)
        df = df[["Month", "Area Affected"]]
        # Append the DataFrame to the list
        dfs.append(df)

# Combine all DataFrames into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)
    

combined_df['Area Affected'] = combined_df['Area Affected'].str.split(':').str[0]



valid_states = ['Maine', 'New Hampshire', 'Vermont', 'Massachusetts', 'Rhode Island', 'Connecticut', 'New York', 'New Jersey', 'Pennsylvania']
combined_df = combined_df[combined_df['Area Affected'].isin(valid_states)]

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
combined_df['Area Affected'] = combined_df['Area Affected'].map(state_name_mapping)

# Display the combined DataFrame
print(combined_df)
breakdown_df=pd.DataFrame()
breakdown_df['Month']=combined_df['Month']
breakdown_df['State']=combined_df['Area Affected']
breakdown_df.to_csv('breakdowns.csv', index=False)

valid_states = ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA']

df_subset= df_subset[df_subset['stateCode'].isin(valid_states)]

frequency_df = breakdown_df.groupby(['Month', 'State']).size().reset_index(name='Frequency')

month_mapping = {
    'January': 1,
    'February': 2,
    'March': 3,
    'April': 4,
    'May': 5,
    'June': 6,
    'July': 7,
    'August': 8,
    'September': 9,
    'October': 10,
    'November': 11,
    'December': 12
}

# Use the map function to replace month names with month numbers
frequency_df['Month'] = frequency_df['Month'].map(month_mapping)

# Print the updated DataFrame
print(frequency_df)

merged_df = pd.merge(df_subset, frequency_df, left_on=['month', 'stateCode'], right_on=['Month', 'State'], how='left')
merged_df
merged_df = merged_df.drop(['Month', 'State'], axis=1)
merged_df['break_Frequency'] = merged_df['Frequency'].fillna(0).astype(int)
merged_df = merged_df.drop(['Frequency'], axis=1)
merged_df.to_csv('breaks_disasters_states.csv', index=False)



merged_df = pd.merge(merged_df, state_month_avg_precip, left_on=['month', 'stateCode'], right_on=['MO', 'state'], how='left')
merged_df
merged_df = merged_df.drop(['MO', 'state'], axis=1)
merged_df.to_csv('breaks_disasters_states_precipitation.csv', index=False)
#%%
final_df_month=merged_df

final_df_month = final_df_month.rename(columns={'PRECTOTCORR': 'precipitation', 
                                  'stateCode': 'state', 
                                  'incidentType': 'disaster', 
                                  'break_Frequency': 'break_rate'})

final_df_month['precipitation'] = pd.qcut(final_df_month['precipitation'], q=[0, 0.25, 0.5, 0.75, 1], labels=False)
final_df_month['break_rate'] = pd.qcut(final_df_month['break_rate'], q=[0, 0.25, 0.5, 0.75, 1], labels=False,duplicates='drop')
final_df_month.to_csv('state_precip_disas_break_.csv', index=False)


#%%
geo_df = geo_df.groupby(['geometry'])['PRECTOTCORR'].mean().reset_index()
geometry = [Point(xy) for xy in zip(geo_df.geometry.apply(lambda geom: geom.x), geo_df.geometry.apply(lambda geom: geom.y))]
geo_df = gpd.GeoDataFrame(geo_df, geometry=geometry)


result = gpd.sjoin(geo_df, gdf_eq_fire, how='right', op='within')
result=result.dropna()


# Assuming 'results' is your DataFrame
result = result.drop(columns=['index_left'])
result = result.rename(columns={'PRECTOTCORR': 'precipitation', 
                                  'Majority_TYPE': 'generation', 
                                  'earthquake_count': 'earthquake risk', 
                                  'fire_count': 'fire_risk'})


#%%

final_df_geo=result
# Assuming 'results' is your DataFrame
final_df_geo['fire_risk'] = pd.qcut(final_df_geo['fire_risk'], q=[0, 0.25, 0.5, 0.75, 1], labels=False)
final_df_geo['precipitation'] = pd.qcut(final_df_geo['precipitation'], q=[0, 0.25, 0.5, 0.75, 1], labels=False)
final_df_geo.to_csv('geo_fire_precip_gen_eq.csv', index=False)

# If you want to assign labels to the bins, you can do the following:
# results['fire_risk'] = pd.cut(results['fire_risk'], bins=4, labels=['Low', 'Medium', 'High', 'Very High'])
# results['precipitation'] = pd.qcut(results['precipitation'], q=[0, 0.25, 0.5, 0.75, 1], labels=['Q1', 'Q2', 'Q3', 'Q4'])












