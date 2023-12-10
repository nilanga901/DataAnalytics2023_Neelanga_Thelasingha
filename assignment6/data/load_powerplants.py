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


#%%

import geopandas as gpd
import matplotlib.pyplot as plt

# Create a GeoDataFrame (replace this with your own data)
data = {'City': ['New York', 'Los Angeles', 'Chicago'],
        'Latitude': [40.7128, 34.0522, 41.8781],
        'Longitude': [-74.0060, -118.2437, -87.6298]}

geometry = gpd.points_from_xy(data['Longitude'], data['Latitude'])
gdf = gpd.GeoDataFrame(data, geometry=geometry)

# Plot the GeoDataFrame on a map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
ax = world.plot(figsize=(10, 6), color='lightgray')

# Plot your GeoDataFrame on top of the world map
gdf.plot(ax=ax, color='red', marker='o', markersize=50)

# Show the map
plt.show()



