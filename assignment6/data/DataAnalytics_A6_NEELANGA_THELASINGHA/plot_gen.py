import matplotlib.pyplot as plt

# Assuming final_df_geo is your DataFrame and 'generation' is the column you want to plot
final_df_month['month'].hist()

# Tilt the x-axis labels
plt.xticks(rotation=45, ha='right')  # You can adjust the rotation angle as needed

# Add labels and title if necessary
plt.xlabel('Disaster months')
plt.ylabel('Frequency')
plt.title('Histogram of Disaster months')

# Show the plot
plt.show()
#%%



# Convert categorical columns to numeric representations
final_df_month['month_code'] = pd.factorize(final_df_month['month'])[0]
final_df_month['state_code'] = pd.factorize(final_df_month['state'])[0]

# Create a 3D histogram
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

hist, xedges, yedges = np.histogram2d(final_df_month['month_code'], final_df_month['state_code'], bins=(len(final_df_month['month'].unique()), len(final_df_month['state'].unique())))
xpos, ypos = np.meshgrid(xedges[:-1] + 0.5, yedges[:-1] + 0.5, indexing="ij")

xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

dx = dy = 1
dz = hist.ravel()

# Create a 3D bar plot
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, cmap='Blues')

# Add labels and title
ax.set_xlabel('Month')
ax.set_ylabel('State')
ax.set_zlabel('Frequency')
ax.set_title('Histogram of Disasters across Month and State')

# Display the plot
plt.show()