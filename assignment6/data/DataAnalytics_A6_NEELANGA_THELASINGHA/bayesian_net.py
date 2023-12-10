import pandas as pd
import bnlearn as bn

# Assuming the CSV file is in the same directory as your script or Jupyter notebook
file_path = 'geo_fire_precip_gen_eq.csv'

# Read the CSV file into a pandas DataFrame
df_geo= pd.read_csv(file_path)

# Assuming the CSV file is in the same directory as your script or Jupyter notebook
file_path = 'state_precip_disas_break_.csv'

# Read the CSV file into a pandas DataFrame
df_month= pd.read_csv(file_path)


#%%

# Convert to onehot
dfhot, dfnum = bn.df2onehot(df_geo)


# Define the network structure
edges = [('geometry', 'state'),('state', 'generation'),('geometry', 'earthquake risk'),('geometry', 'fire_risk'),('generation', 'earthquake risk'),('generation', 'fire_risk'),('precipitation', 'fire_risk'),('precipitation', 'earthquake risk'),('geometry', 'precipitation')]

# Make the actual Bayesian DAG
DAG = bn.make_DAG(edges)

# Structure learning
# model = bn.structure_learning.fit(dfnum, methodtype='cl', black_list=['Embarked','Parch','Name'], root_node='Survived', bw_list_method='nodes')
# model = bn.structure_learning.fit(dfnum)
# Plot
G = bn.plot(DAG, interactive=False)

# # Compute edge strength with the chi_square test statistic
# model = bn.independence_test(DAG, dfnum, test='chi_square', prune=True)
# # Plot
# bn.plot(model, interactive=False, pos=G['pos'])

# Parameter learning
model = bn.parameter_learning.fit(DAG, dfnum)

#%%

# Convert to onehot
dfhot, dfnum = bn.df2onehot(df_month)


# Define the network structure
edges = [('month', 'disaster'),('state', 'disaster'),('month', 'precipitation'),('state', 'precipitation'),('precipitation', 'break_rate'),('disaster', 'break_rate')]

# Make the actual Bayesian DAG
DAG = bn.make_DAG(edges)

# Structure learning
# model = bn.structure_learning.fit(dfnum, methodtype='cl', black_list=['Embarked','Parch','Name'], root_node='Survived', bw_list_method='nodes')
# model = bn.structure_learning.fit(dfnum)
# Plot
G = bn.plot(DAG, interactive=False)

# # Compute edge strength with the chi_square test statistic
# model = bn.independence_test(DAG, dfnum, test='chi_square', prune=True)
# # Plot
# bn.plot(model, interactive=False, pos=G['pos'])

# Parameter learning
model = bn.parameter_learning.fit(DAG, dfnum)
