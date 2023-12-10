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
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Assuming df_geo is your DataFrame
# Let's encode the categorical variables using LabelEncoder
le_state = LabelEncoder()
le_generation = LabelEncoder()
le_fire_risk = LabelEncoder()

df_geo['state'] = le_state.fit_transform(df_geo['state'])
df_geo['generation'] = le_generation.fit_transform(df_geo['generation'])
df_geo['fire_risk'] = le_fire_risk.fit_transform(df_geo['fire_risk'])

# Extract features and target variable
X_categorical = df_geo[['state', 'generation', 'fire_risk']]
X_numeric = df_geo[['precipitation']]  # Add other numeric columns as needed

# Concatenate categorical and numeric features
X = pd.concat([X_categorical, X_numeric], axis=1)

y = df_geo['earthquake risk']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree model
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_report_str)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Assuming 'model' is your trained Decision Tree model
class_names_str = list(map(str, model.classes_))

plt.figure(figsize=(15, 10))
plot_tree(model, feature_names=list(X.columns), class_names=class_names_str, filled=True, rounded=True)
plt.show()
#%%
# Import necessary libraries for Naive Bayes
from sklearn.naive_bayes import GaussianNB


# Create a Naive Bayes model (Gaussian Naive Bayes for continuous features)
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy NB: {accuracy}")
print("Classification Report:\n", classification_report_str)


import seaborn as sns

# Concatenate features and target variable for training data
train_data = pd.concat([X_train, y_train], axis=1)

# Pair plot
sns.pairplot(train_data, hue='earthquake risk', palette='viridis', markers=["o", "s", "D"])
plt.show()

#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train.values)
y_train_tensor = torch.LongTensor(y_train.values)
X_test_tensor = torch.FloatTensor(X_test.values)
y_test_tensor = torch.LongTensor(y_test.values)

# Create a DataLoader for training and testing data
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Define the neural network
class EarthquakeRiskComplexNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(EarthquakeRiskComplexNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

# Example usage:
hidden_size1 = 64
hidden_size2 = 32
hidden_size3 = 16

# Instantiate the model
input_size = X_train.shape[1]
output_size = len(class_names_str)  # Number of classes
model = EarthquakeRiskComplexNN(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)

    print(f"Accuracy_NN: {accuracy}")

# Set the model back to training mode
model.train()

from sklearn.metrics import classification_report
import torch

# Assuming you have the true labels (y_test) and predicted labels (predicted)
# Convert tensors to numpy arrays
y_test_np = y_test_tensor.numpy()
predicted_np = predicted.numpy()

# Generate classification report
report = classification_report(y_test_np, predicted_np)

# Print the report
print(report)

#%%
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Assuming df_geo is your DataFrame
# Let's encode the categorical variables using LabelEncoder
le_state = LabelEncoder()
le_generation = LabelEncoder()
le_fire_risk = LabelEncoder()

df_geo['state'] = le_state.fit_transform(df_geo['state'])
df_geo['generation'] = le_generation.fit_transform(df_geo['generation'])
df_geo['fire_risk'] = le_fire_risk.fit_transform(df_geo['fire_risk'])

# Extract features and target variable
X_categorical = df_geo[['state', 'generation', 'earthquake risk']]
X_numeric = df_geo[['precipitation']]  # Add other numeric columns as needed

# Concatenate categorical and numeric features
X = pd.concat([X_categorical, X_numeric], axis=1)

y = df_geo['fire_risk']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree model
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_report_str)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Assuming 'model' is your trained Decision Tree model
class_names_str = list(map(str, model.classes_))

plt.figure(figsize=(15, 10))
plot_tree(model, feature_names=list(X.columns), class_names=class_names_str, filled=True, rounded=True)
plt.show()
#%%

# Import necessary libraries for Naive Bayes
from sklearn.naive_bayes import GaussianNB


# Create a Naive Bayes model (Gaussian Naive Bayes for continuous features)
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy NB: {accuracy}")
print("Classification Report:\n", classification_report_str)

import seaborn as sns

# Concatenate features and target variable for training data
train_data = pd.concat([X_train, y_train], axis=1)

# Pair plot
sns.pairplot(train_data, hue='earthquake risk', palette='viridis', markers=["o", "s", "D"])
plt.show()


#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train.values)
y_train_tensor = torch.LongTensor(y_train.values)
X_test_tensor = torch.FloatTensor(X_test.values)
y_test_tensor = torch.LongTensor(y_test.values)

# Create a DataLoader for training and testing data
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Define the neural network
class EarthquakeRiskComplexNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(EarthquakeRiskComplexNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

# Example usage:
hidden_size1 = 64
hidden_size2 = 32
hidden_size3 = 16

# Instantiate the model
input_size = X_train.shape[1]
output_size = len(class_names_str)  # Number of classes
model = EarthquakeRiskComplexNN(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)

    print(f"Accuracy_NN: {accuracy}")

# Set the model back to training mode
model.train()

from sklearn.metrics import classification_report
import torch

# Assuming you have the true labels (y_test) and predicted labels (predicted)
# Convert tensors to numpy arrays
y_test_np = y_test_tensor.numpy()
predicted_np = predicted.numpy()

# Generate classification report
report = classification_report(y_test_np, predicted_np)

# Print the report
print(report)


#%%
# Import necessary libraries
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import pandas as pd

# Assuming df_month is your DataFrame
# Make sure to encode categorical variables into numerical values
le_month = LabelEncoder()
le_state = LabelEncoder()
le_disaster = LabelEncoder()

df_month['month'] = le_month.fit_transform(df_month['month'])
df_month['state'] = le_state.fit_transform(df_month['state'])
df_month['disaster'] = le_disaster.fit_transform(df_month['disaster'])

# Define features (X) and target variable (y)
X = df_month[['month', 'state', 'disaster', 'precipitation']]
y = df_month['break_rate']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create a Decision Tree model
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_report_str)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Assuming 'model' is your trained Decision Tree model
class_names_str = list(map(str, model.classes_))

plt.figure(figsize=(15, 10))
plot_tree(model, feature_names=list(X.columns), class_names=class_names_str, filled=True, rounded=True)
plt.show()
#%%
# Import necessary libraries for Naive Bayes
from sklearn.naive_bayes import GaussianNB


# Create a Naive Bayes model (Gaussian Naive Bayes for continuous features)
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy NB: {accuracy}")
print("Classification Report:\n", classification_report_str)

import seaborn as sns

# Concatenate features and target variable for training data
train_data = pd.concat([X_train, y_train], axis=1)

# Pair plot
sns.pairplot(train_data, hue='break_rate', palette='viridis', markers=["o", "s", "D"])
plt.show()

#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train.values)
y_train_tensor = torch.LongTensor(y_train.values)
X_test_tensor = torch.FloatTensor(X_test.values)
y_test_tensor = torch.LongTensor(y_test.values)

# Create a DataLoader for training and testing data
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Define the neural network
class EarthquakeRiskComplexNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(EarthquakeRiskComplexNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

# Example usage:
hidden_size1 = 64
hidden_size2 = 32
hidden_size3 = 16

# Instantiate the model
input_size = X_train.shape[1]
output_size = len(class_names_str)  # Number of classes
model = EarthquakeRiskComplexNN(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)

    print(f"Accuracy_NN: {accuracy}")

# Set the model back to training mode
model.train()


from sklearn.metrics import classification_report
import torch

# Assuming you have the true labels (y_test) and predicted labels (predicted)
# Convert tensors to numpy arrays
y_test_np = y_test_tensor.numpy()
predicted_np = predicted.numpy()

# Generate classification report
report = classification_report(y_test_np, predicted_np)

# Print the report
print(report)
