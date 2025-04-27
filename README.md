# Lab 5.1: Demand Forecasting in Supply Chain Management

# 📚 Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ⚙️ Load the dataset
# (Replace 'your_data.csv' with your actual dataset)
df = pd.read_csv('your_data.csv')

# 👀 Explore the data
print(df.head())
print(df.info())
print(df.describe())

# 📊 Visualize the data
plt.figure(figsize=(10,6))
sns.histplot(df['demand'], kde=True)
plt.title('Demand Distribution')
plt.show()

# 🔧 Data Preprocessing
# Check for missing values
print(df.isnull().sum())

# Fill or drop missing values (Example: forward fill)
df.fillna(method='ffill', inplace=True)

# 🎯 Feature Selection
# (Assuming features are ['feature1', 'feature2', ...])
X = df[['feature1', 'feature2', 'feature3']]  # replace with actual feature names
y = df['demand']

# ✂️ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🚀 Build the Model
model = LinearRegression()
model.fit(X_train, y_train)

# 🔮 Make Predictions
y_pred = model.predict(X_test)

# 📈 Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# 📊 Plot Actual vs Predicted
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Demand')
plt.ylabel('Predicted Demand')
plt.title('Actual vs Predicted Demand')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.show()
