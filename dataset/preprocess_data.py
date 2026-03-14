import pandas as pd

# Load dataset from CSV file
data = pd.read_csv("../data/gestures.csv", header=None)

# Print number of rows and columns
print("Dataset shape:", data.shape)

# Show first few rows of dataset
print("\nFirst 5 rows:")
print(data.head())

# Check if dataset contains missing values
missing = data.isnull().sum().sum()
print("\nMissing values:", missing)

# Save cleaned dataset
data.to_csv("../data/gestures_clean.csv", index=False, header=False)

print("\nClean dataset saved successfully")