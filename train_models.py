import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('nutrition_data.csv')

# Print the DataFrame columns
print("Columns in the dataset:", df.columns.tolist())

# Check if the required columns exist
if 'Age' not in df.columns or 'Activity_Level' not in df.columns or 'Calories_Intake' not in df.columns:
    raise ValueError("Required columns are missing from the dataset.")

# Assume you have the following columns in your dataset
X = df[['Age', 'Activity_Level']]  # Features
y = df['Calories_Intake']           # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'LinearRegression': LinearRegression(),
    'RandomForestRegressor': RandomForestRegressor(),
    'GradientBoostingRegressor': GradientBoostingRegressor()
}

# Train models and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"{name} Mean Squared Error: {mse}")
