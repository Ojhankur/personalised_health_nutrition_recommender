import pandas as pd

# Load the existing dataset
df = pd.read_csv('nutrition_data.csv')

# Define the meal recommendations
recommendations = ['Chicken Salad', 'Vegan Wrap', 'Quinoa Bowl', 'Pasta Primavera']

# Create a list of meal recommendations that matches the length of the DataFrame
# Use the modulo operator to cycle through the recommendations if needed
df['Meal_Recommendation'] = [recommendations[i % len(recommendations)] for i in range(len(df))]

# Save the modified dataset
df.to_csv('nutrition_data.csv', index=False)
