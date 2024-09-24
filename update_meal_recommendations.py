import pandas as pd

# Load the existing dataset
df = pd.read_csv('nutrition_data.csv')

# Define new meal recommendations
new_recommendations = ['Grilled Salmon', 'Vegetable Stir Fry', 'Turkey Wrap', 'Fruit Salad']

# Create a list of meal recommendations that matches the length of the DataFrame
df['Meal_Recommendation'] = [new_recommendations[i % len(new_recommendations)] for i in range(len(df))]

# Save the modified dataset
df.to_csv('nutrition_data.csv', index=False)
