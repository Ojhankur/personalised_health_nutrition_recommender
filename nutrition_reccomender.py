class NutritionRecommender:
    def __init__(self, age, weight, dietary_preferences):
        self.age = age
        self.weight = weight
        self.dietary_preferences = dietary_preferences.lower()  # Convert to lowercase for consistency

    def recommend_calories(self):
        # Basic calorie recommendation based on weight
        if self.weight < 60:
            base_calories = 1800
        elif 60 <= self.weight <= 80:
            base_calories = 2200
        else:
            base_calories = 2600

        # Adjust calories based on age
        if self.age < 30:
            age_adjustment = 200
        elif 30 <= self.age <= 50:
            age_adjustment = 0
        else:
            age_adjustment = -200
        
        # Adjust calories based on dietary preference
        if self.dietary_preferences == 'vegetarian':
            diet_adjustment = 0
        elif self.dietary_preferences == 'non-vegetarian':
            diet_adjustment = 100  # Slightly more calories for non-vegetarians
        elif self.dietary_preferences == 'vegan':
            diet_adjustment = -100  # Slightly fewer calories for vegans
        else:
            return None  # Return None for unrecognized dietary preferences

        # Calculate final recommended calories
        recommended_calories = base_calories + age_adjustment + diet_adjustment
        return recommended_calories

# Function to get user input
def get_user_data():
    print("Welcome to the Nutrition Recommender!")
    
    # Get user input for age, weight, and dietary preference
    age = int(input("Please enter your age: "))
    weight = float(input("Please enter your weight (in kg): "))
    
    # Display available dietary preferences
    print("Dietary preferences available: Vegetarian, Non-Vegetarian, Vegan")
    dietary_preferences = input("Please enter your dietary preference: ").strip().lower()  # Convert to lowercase
    
    return age, weight, dietary_preferences

# Main program
if __name__ == "__main__":
    # Get user data
    age, weight, dietary_preferences = get_user_data()

    # Initialize the Nutrition Recommender with user data
    user = NutritionRecommender(age=age, weight=weight, dietary_preferences=dietary_preferences)

    # Get recommended calories
    recommended_calories = user.recommend_calories()

    # Display the user's input values and recommended calories per day
    print("\nUser Input:")
    print(f"Age: {age} years")
    print(f"Weight: {weight} kg")
    print(f"Dietary Preference: {dietary_preferences.capitalize()}")  # Capitalize for display

    if recommended_calories is not None:
        print(f"\nBased on your input, your recommended daily calorie intake is: {recommended_calories} kcal per day.")
    else:
        print("\nDietary preference not recognized. Please enter 'Vegetarian', 'Non-Vegetarian', or 'Vegan'.")
