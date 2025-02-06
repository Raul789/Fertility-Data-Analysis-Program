import pandas as pd

# Load the text file into a DataFrame
data = pd.read_csv('fertility_data.txt', header=None)

# Set column names according to the dataset description
data.columns = [
    'Season', 'Age', 'Childish_Diseases', 'Accident_or_Trauma', 
    'Surgical_Intervention', 'High_Fever', 'Alcohol_Consumption', 
    'Smoking_Habit', 'Sitting_Hours', 'Diagnosis'
]

# Mapping values according to the provided information
# Season: -1 -> winter, -0.33 -> spring, 0.33 -> summer, 1 -> fall
season_mapping = {-1: 'winter', -0.33: 'spring', 0.33: 'summer', 1: 'fall'}
data['Season'] = data['Season'].map(season_mapping)

# Age: Already in (0, 1) scale, no need for mapping if we leave it as is.

# Childish Diseases, Accident or Trauma, Surgical Intervention: 0 -> yes, 1 -> no
binary_mapping = {1: 'no', 0: 'yes'}
data['Childish_Diseases'] = data['Childish_Diseases'].map(binary_mapping)
data['Accident_or_Trauma'] = data['Accident_or_Trauma'].map(binary_mapping)
data['Surgical_Intervention'] = data['Surgical_Intervention'].map(binary_mapping)

# High Fever: -1 -> less than three months ago, 0 -> more than three months ago, 1 -> no
high_fever_mapping = {-1: 'less_than_three_months', 0: 'more_than_three_months', 1: 'no'}
data['High_Fever'] = data['High_Fever'].map(high_fever_mapping)

# Alcohol Consumption: No direct mapping needed; assume the values are normalized between (0, 1).

# Smoking Habit: -1 -> never, 0 -> occasional, 1 -> daily
smoking_mapping = {-1: 'never', 0: 'occasional', 1: 'daily'}
data['Smoking_Habit'] = data['Smoking_Habit'].map(smoking_mapping)

# Sitting Hours: Already in (0, 1) scale, no need for mapping if we leave it as is.

# Diagnosis: N -> normal, O -> altered
diagnosis_mapping = {'N': 'normal', 'O': 'altered'}
data['Diagnosis'] = data['Diagnosis'].map(diagnosis_mapping)

# Display the processed data
print(data.head())
