#%% Synthetic Data Generation
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Number of data points
n_samples = 30000

# Helper functions
def get_season(date):
    month = date.month
    if month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    elif month in [9, 10, 11]:
        return 'autumn'
    else:
        return 'winter'

def generate_temperature(product_type):
    temp_ranges = {
        'produce': (32, 70),
        'dairy': (32, 40),
        'bakery': (0, 77),
        'meat': (0, 40)
    }
    return np.random.uniform(*temp_ranges[product_type])

def generate_humidity(product_type):
    humidity_ranges = {
        'bakery': (40, 60),
        'dairy': (0, 65),
        'produce': (80, 95),
        'meat': (65, 70)
    }
    return np.random.uniform(*humidity_ranges[product_type])

def generate_storage_duration(product_type):
    duration_ranges = {
        'dairy': (1, 30),
        'bakery': (1, 7),
        'meat': (1, 5),
        'produce': (1, 21)
    }
    return np.random.randint(*duration_ranges[product_type])

# Generate base dataset
data = {
    'Initial_Moisture_Content': np.random.uniform(0, 100, n_samples),
    'Initial_pH_Level': np.random.uniform(2, 9, n_samples),
    'Initial_Microbial_Load': np.random.uniform(0, 1e8, n_samples),
    'Product_Type': np.random.choice(['dairy', 'meat', 'produce', 'bakery'], n_samples),
    'Packaging_Type': np.random.choice(['plastic', 'glass', 'cardboard', 'aluminum'], n_samples),
    'Temperature_Fluctuations': np.random.randint(0, 21, n_samples),
    'Facility_ID': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
    'Batch_Number': np.random.randint(1000, 10000, n_samples)
}

# Generate dates over a 5-year period
start_date = datetime(2019, 1, 1)
end_date = datetime(2024, 1, 1)
date_range = (end_date - start_date).days
random_days = np.random.randint(0, date_range, n_samples)
dates = [start_date + timedelta(days=int(day)) for day in random_days]
data['Date'] = dates
data['Season'] = [get_season(date) for date in dates]

# Generate temperature, humidity, storage duration, and light exposure based on product type
data['Storage_Temperature'] = [generate_temperature(product_type) for product_type in data['Product_Type']]
data['Humidity'] = [generate_humidity(product_type) for product_type in data['Product_Type']]
data['Storage_Duration'] = [generate_storage_duration(product_type) for product_type in data['Product_Type']]
data['Light_Exposure'] = np.random.uniform(50, 200, n_samples)

# Convert to DataFrame
df = pd.DataFrame(data)

# Apply complexities

# Product-specific sensitivities
sensitivity = {'dairy': 0.75, 'meat': 0.85, 'produce': 0.65, 'bakery': 0.55}
df['Product_Sensitivity'] = df['Product_Type'].map(sensitivity)

# Packaging effectiveness
effectiveness = {'plastic': 0.7, 'glass': 0.9, 'cardboard': 0.5, 'aluminum': 0.8}
df['Packaging_Effectiveness'] = df['Packaging_Type'].map(effectiveness)

# Facility variations
facility_quality = {'A': 0.95, 'B': 0.85, 'C': 0.75, 'D': 0.65}
df['Facility_Quality'] = df['Facility_ID'].map(facility_quality)

# Temperature abuse
df.loc[np.random.choice(df.index, int(n_samples * 0.02)), 'Storage_Temperature'] += np.random.uniform(10, 20)


# Cross-contamination (rare events)
df.loc[np.random.choice(df.index, int(n_samples * 0.005)), 'Initial_Microbial_Load'] *= 10



# Batch-to-batch variations
df['Batch_Variation'] = np.random.normal(1, 0.1, n_samples)
df['Initial_Moisture_Content'] *= df['Batch_Variation']
df['Initial_pH_Level'] += np.random.normal(0, 0.5, n_samples)

# Calculate Quality Score
df['Quality_Score'] = (
    df['Product_Sensitivity'] * 
    df['Packaging_Effectiveness'] * 
    df['Facility_Quality'] * 
    (1 - df['Storage_Duration'] / 40) * 
    (1 - df['Temperature_Fluctuations'] / 40) * 
    (1 - abs(df['Storage_Temperature'] - 40) / 80) * 
    (1 - abs(df['Humidity'] - 60) / 100) * 
    (1 - df['Light_Exposure'] / 250) * 
    (1 - df['Initial_Microbial_Load'] / 2e8) * 
    np.exp(-df['Storage_Duration'] / 35)
    )

# Interaction effect between temperature and humidity
df['Quality_Score'] *= 1 - 0.08 * np.abs((df['Storage_Temperature'] - 50) * (df['Humidity'] - 60) / 1000)

# Normalize Quality Score
df['Quality_Score'] = (df['Quality_Score'] - df['Quality_Score'].min()) / (df['Quality_Score'].max() - df['Quality_Score'].min())

# Set target acceptability rate
target_acceptability = 0.86  # Aiming for 86% acceptability

# Function to calculate threshold for a given acceptability
def calculate_threshold(scores, target_acc):
    sorted_scores = np.sort(scores)
    index = int((1 - target_acc) * len(sorted_scores))
    return sorted_scores[index]

# Calculate overall threshold
overall_threshold = calculate_threshold(df['Quality_Score'], target_acceptability)

# Calculate facility-specific thresholds
facility_thresholds = {}
for facility in ['A', 'B', 'C', 'D']:
    facility_scores = df[df['Facility_ID'] == facility]['Quality_Score']
    facility_target = target_acceptability * (1 + (facility_quality[facility] - 0.75) / 0.75)
    facility_target = min(facility_target, 0.98)  # Cap the target at 98% to maintain some variation
    facility_thresholds[facility] = calculate_threshold(facility_scores, facility_target)

# Apply thresholds to determine quality
df['Quality_After_Storage'] = df.apply(lambda row: 
    1 if row['Quality_Score'] > facility_thresholds[row['Facility_ID']] else 0, axis=1)

# Add some randomness to account for unmeasured factors
random_factor = np.random.normal(0, 0.03, len(df))
df['Quality_After_Storage'] = df.apply(lambda row: 
    1 if (row['Quality_Score'] + random_factor[row.name]) > facility_thresholds[row['Facility_ID']] else 0, axis=1)

# Calculate final acceptability rates
overall_acceptability = df['Quality_After_Storage'].mean()
facility_acceptability = {facility: df[df['Facility_ID'] == facility]['Quality_After_Storage'].mean() 
                          for facility in ['A', 'B', 'C', 'D']}

print(f"Overall acceptability rate: {overall_acceptability:.2f}")
print("Facility acceptability rates:")
for facility, rate in facility_acceptability.items():
    print(f"Facility {facility}: {rate:.2f}")
print(f"Facility thresholds: {facility_thresholds}")

# Drop intermediate columns
df = df.drop(['Product_Sensitivity', 'Packaging_Effectiveness', 'Facility_Quality', 'Batch_Variation', 'Quality_Score'], axis=1)

# Save to CSV
df.to_csv('food_storage_quality_dataset.csv', index=False)

print(f"Dataset with {n_samples} samples has been generated and saved as 'food_storage_quality_dataset.csv'")
print(f"Overall acceptable items: {df['Quality_After_Storage'].sum()} ({df['Quality_After_Storage'].mean()*100:.2f}%)")
print(f"Overall unacceptable items: {len(df) - df['Quality_After_Storage'].sum()} ({(1-df['Quality_After_Storage'].mean())*100:.2f}%)")

# Print facility-specific statistics
for facility in ['A', 'B', 'C', 'D']:
    facility_data = df[df['Facility_ID'] == facility]
    acceptable_rate = facility_data['Quality_After_Storage'].mean() * 100
    print(f"Facility {facility} acceptable rate: {acceptable_rate:.2f}%")

# Print date range statistics
print(f"Date range: from {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"Number of unique dates: {df['Date'].nunique()}")