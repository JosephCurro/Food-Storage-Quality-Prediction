#%% Analysis and Modeling

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv('food_storage_quality_dataset.csv')

# 2. Exploratory Data Analysis
print(df.describe())
print(df.info())

# Visualize the distribution of the target variable
plt.figure(figsize=(8, 6))
df['Quality_After_Storage'].value_counts().plot(kind='bar')
plt.title('Distribution of Quality After Storage')
plt.xlabel('Quality (0: Unacceptable, 1: Acceptable)')
plt.ylabel('Count')
plt.show()

# Visualize the relationship between numerical features and quality
numerical_features = ['Initial_Moisture_Content', 'Initial_pH_Level', 'Initial_Microbial_Load', 
                      'Temperature_Fluctuations', 'Storage_Temperature', 'Humidity', 
                      'Storage_Duration', 'Light_Exposure']

fig, axes = plt.subplots(4, 2, figsize=(20, 30))
axes = axes.ravel()

for idx, col in enumerate(numerical_features):
    sns.boxplot(x='Quality_After_Storage', y=col, data=df, ax=axes[idx])
    axes[idx].set_title(f'{col} vs Quality')

plt.tight_layout()
plt.show()

# 3. Data Preprocessing
# Separate features and target
X = df.drop('Quality_After_Storage', axis=1)
y = df['Quality_After_Storage']

# Identify numerical and categorical columns
numerical_features = ['Initial_Moisture_Content', 'Initial_pH_Level', 'Initial_Microbial_Load', 
                      'Temperature_Fluctuations', 'Storage_Temperature', 'Humidity', 
                      'Storage_Duration', 'Light_Exposure']
categorical_features = ['Product_Type', 'Packaging_Type', 'Facility_ID', 'Season']

# Create preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model Training
# Create a pipeline that preprocesses the data and then trains a Random Forest
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Fit the model
model.fit(X_train, y_train)

# 6. Model Evaluation
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)

# Precision, Recall, and F1-score for both classes
precision_acceptable = precision_score(y_test, y_pred, pos_label=1)
precision_unacceptable = precision_score(y_test, y_pred, pos_label=0)

recall_acceptable = recall_score(y_test, y_pred, pos_label=1)
recall_unacceptable = recall_score(y_test, y_pred, pos_label=0)

f1_acceptable = f1_score(y_test, y_pred, pos_label=1)
f1_unacceptable = f1_score(y_test, y_pred, pos_label=0)
f1_macro = f1_score(y_test, y_pred, average='macro') # macro averaged F1 score
f1_weighted = f1_score(y_test, y_pred, average='weighted') # weighted average F1 score

print(f"Accuracy: {accuracy:.4f}")
print("\nAcceptable Quality (Class 1):")
print(f"Precision: {precision_acceptable:.4f}")
print(f"Recall: {recall_acceptable:.4f}")
print(f"F1-score: {f1_acceptable:.4f}")
print("\nUnacceptable Quality (Class 0):")
print(f"Precision: {precision_unacceptable:.4f}")
print(f"Recall: {recall_unacceptable:.4f}")
print(f"F1-score: {f1_unacceptable:.4f}")
print(f"Macro-averaged F1 Score: {f1_macro:.4f}")
print(f"Weighted-averaged F1 Score: {f1_weighted:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 7. Feature Importance
feature_names = (numerical_features + 
                 [f"{feature}_{category}" for feature, categories in 
                  zip(categorical_features, model.named_steps['preprocessor']
                      .named_transformers_['cat'].categories_) 
                  for category in categories[1:]])

importances = model.named_steps['classifier'].feature_importances_
feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()