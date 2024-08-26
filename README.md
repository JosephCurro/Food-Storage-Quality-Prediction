# Food Storage Quality Prediction

## Project Overview
This project simulates and analyzes food storage quality across multiple facilities. It generates a synthetic dataset incorporating various factors affecting food quality during storage, then uses machine learning to predict quality outcomes.

## Key Features
- Synthetic data generation with realistic parameters for food storage
- Incorporation of multiple factors: product type, storage conditions, facility variations
- Machine learning model to predict food quality after storage
- Comprehensive data analysis and visualization

## Tools and Technologies
- Python 3.12
- pandas, numpy for data manipulation
- scikit-learn for machine learning
- matplotlib, seaborn for data visualization

## Project Structure
1. Data Generation
   - Creation of synthetic dataset with various food storage parameters
   - Incorporation of real-world complexities (e.g., temperature fluctuations, facility variations)

2. Data Analysis
   - Exploratory Data Analysis (EDA) of the generated dataset
   - Visualization of key trends and relationships

3. Machine Learning
   - Preprocessing of data for machine learning
   - Training a Random Forest model to predict food quality
   - Model evaluation and feature importance analysis

## Key Findings
- Approximately 75% of the products kept an acceptable quality.
- Light exposure is the most crucial factor affecting food quality. Other important environmental factors were humidity and storage duration. Lower levels of each of these correspond to less product loss.
- Temperature and temperature fluctuations had a lower impact than other environmental factors, but had the same relationship to quality.
- Initial moisture content and pH levels appear to have little impact on acceptability, while initial microbial load does.
- Cardboard packaging has a larger impact on food quality than other methods of packaging.

## Model Performance
- Accuracy: 0.7860
- Precision: 0.8095 (acceptable)/ 0.5705 (unacceptable)
- Recall: 0.9454 (acceptable) / 0.2458 (unacceptable)
- F1-score: 0.8722 (acceptable) / 0.3436 (unacceptable)
- Macro averaged F1-score: 0.0679
- Weighted-averaged F1-score:0.7517


## Interpretation
- The model performs much better with recall for acceptable quality (94.54%) versus recall for unacceptable quality (24.57%).
- The results suggest the model needs a larger sample of unacceptable product to train on in order to more accurately judge quality.

## Future Work
- Address class imbalance by using over/undersampling.
- Review data generation to ensure complexities are in line with standard industry practices.
- Explore relationship between light exposure and food quality more in-depth.
- Experiment with different machine learning algorithms like gradient boosting.
- Develop a predictive system for optimal storage conditions.
- Collect real-world data to validate and refine the model's predictions.

## How to Run
1. Clone the repository
2. Install dependencies
3. Run the data generation script to create the synthetic dataset
4. Execute the analysis and modeling script to perform EDA and train the model

## Dependencies
- Python 3.12
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Contact
Joseph Curro
josephjcurro@gmail.com
