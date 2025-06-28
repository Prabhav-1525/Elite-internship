import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Function to load data from a CSV file
def load_data(file_path):
    return pd.read_csv(r'C:\Users\prabh\OneDrive\Desktop\Elite internship\Task 1\input.csv')

# Function to preprocess and transform data
def preprocess_transform_data(data):
    # Example: Drop duplicates
    data = data.drop_duplicates()
    # Example: Fill missing values with mean
    data.fillna(data.mean(), inplace=True)
    # Example: Select features and target
    X = data.drop('target', axis=1)
    y = data['target']
    return X, y

# Function to create a pipeline for scaling features
def create_pipeline():
    return Pipeline([('scaler', StandardScaler())])

# Function to split data into train and test sets
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Function to save processed data to CSV
def load_to_csv(data, file_path):
    data.to_csv(file_path, index=False)

# Main ETL function
def etl_process(input_file, output_train_file, output_test_file):
    # Load data
    data = load_data(input_file)
    # Preprocess and transform
    X, y = preprocess_transform_data(data)
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    # Create pipeline
    pipeline = create_pipeline()
    # Fit and transform training data
    X_train_scaled = pipeline.fit_transform(X_train)
    # Transform test data
    X_test_scaled = pipeline.transform(X_test)
    # Save processed data
    train_processed = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_processed['target'] = y_train.reset_index(drop=True)
    load_to_csv(train_processed, output_train_file)
    test_processed = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_processed['target'] = y_test.reset_index(drop=True)
    load_to_csv(test_processed, output_test_file)

# Example usage
# Example usage
etl_process('input.csv', 'train_processed.csv', 'test_processed.csv')