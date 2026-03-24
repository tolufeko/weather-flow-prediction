import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # weights, velocity and biases with small random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        output = sigmoid(self.z2)
        return output

    def backward(self, X, y, output, learning_rate, momentum, weight_decay):
        error = y - output
        d_output = error * sigmoid_derivative(output)

        d_hidden = d_output.dot(self.W2.T) * sigmoid_derivative(self.a1)

        # Momentum updates with L2 regularization (weight decay)
        self.v_W2 = momentum * self.v_W2 + learning_rate * (self.a1.T.dot(d_output) - weight_decay * self.W2)
        self.W2 += self.v_W2

        self.v_b2 = momentum * self.v_b2 + learning_rate * np.sum(d_output, axis=0, keepdims=True)
        self.b2 += self.v_b2

        self.v_W1 = momentum * self.v_W1 + learning_rate * (X.T.dot(d_hidden) - weight_decay * self.W1)
        self.W1 += self.v_W1

        self.v_b1 = momentum * self.v_b1 + learning_rate * np.sum(d_hidden, axis=0, keepdims=True)
        self.b1 += self.v_b1

    def train(self, X, y, epochs, learning_rate, momentum, weight_decay):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate, momentum, weight_decay)

            loss = np.mean((y - output) ** 2)  # MSE loss

            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    def predict(self, X):
        return self.forward(X)
    
file_path = "Ouse93-96 - Student.xlsx"
flow_df = pd.read_excel(file_path, sheet_name='Mean Daily Flow') 
rainfall_df = pd.read_excel(file_path, sheet_name='Daily Rainfall Total') 

flow_df['Crakehill Previous Day Flow'] = flow_df['Crakehill fl'].shift(1)
flow_df['Skip Bridge Previous Day Flow'] = flow_df['Skip Bridge fl'].shift(1)
flow_df['Westwick Previous Day Flow'] = flow_df['Westwick fl'].shift(1)
flow_df['Skelton Previous Day Flow'] = flow_df['Skelton fl'].shift(1)
flow_df = flow_df.drop(index=0)

rainfall_df['Arkengarthdale Previous Day Rainfall'] = rainfall_df['Arkengarthdale rf'].shift(1)
rainfall_df['Malham Tarn Previous Day Rainfall'] = rainfall_df['Malham Tarn rf'].shift(1)
rainfall_df['Snaizeholme Previous Day Rainfall'] = rainfall_df['Snaizeholme rf'].shift(1)
rainfall_df = rainfall_df.drop(index=0)

rainfall_df['Month'] = rainfall_df['Date'].dt.month

def main():
    file_path = "Ouse93-96 - Student.xlsx"
    flow_df = pd.read_excel(file_path, sheet_name='Mean Daily Flow') 
    rainfall_df = pd.read_excel(file_path, sheet_name='Daily Rainfall Total') 

    flow_df['Crakehill Previous Day Flow'] = flow_df['Crakehill fl'].shift(1)
    flow_df['Skip Bridge Previous Day Flow'] = flow_df['Skip Bridge fl'].shift(1)
    flow_df['Westwick Previous Day Flow'] = flow_df['Westwick fl'].shift(1)
    flow_df['Skelton Previous Day Flow'] = flow_df['Skelton fl'].shift(1)
    flow_df = flow_df.drop(index=0)

    rainfall_df['Arkengarthdale Previous Day Rainfall'] = rainfall_df['Arkengarthdale rf'].shift(1)
    rainfall_df['Malham Tarn Previous Day Rainfall'] = rainfall_df['Malham Tarn rf'].shift(1)
    rainfall_df['Snaizeholme Previous Day Rainfall'] = rainfall_df['Snaizeholme rf'].shift(1)
    rainfall_df['East Cowton Previous Day Rainfall'] = rainfall_df['East Cowton rf'].shift(1)
    rainfall_df = rainfall_df.drop(index=0)

    rainfall_df['Month'] = rainfall_df['Date'].dt.month

    data = pd.merge(rainfall_df, flow_df, on='Date')
    data = clean_non_numeric_values(data)
    data = remove_outliers(data)

    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)

    data = data.drop(columns=['Arkengarthdale rf', 'East Cowton rf', 'Malham Tarn rf', 'Snaizeholme rf', 'Crakehill fl', 'Skip Bridge fl', 'Westwick fl', 'Month'])
    
    data = standardise_data(data)

    # Separate input (X) and output (y)
    X = data.drop(columns=['Date', 'Skelton fl']).values
    y = data['Skelton fl'].values.reshape(-2, 1)

    # Split the data into training, validation and testing
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_val_split(X, y)

    # Create and train the MLP
    input_size = X_train.shape[1]
    hidden_size = 6
    output_size = 1
    epochs = 5000

    mlp = MLP(input_size, hidden_size, output_size)
    mlp.train(X_train, y_train, epochs, learning_rate=0.005, momentum=0.6, weight_decay=0.01)

    # Evaluate on validation set
    val_preds = mlp.predict(X_val)
    val_loss = np.mean((y_val - val_preds) ** 2)
    print(f"MLP Validation Loss: {val_loss:.4f}")

    # Evaluate on test set
    test_preds = mlp.predict(X_test)
    test_loss = np.mean((y_test - test_preds) ** 2)
    print(f"MLP Test Loss: {test_loss:.6f}")

    # Create a DataFrame for the predictions, with columns for the features in X_test
    columns = data.drop(columns=['Date', 'Skelton fl']).columns.tolist()
    test_preds_df = pd.DataFrame(X_test, columns = columns)
    test_preds_df['Predicted MLP Skelton fl'] = test_preds 

    # Add the actual 'Skelton fl' from the data for comparison
    test_preds_df['Actual Skelton fl'] = y_test
    
    test_indices = np.arange(len(data))[-len(X_test):]  # Get indices for X_test from the original data
    test_preds_df['Date'] = data.iloc[test_indices]['Date'].values

    baseline_model = LinearRegression()
    baseline_model.fit(X_train, y_train)

    # Make predictions using the trained model
    y_pred_test_baseline = baseline_model.predict(X_test)
    y_pred_val_baseline = baseline_model.predict(X_val)

    # Calculate MSE baseline model
    mse_test_baseline = mean_squared_error(y_test, y_pred_test_baseline)
    mse_val_baseline = mean_squared_error(y_val, y_pred_val_baseline)

    # Print the performance of the baseline model
    print(f"Baseline Model Test Loss: {mse_test_baseline:.6f}")
    print(f"Baseline Model Validation Loss: {mse_val_baseline:.6f}")

    test_preds_df['Predicted Baseline Skelton fl'] = y_pred_test_baseline

    with pd.ExcelWriter(file_path, mode='a', if_sheet_exists='replace') as writer:
        data.to_excel(writer, sheet_name='merged_data_standarised', index=False)
        test_preds_df.to_excel(writer, sheet_name='test_predictions', index=False)

    create_graphs()

def train_test_val_split(X, y, test_size=0.2, val_size=0.5):
    indices = np.arange(len(X))
    
    # 60% for training, 40% for temporary (validation + test)
    temp_size = int(len(X) * (1 - test_size))  # 60% for training
    X_train, X_temp = X[indices[:temp_size]], X[indices[temp_size:]]
    y_train, y_temp = y[indices[:temp_size]], y[indices[temp_size:]]
    
    # Now split the temporary set (40% of the original data) into validation (50%) and test (50%)
    val_size = int(len(X_temp) * val_size)  # 50% of the 40% => 20% of the original dataset
    X_val, X_test = X_temp[:val_size], X_temp[val_size:]
    y_val, y_test = y_temp[:val_size], y_temp[val_size:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def standardise_data(data, feature_range=(0.1, 0.9)):
    data_without_date = data.drop(columns=['Date'], errors='ignore')
    
    min_vals = np.min(data_without_date, axis=0)
    max_vals = np.max(data_without_date, axis=0)
    
    scale = feature_range[1] - feature_range[0]
    min_range = feature_range[0]

    scaled_data = min_range + ((data_without_date - min_vals) / (max_vals - min_vals)) * scale

    scaled_data['Date'] = data['Date'] 

    return scaled_data

def clean_non_numeric_values(df):
    df_cleaned = df.copy()
    non_numeric_rows_info = []

    for col in df_cleaned.columns:
        if col != 'Date':
            # Find non-numeric values in the column
            non_numeric_mask = pd.to_numeric(df_cleaned[col], errors='coerce').isna()
            non_numeric_rows = df_cleaned[non_numeric_mask]

            # Record info about each non-numeric value
            for _, row in non_numeric_rows.iterrows():
                non_numeric_rows_info.append({
                    'Date': row['Date'],
                    'Column': col,
                    'Non Numeric Value': row[col]
                })

            # Convert column to numeric (invalid values become NaN)
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

    # Ensure date is parsed correctly
    df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'], errors='coerce').dt.date

    # Drop any rows with NaNs (either from conversion or bad dates)
    df_cleaned = df_cleaned.dropna()

    return df_cleaned

def remove_outliers(df):
    df_cleaned = df.copy()
    outlier_info = []  # To collect outlier details
    bounds = {}

    # Identify and remove outliers
    for col in df_cleaned.columns:
        if col != 'Date':
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (10 * IQR)
            upper_bound = Q3 + (10 * IQR)
            bounds[col] = (lower_bound, upper_bound)

            outliers = df_cleaned[(df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)]

            for _, row in outliers.iterrows():
                outlier_info.append({
                    'Date': row['Date'],
                    'Column': col,
                    'Outlier Value': row[col]
                })

            # Remove outliers from the main DataFrame
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]

    return df_cleaned

def sigmoid(x):
    x = np.asarray(x, dtype=float)  # Ensure x is a float array
    x = np.nan_to_num(x)  # Replace NaNs with 0
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def create_graphs():
    # Plot actual vs predicted flow over time
    df = pd.read_excel(file_path, sheet_name='test_predictions')

    plt.figure(figsize=(14, 6))
    plt.plot(df['Date'], df['Actual Skelton fl'], label='Actual Skelton Flow', linewidth=2)
    plt.plot(df['Date'], df['Predicted MLP Skelton fl'], label='Predicted MLP Skelton Flow', linewidth=2, linestyle='--')
    plt.plot(df['Date'], df['Predicted Baseline Skelton fl'], label='Predicted Baseline Skelton Flow', linewidth=2, linestyle=':')

    plt.title('Predicted vs Actual Skelton Flow Over Time')
    plt.xlabel('Date')
    plt.ylabel('Skelton Flow')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

main()