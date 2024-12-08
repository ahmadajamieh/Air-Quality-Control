import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import mean_squared_error
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
data = pd.read_csv('C:/Users/ahmad/OneDrive/Desktop/greater_amman_municipality_amman_air_quality.csv')

# Reduce the dataset to 768 rows
data = data.head(768)

# Strip whitespace from column names
data.columns = data.columns.str.strip()

# Remove outliers (values above 100)
pollutants = ['pm10', 'no2', 'so2', 'co']
for pollutant in pollutants:
    data = data[data[pollutant] <= 100]

# Save or inspect the cleaned data
data.to_csv('cleaned_air_quality_data.csv' , index=False)

# Convert 'date' to datetime format
data['date'] = pd.to_datetime(data['date'])

# Set 'date' as the index
data.set_index('date', inplace=True)

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Fill missing values with the mean of each column
data.fillna(data.mean(numeric_only=True), inplace=True)

# List of pollutants
pollutants = ['pm10', 'no2', 'so2', 'co']

# Plot pollutant levels over time
plt.figure(figsize=(12, 8))
for pollutant in pollutants:
    plt.scatter(data.index, data[pollutant], label=pollutant, s=10)
plt.title('Pollutant Levels Over Time')
plt.xlabel('Date')
plt.ylabel('Concentration')
plt.legend()
plt.show()

# Reset index to access 'date' column
data.reset_index(inplace=True)

# Extract time-based features
data['hour'] = data['date'].dt.hour
data['day_of_week'] = data['date'].dt.dayofweek  # Monday=0, Sunday=6
data['month'] = data['date'].dt.month
data['day_of_year'] = data['date'].dt.dayofyear
data['week_of_year'] = data['date'].dt.isocalendar().week

# Encode cyclic time features using sine and cosine transformations
data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

# Add seasonal indicators to retain and arrange seasonal patterns
data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
data['day_of_year_sin'] = np.sin(2 * np.pi * data['day_of_year'] / 365)
data['day_of_year_cos'] = np.cos(2 * np.pi * data['day_of_year'] / 365)

# Update the list of features to include seasonal indicators
features = ['pm10', 'no2', 'so2', 'co',
            'hour', 'day_sin', 'day_cos', 
            'month_sin', 'month_cos', 
            'day_of_year_sin', 'day_of_year_cos']

# Fill any remaining missing values in features
data[features] = data[features].fillna(data[features].mean())

# Scale the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[features])

##############################################################################################################################

# K-Means Clustering
data.set_index('date', inplace=True)

# Apply K-Means with the optimal number of clusters (e.g., K=6)
optimal_k = 6  # Replace with the optimal number based on the Elbow Curve
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_data)

# Add cluster labels to the data
data['KMeans_Cluster'] = kmeans_labels

# Divide the data into three equal time segments
time_splits = np.array_split(data.index, 6)
colors = ['blue', 'green', 'purple', 'orange', 'pink', 'brown']  # Colors for the time segments

# Plot each pollutant individually with clusters and centroids
for pollutant in pollutants:
    plt.figure(figsize=(10, 6))

    # Show Cluster Centroides 
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    centroids_df = pd.DataFrame(centroids, columns=features)

    # Plot each time segment with its own color
    for i, time_segment in enumerate(time_splits):
        segment_data = data.loc[time_segment]
        plt.scatter(segment_data.index, segment_data[pollutant], c=colors[i], s=10, alpha=0.6, label=f'Segment {i+1}')
        cluster_label = segment_data['KMeans_Cluster'].iloc[0]

    # Plot centroid as a dot at the midpoint of each time segment
        midpoint_index = time_segment[len(time_segment) // 2]
        plt.scatter(midpoint_index, centroids_df[pollutant][i], color='red', marker='o', s=10, 
                    label=f'Centroid Cluster {i+1}' if i == 0 else "")
    
    # Calculate distances
        distances = np.abs(segment_data[pollutant] - centroids_df[pollutant][cluster_label])
        print(f"\nSegment {i+1} - Cluster {cluster_label+1} Distances for {pollutant.upper()}:")
        print(distances.describe())  # Summary statistics of distances

    plt.title(f'{pollutant.upper()} Levels by Time Segments')
    plt.ylabel(f'{pollutant.upper()} Concentration')
    plt.xlabel('Date')
    plt.legend()
    plt.show()

# Evaluate K-Means clustering
kmeans_silhouette = silhouette_score(scaled_data, kmeans_labels)
print(f'K-Means Silhouette Score: {kmeans_silhouette:.2f}')

##############################################################################################################################

# Gaussian Mixture Model Clustering
# Determine the optimal number of components using BIC
bic_scores = []
n_components = range(1, 11)
for n in n_components:
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(scaled_data)
    bic_scores.append(gmm.bic(scaled_data))

# Plot BIC Scores
plt.figure(figsize=(8, 4))
plt.scatter(n_components, bic_scores, marker='o')
plt.title('BIC Scores for GMM')
plt.xlabel('Number of Components')
plt.ylabel('Bayesian Information Criterion (BIC)')
plt.show()

# Apply GMM with the optimal number of components (e.g., n_components=3)
optimal_n = 6  # Replace with the optimal number based on the BIC plot
gmm = GaussianMixture(n_components=optimal_n, random_state=42)
gmm_labels = gmm.fit_predict(scaled_data)
print(gmm_labels)
means = gmm.means_
covariences = gmm.covariances_
print("means:\n", means)
print("covariences:\n", covariences)

# Add cluster labels to the data
data['GMM_Cluster'] = gmm_labels

# Evaluate GMM clustering
gmm_silhouette = silhouette_score(scaled_data, gmm_labels)
print(f'GMM Silhouette Score: {gmm_silhouette:.2f}')

##############################################################################################################################

# Analyze Cluster Characteristics
# K-Means Cluster Analysis
kmeans_analysis = data.groupby('KMeans_Cluster')[pollutants + ['hour']].mean()
print("\nK-Means Cluster Analysis:")
print(kmeans_analysis)

# GMM Cluster Analysis
gmm_analysis = data.groupby('GMM_Cluster')[pollutants + ['hour']].mean()
print("\nGMM Cluster Analysis:")
print(gmm_analysis)

##############################################################################################################################

# Perform hierarchical clustering
linked = linkage(scaled_data, method='ward')  # Ward's method for minimizing variance

# Plot the dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linked, truncate_mode='lastp', p=12, leaf_rotation=45, leaf_font_size=12, show_contracted=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
plt.show()

# Apply Agglomerative Clustering with a specific number of clusters
n_clusters = 3
hc = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
hc_labels = hc.fit_predict(scaled_data)

# Add hierarchical cluster labels to the dataset
data['HC_Cluster'] = hc_labels

# Visualize HC clusters for all pollutants
for pollutant in ['pm10', 'no2', 'so2', 'co']:
    plt.figure(figsize=(12, 6))
    plt.scatter(data.index, data[pollutant], c=hc_labels, cmap='viridis', s=10)
    plt.title(f'{pollutant.upper()} Levels Colored by Hierarchical Clusters')
    plt.xlabel('Date')
    plt.ylabel(f'{pollutant.upper()} Concentration')
    plt.colorbar(label='Cluster')
    plt.show()

# Save the clustered data (Optional)
data.reset_index(inplace=True)  # Reset index if needed
data.to_csv('clustered_air_quality_data.csv', index=False)

##############################################################################################################################

# Neural Network

# Convert 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')

# Set 'date' as the index
data.set_index('date', inplace=True)

# Sort the index to ensure chronological order
data = data.sort_index()

# Define the sequence length for LSTM
sequence_length = 30

for pollutant in ['pm10', 'no2', 'so2', 'co']:
    print(f"\nProcessing {pollutant.upper()}...")

    # Split the data into training (2021-2023) and testing (2024)
    train_data = data.loc['2021-01-01':'2023-12-31']
    test_data = data.loc['2024-01-01':'2024-12-31']

    # Select a pollutant to forecast
    train_series = train_data[[pollutant]]
    test_series = test_data[[pollutant]]

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_series)
    test_scaled = scaler.transform(test_series)

    # Create sequences for LSTM
    def create_sequences(data, sequence_length):
        x, y = [], []
        for i in range(len(data) - sequence_length):
            x.append(data[i:i + sequence_length, 0])
            y.append(data[i + sequence_length, 0])
        return np.array(x), np.array(y)

    sequence_length = 30  # Number of past days to use for prediction
    x_train, y_train = create_sequences(train_scaled, sequence_length)
    x_test, y_test = create_sequences(test_scaled, sequence_length)

    # Reshape input to [samples, time steps, features]
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # Build the LSTM model
    model = Sequential([
       LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
       Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1)

    # Forecast for the testing period
    predicted_scaled = model.predict(x_test)
    predicted = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))

    # Calculate and print the MSE
    mse = mean_squared_error(test_series[sequence_length:], predicted)
    print(f'Mean Squared Error for {pollutant.upper()}: {mse:.2f}')

    # Visualize actual vs predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(train_series.index, train_series, label='Training Data', color='blue')
    plt.plot(test_series.index, test_series, label='Actual Testing Data', color='green')
    plt.plot(test_series.index[sequence_length:], predicted, label='Forecasted Data', color='red', linestyle='--')
    plt.title(f'{pollutant.upper()} Forecast: Training vs Testing')
    plt.xlabel('Date')
    plt.ylabel(f'{pollutant.upper()} Concentration')
    plt.legend()
    plt.show()

 