import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Get market data
def get_stock_data():
    data = pd.read_csv('HistoricalData_SPY_1Y.csv')
    # Preprocess the data
    data.columns = data.columns.str.strip()  # Remove leading/trailing spaces from column names
    data['Date'] = pd.to_datetime(data['Date'])  # Convert 'Date' to datetime
    data = data.sort_values('Date')  # Sort by date
    data['Returns'] = data['Close/Last'].pct_change()  # Calculate daily returns
    data = data.dropna()  # Drop rows with NaN values
    return data
    
# Compute Shannon Entropy (Market Uncertainty)
def compute_entropy(returns, window=20):
    entropies = []
    for i in range(len(returns)):
        window_data = returns[max(0, i-window):i]
        hist, _ = np.histogram(window_data, bins=10, density=True)
        hist = hist / hist.sum()  # Normalize
        entropies.append(entropy(hist + 1e-10))  # Add small value to avoid log(0)
    return np.array(entropies)

# Compute Momentum (Price Rate of Change)
def compute_momentum(df, window=10): #short-term momentum window
    return df['Close/Last'].diff(periods=window) #close_(i) - close_(i-5)

# Compute volatility (brownian motion)
def compute_volatility_brownian(df, window=14):
    
    # Calculate log returns (log of price ratios)
    df['Log_Returns'] = np.log(df['Close/Last'] / df['Close/Last'].shift(1))
    
    # Calculate rolling standard deviation of log returns as the volatility
    volatility_brownian = df['Log_Returns'].rolling(window=window).std()
    
    return volatility_brownian

# Compute Volatility (Average True Range)
def compute_volatility(df, window=14): #14 days is the default 
    df['High-Low'] = df['High'] - df['Low'] #day's highest - lowest price
    df['High-Close'] = np.abs(df['High'] - df['Close/Last'].shift(1)) # day_i high price - day_(i-1) closing price
    df['Low-Close'] = np.abs(df['Low'] - df['Close/Last'].shift(1)) # day_i low price - day_(i-1) closing price
    df['TR'] = df[['High-Low', 'High-Close', 'Low-Close']].max(axis=1) # max of HL, HC and LC
    return df['TR'].rolling(window=window).mean() #rolling avg over 14 days

# Label Market Conditions (Target Variable)
def label_market(df):
    
    df["Market_Condition"] = np.where(df["Returns"] > 0.003, "Bull",  np.where(df["Returns"] < -0.003, "Bear", "Sideways"))
    return df

# Fetch data and compute indicators
df = get_stock_data()
df["Entropy"] = compute_entropy(df["Returns"])
df["Momentum"] = compute_momentum(df)
df["Volatility"] = compute_volatility(df)
df["Volatility_GBM"] = compute_volatility_brownian(df)

df = label_market(df).dropna()

# Prepare Data for Naive Bayes (Standard and Physics-Enhanced)
X_standard = df[["Returns", "Volatility"]]  # Standard features
X_physics = df[["Returns",  "Entropy", "Momentum", "Volatility_GBM", ]]  # Physics-enhanced features

y = df["Market_Condition"]

# Standardize the data
scaler = StandardScaler() #shifts data so mean = 0, stdev=1
X_standard_scaled = scaler.fit_transform(X_standard)
X_physics_scaled = scaler.fit_transform(X_physics)

X_train_standard, X_test_standard, y_train, y_test = train_test_split(X_standard_scaled, y, test_size=0.2, random_state=42)
X_train_physics, X_test_physics, _, _ = train_test_split(X_physics_scaled, y, test_size=0.2, random_state=42)
#test size: 20% (20 % of data used for testing, 80% data used for training)
#random_state = const; ensures same results acquired every time code is run 

rf = RandomForestClassifier()
rf.fit(X_physics_scaled, y)
print(dict(zip(['Returns', 'Entropy', 'Momentum', 'Volatility_GBM', 'Volatility'], rf.feature_importances_)))

# Train Standard Naïve Bayes Model

standard_nb = GaussianNB() #initialises a NB classifier
standard_nb.fit(X_train_standard, y_train) # trains Gaussian NB classifier 

# Train Physics-Enhanced Naïve Bayes Model
physics_nb = GaussianNB()
physics_nb.fit(X_train_physics, y_train)

# Predict and Evaluate both models
y_pred_standard = standard_nb.predict(X_test_standard)
y_pred_physics = physics_nb.predict(X_test_physics)

# Accuracy Scores for both models
acc_standard = accuracy_score(y_test, y_pred_standard)
acc_physics = accuracy_score(y_test, y_pred_physics)


y_proba_standard = standard_nb.predict_proba(X_test_standard)  # For the standard Naive Bayes model. an array with each row = a sample, each column = prob of sample belonging to each class 
y_proba_physics = physics_nb.predict_proba(X_test_physics)  # For the physics-enhanced Naive Bayes model


print("Predicted probabilities for the first sample (Standard Naive Bayes):")
print(f"Bull: {y_proba_standard[0][0]:.2f}, Bear: {y_proba_standard[0][1]:.2f}, Sideways: {y_proba_standard[0][2]:.2f}")

# Similarly for the physics-enhanced model
print("Predicted probabilities for the first sample (Physics-Enhanced Naive Bayes):")
print(f"Bull: {y_proba_physics[0][0]:.2f}, Bear: {y_proba_physics[0][1]:.2f}, Sideways: {y_proba_physics[0][2]:.2f}")

print(f"Standard Naïve Bayes Accuracy: {acc_standard * 100:.2f}%")
print(f"Physics-Enhanced Naïve Bayes Accuracy: {acc_physics * 100:.2f}%")