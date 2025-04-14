import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler  # ADDED

# Load training data
df = pd.read_csv('train.csv')

# Drop the 'Id' column as it's irrelevant for prediction
df = df.drop(['Id'], axis=1)

# Select only numeric columns for correlation matrix (this helps in calculating correlations only for numeric features)
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Now calculate the correlation matrix
corr = numeric_df.corr()

# Filter columns with positive correlation between 0.1 and 1 with 'SalePrice'  # MODIFIED threshold from 0.045 to 0.1
# This step selects the features that have a correlation with 'SalePrice' between 0.1 and 1.
# It helps to focus only on features that have some positive correlation with the target.
filtered_columns = corr[(corr['SalePrice'] > 0.1) & (corr['SalePrice'] <= 1)].index  # MODIFIED

# Select these columns from the original dataframe
# This creates a new dataframe with only the filtered columns (those that correlate positively with 'SalePrice')
df_filtered = df[filtered_columns]

# Handle columns with many missing values (e.g., drop columns with more than 50% missing values)
# This step ensures that we keep only the columns that have less than 50% missing values
df_filtered = df_filtered.loc[:, df_filtered.isnull().mean() < 0.5]

# Fill missing values in LotFrontage using the median (since it's a skewed column)
df_filtered['LotFrontage'] = df_filtered['LotFrontage'].fillna(df_filtered['LotFrontage'].median())

# Fill missing values in GarageYrBlt with the median (since it has a moderate correlation with 'SalePrice')
df_filtered['GarageYrBlt'] = df_filtered['GarageYrBlt'].fillna(df_filtered['GarageYrBlt'].median())

# **Handle MasVnrArea**: Fill missing values with the median (since it's numeric)
df_filtered['MasVnrArea'] = df_filtered['MasVnrArea'].fillna(df_filtered['MasVnrArea'].median())

# Fill remaining missing values (if any) using the median of their respective columns  # ADDED
df_filtered = df_filtered.fillna(df_filtered.median())  # ADDED

# Now, df_filtered has the selected columns with fewer missing values
# Display the null counts for all columns without truncation
# pd.set_option('display.max_rows', None)  # This will ensure all rows are shown
# print(df_filtered.isnull().sum())  # Check the missing values in the filtered dataframe
# pd.reset_option('display.max_rows')  # Reset the option to default after you're done

# Calculate correlation for the filtered dataframe
corr_filtered = df_filtered.corr()

# Plotting the correlation matrix to visualize relationships between features and 'SalePrice'
# plt.figure(figsize=(12, 8))  # Set the size of the plot
# sns.heatmap(corr_filtered[['SalePrice']].sort_values(by='SalePrice', ascending=False), annot=True, cmap='coolwarm')
# plt.show()  # Show the heatmap plot

# Assuming df_filtered is your modified dataframe with all irrelevant columns dropped and missing values filled
file_path = 'modified_house_data.csv'

# Save the modified dataframe to a CSV file with a try-except block
try:
    df_filtered.to_csv(file_path, index=False)
    print(f"Modified data saved to {file_path}")
except Exception as e:
    print(f"Error saving the file: {e}")

X = df_filtered.drop('SalePrice', axis=1)  # dropping SalePrice cuz that's the one we need to predict
y = df_filtered['SalePrice']     # making SalePrice as dependent variable  # FIXED COMMENT

# Feature scaling using StandardScaler  # ADDED
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting scaled data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)   # random state can be any number and 42 is arbitrary, and random_state is used for ensuring the training set is reproducible

print(pd.DataFrame(X_train).isnull().sum())  # Check for NaNs

# Training the model using Linear Regression
clf = LinearRegression()
clf.fit(X_train, y_train)

# predict y using the test data
y_pred = clf.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Plot actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Actual vs Predicted SalePrice')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # ADDED ideal prediction line
plt.show()

# Residual plot to check errors distribution  # ADDED
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.xlabel("Prediction Error (Residual)")
plt.show()
