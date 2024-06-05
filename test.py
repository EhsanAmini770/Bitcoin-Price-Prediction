import pandas as pd


# data = yf.download(tickers="^RUI", start="2012-03-11", end="2022-07-10")
data = pd.read_csv('afterindicatoradded.csv', parse_dates=True)
# Display the first few rows of the DataFrame
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Display data types of each column
print(data.dtypes)

import matplotlib.pyplot as plt
import seaborn as sns

# Set the date as the index
data.index = pd.to_datetime(data.index)

# Plot the closing price over time
plt.figure(figsize=(14, 7))
plt.title('Bitcoin Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.show()


# Display correlation matrix
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
