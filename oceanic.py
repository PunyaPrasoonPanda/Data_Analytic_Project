
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

sns.set(style="whitegrid", palette="pastel", font_scale=1.1)
plt.rcParams['figure.dpi'] = 120

# 1. DATA PREPARATION 
# Step-1 We will first load the oceanic countries dataset
df_oceanic = pd.read_csv("D:/project/exports-to-oceanic-countries.csv")
print(df_oceanic.head())
print(df_oceanic.info())
print(df_oceanic.describe())

# Step-2 We will now simplify column names by renaming for easier coding
# value_dl = Export Value ($), value_qt = Quantity, country_name = Country
df = df_oceanic.rename(columns={
    'value_dl': 'Value',
    'value_qt': 'Quantity',
    'country_name': 'Country',
    'commodity': 'Commodity'
})


# Step-2.1 Handling date column (for time-based analysis)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month
    print("Date column processed successfully")
else:
    print("No date column found, skipping time-based features")
    
# Step-3 We will now Clean the raw data: Convert to numeric and remove missing rows
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df = df.dropna(subset=['Value', 'Quantity'])

print("Data Loaded Successfully")
print("Data after cleaning:", df.shape)

# 2. EDA & OUTLIER DETECTION USING IQR Method
# Step-1 We will first calculate Interquartile Range (IQR)
Q1 = df['Value'].quantile(0.25)
Q3 = df['Value'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Step-3 Then we Identify outliers
outliers = df[(df['Value'] < lower_bound) | (df['Value'] > upper_bound)]

print("\n--- EDA: Outlier Detection ---")
print(f"Q1: {Q1:.4f}, Q3: {Q3:.4f}, IQR: {IQR:.4f}")
print(f"Total Outliers Detected: {len(outliers)}")



# 3. STATISTICAL ANALYSIS USING T-Test AND Z-Test
print("Statistical Hypothesis Testing")
# Step-1 We will first do Null Hypothesis (H0): The population mean of export value is 0.1
null_mean = 0.1

# Step-2 Then we perform T-Test
t_stat, p_val_t = stats.ttest_1samp(df['Value'], null_mean)

# Step-3 Then we perform Z-Test(Manual Calculation)
sample_mean = df['Value'].mean()
std_error = df['Value'].std() / np.sqrt(len(df))
z_stat = (sample_mean - null_mean) / std_error
p_val_z = 2 * (1 - stats.norm.cdf(abs(z_stat)))

print(f"T-test P-Value: {p_val_t:.4f}")
print(f"Z-test P-Value: {p_val_z:.4f}")


# 4. MACHINE LEARNING Using Simple Linear Regression - SLR
# Step-1 First we create an independent variable (X) and Dependent variable (y)
X = df[['Quantity']]
y = df['Value']

# Step-2 Then we Create the Regression and fit it to the model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

print("\n--- SLR Model Output ---")
print(f"Intercept (Beta 0): {model.intercept_:.4f}")
print(f"Slope (Beta 1): {model.coef_[0]:.8f}")

r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

print(f"R-Squared Score: {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


#5. VISUALIZATIONS

# 1. Boxplot - outliers
plt.figure(figsize=(7,4))
sns.boxplot(x=df['Value'])
plt.title("Outliers in Export Value")
plt.xlabel("Export Value")
plt.show()


# 2. Correlation heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df[['Value','Quantity']].corr(), annot=True)
plt.title("Correlation between Quantity and Value")
plt.xlabel("Features")
plt.ylabel("Features")
plt.show()


# 3. Scatter plot
plt.figure(figsize=(7,4))
plt.scatter(df['Quantity'], df['Value'], alpha=0.3)
plt.title("Quantity vs Export Value")
plt.xlabel("Quantity")
plt.ylabel("Export Value")
plt.show()


# 4. Regression plot
plt.figure(figsize=(7,4))
sns.regplot(x='Quantity', y='Value', data=df)
plt.title("Trend between Quantity and Value")
plt.xlabel("Quantity")
plt.ylabel("Export Value")
plt.show()


# 5. Histogram + KDE
plt.figure(figsize=(7,4))
sns.histplot(df['Value'], kde=True)
plt.title("Distribution of Export Values")
plt.xlabel("Export Value")
plt.ylabel("Frequency")
plt.show()


# 6. KDE plot
plt.figure(figsize=(7,4))
sns.kdeplot(df['Value'], fill=True)
plt.title("Density of Export Values")
plt.xlabel("Export Value")
plt.ylabel("Density")
plt.show()


# 7. Top countries (bar chart)
top10 = df.groupby('Country')['Value'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(8,5))
top10.plot(kind='bar')
plt.title("Top 10 Countries by Average Export Value")
plt.xlabel("Country")
plt.ylabel("Average Export Value")
plt.xticks(rotation=45)
plt.show()


# 8. Horizontal bar
plt.figure(figsize=(8,5))
top10.sort_values().plot(kind='barh')
plt.title("Top 10 Countries Comparison")
plt.xlabel("Average Export Value")
plt.ylabel("Country")
plt.show()


# 9. Pie chart
plt.figure(figsize=(6,6))
top10.plot(kind='pie', autopct='%1.1f%%')
plt.title("Export Share of Top Countries")
plt.ylabel("")
plt.show()


# 10. Actual vs Predicted
plt.figure(figsize=(6,4))
plt.scatter(y, y_pred)

plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')

plt.title("Actual vs Predicted Values")
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.show()


# 11. Yearly trend
if 'Year' in df.columns:
    yearly = df.groupby('Year')['Value'].sum()

    plt.figure(figsize=(8,5))
    plt.plot(yearly.index, yearly.values, marker='o')
    plt.title("Yearly Export Trend")
    plt.xlabel("Year")
    plt.ylabel("Total Export Value")
    plt.show()


# 12. Area plot
if 'Year' in df.columns:
    plt.figure(figsize=(8,5))
    plt.fill_between(yearly.index, yearly.values)
    plt.title("Overall Export Trend")
    plt.xlabel("Year")
    plt.ylabel("Export Value")
    plt.show()


# 13. Stacked area
if 'Year' in df.columns:
    top3 = df.groupby('Country')['Value'].sum().nlargest(3).index

    area = df[df['Country'].isin(top3)]
    area = area.pivot_table(values='Value', index='Year', columns='Country', aggfunc='sum')

    area.plot(kind='area', figsize=(8,5))
    plt.title("Top Countries Contribution Over Time")
    plt.xlabel("Year")
    plt.ylabel("Export Value")
    plt.show()

    
#summary
summary = df[['Value','Quantity']].describe()

print("Summary Statistics:")
print(summary)

print("Full Project Analysis for Oceanic Dataset Complete")
print("Insights")
print("✔️ Export values are highly skewed with some extreme outliers.")
print("✔️ Quantity and Value show a positive relationship.")
print("✔️ Few countries and commodities dominate exports.")
print("✔️ Market is uneven — strong opportunities in top performers.")
