
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

sns.set(style="whitegrid", palette="pastel", font_scale=1.1)
plt.rcParams['figure.dpi'] = 120

# =========================================================
# 1. DATA PREPARATION 
# =========================================================
# Step-1 We will first load the oceanic countries dataset
df_oceanic = pd.read_csv("D:/project/exports-to-oceanic-countries.csv")
print(df_oceanic.head().sum())
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



# =========================================================
# 2. EDA & OUTLIER DETECTION USING IQR Method
# =========================================================
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



# =========================================================
# 3. STATISTICAL ANALYSIS USING T-Test AND Z-Test
# =========================================================
print("\n--- Statistical Hypothesis Testing ---")
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



# =========================================================
# 4. MACHINE LEARNING Using Simple Linear Regression - SLR
# =========================================================
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




# =========================================================
# 5. VISUALIZATIONS (Matplotlib & Seaborn)
# =========================================================

# Plot 1: Box Plot for Outlier Detection
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Value'], color='#66c2a5')
plt.title('Outlier Detection in Export Value', fontsize=14, weight='bold')
plt.xlabel('Export Value ($)')
plt.grid(alpha=0.2)
plt.show()

# Plot 2: Correlation Heatmap
plt.figure(figsize=(8,6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Between Export Metrics', fontsize=14, weight='bold')
plt.show()

# Plot 3: SLR Regression Plot (Scatter + Line)
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.3, s=10, color='gray', label='Actual Trade Data')
#Sort values before plotting line
sorted_indices = X['Quantity'].argsort()
X_sorted = X['Quantity'].iloc[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]

plt.plot(X_sorted, y_pred_sorted, color='red', linewidth=2, label='Regression Trend Line')
plt.title('Simple Linear Regression: Quantity vs Value', fontsize=14, weight='bold')
plt.xlabel('Quantity of Goods')
plt.ylabel('Export Value ($)')
plt.legend()
plt.grid(alpha=0.2)
plt.show()

# Plot 4: Conclusion Bar Graph (Top 15 Countries)
top15 = df.groupby('Country')['Value'].mean().sort_values().tail(15)
plt.figure(figsize=(10,7))
bars = plt.barh(top15.index, top15.values, color='#4c72b0')
plt.title('Top 15 Oceanic Countries by Export Value', fontsize=14, weight='bold')
plt.xlabel('Average Export Value ($)')
# Add labels on bars
for i, v in enumerate(top15.values):
    plt.text(v, i, f'{v:,.0f}', va='center')
plt.tight_layout()
plt.show()

# Plot 5: Actual vs Predicted Values
plt.figure(figsize=(8,6))
plt.scatter(y, y_pred, alpha=0.4)

plt.plot([y.min(), y.max()], [y.min(), y.max()],
         color='red', linewidth=2)

plt.title('Model Accuracy: Actual vs Predicted', fontsize=14, weight='bold')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(alpha=0.2)
plt.show()

# Plot 6: Histogram (Distribution of Values)
plt.figure(figsize=(8,5))
sns.histplot(df['Value'], kde=True, bins=40, color='#ff9f40')
plt.title('Distribution of Export Values', fontsize=14, weight='bold')
plt.xlabel('Export Value ($)')
plt.show()

# Plot 7: KDE Density Plot
plt.figure(figsize=(9,5))
sns.kdeplot(
    df['Value'], 
    fill=True, 
    color='#4c72b0', 
    alpha=0.6, 
    linewidth=2
)
# We Add mean line
mean_val = df['Value'].mean()
plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:,.0f}')

plt.title('Density Distribution of Export Values', fontsize=14, weight='bold')
plt.xlabel('Export Value ($)')
plt.ylabel('Density')
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# Plot 8: Scatter Plot (Raw Relationship)
plt.figure(figsize=(9,5))
plt.scatter(
    df['Quantity'], 
    df['Value'], 
    alpha=0.4, 
    s=25, 
    color='#4c72b0', 
    edgecolor='black'
)
plt.title('Scatter Plot: Quantity vs Export Value', fontsize=14, weight='bold')
plt.xlabel('Quantity of Goods')
plt.ylabel('Export Value ($)')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# Plot 9: Regression Plot (Seaborn Improved)
plt.figure(figsize=(9,5))
sns.regplot(
    x='Quantity',
    y='Value',
    data=df,
    scatter_kws={'alpha':0.3, 's':20, 'color':'#4c72b0'},
    line_kws={'color':'#d62728', 'linewidth':2}
)
plt.title(' Regression Trend: Quantity vs Export Value', fontsize=14, weight='bold')
plt.xlabel('Quantity of Goods')
plt.ylabel('Export Value ($)')
plt.grid(alpha=0.2)

plt.tight_layout()
plt.show()


# Plot 10: Top 10 Countries Horizontal Bar
top10 = df.groupby('Country')['Value'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 6))
top10.sort_values().plot(kind='barh', color='skyblue')
plt.title('Top 10 Countries by Export Value')
plt.xlabel('Average Value ($)')
plt.ylabel('Country')
plt.show()

# Plot 11: Pie Chart (Top Countries Share)
plt.figure(figsize=(8,8))
colors = sns.color_palette('pastel')
top10.plot(
    kind='pie',
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    pctdistance=0.8
)
plt.title(' Export Share Distribution (Top 10 Countries)', fontsize=14, weight='bold')
plt.ylabel('')
#we impose draw circle for donut
centre_circle = plt.Circle((0,0), 0.60, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.tight_layout()
plt.show()

#STACK AREA CHART
# STACK AREA CHART (only if Year exists)
if 'Year' in df.columns:
    top_countries = df.groupby('Country')['Value'].sum().nlargest(5).index

    area_data = df[df['Country'].isin(top_countries)]
    area_data = area_data.pivot_table(values='Value', index='Year', columns='Country', aggfunc='sum')

    area_data.plot(kind='area', figsize=(10,6), alpha=0.7)

    plt.title('Top Countries Export Trend (Stacked Area)')
    plt.xlabel('Year')
    plt.ylabel('Export Value')

    plt.show()
else:
    print("Skipping area chart (Year not available)")

#summary
summary = df[['Value','Quantity']].describe()

print("📊 Summary Statistics:")
print(summary)

print("\n--- Full Project Analysis for Oceanic Dataset Complete ---")
print("\n Insights:")
print("✔️ Export values are highly skewed with some extreme outliers.")
print("✔️ Quantity and Value show a positive relationship.")
print("✔️ Few countries and commodities dominate exports.")
print("✔️ Market is uneven — strong opportunities in top performers.")
