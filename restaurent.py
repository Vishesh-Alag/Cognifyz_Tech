import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import folium 
from folium.plugins import MarkerCluster
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns


# Ignore all warnings
warnings.filterwarnings("ignore")
# Load the dfset
df = pd.read_csv('Dataset.csv')

# LEVEL 1 - TASK 1

# Check the number of rows and columns
print(f"Number of Rows: {df.shape[0]}")
print(f"Number of Columns: {df.shape[1]}")

# Display the first few rows to understand the dfset structure
#print(df.head())

# Check for missing values
missing_values = df.isnull().sum()
print("\nSum of Missing Values in Columns -- ")
print(missing_values)

# Handle missing values:
#replace missing 'Cuisines' with the mode or most frequent value
#df['Cuisines'].fillna(df['Cuisines'].mode()[0], inplace=True)

# Confirm there are no missing values left
#print(df.isnull().sum())

# Check df types
print(df.dtypes)

# Convert 'Country Code' to a string if it is treated as an integer
df['Country Code'] = df['Country Code'].astype(str)

# Convert 'Has Table booking', 'Has Online delivery', etc., to boolean values if needed
df['Has Table booking'] = df['Has Table booking'].map({'Yes': True, 'No': False})
df['Has Online delivery'] = df['Has Online delivery'].map({'Yes': True, 'No': False})
df['Is delivering now'] = df['Is delivering now'].map({'Yes': True, 'No': False})
df['Switch to order menu'] = df['Switch to order menu'].map({'Yes': True, 'No': False})

# Confirm changes
print(df.dtypes)




# Check distribution of 'Aggregate rating'
'''plt.figure(figsize=(8, 6))
sns.histplot(df['Aggregate rating'], bins=10, kde=True)
plt.title('Distribution of Aggregate Rating')
plt.xlabel('Aggregate Rating')
plt.ylabel('Frequency')
plt.show()'''

# Check for class imbalance by value counts
rating_counts = df['Aggregate rating'].value_counts()
print("\n Aggregate Rating Counts -- ")
print(rating_counts)
#class imbalance is present, 0 rating has the most number 

# LEVEL 1 - TASK 2

# 1. Numerical Statistics
numerical_columns = ['Average Cost for two', 'Aggregate rating', 'Votes']
numerical_stats = df[numerical_columns].describe()
print("Numerical Statistics:\n", numerical_stats)


# 2. Distribution of Categorical Variables
# Country Code Distribution
plt.figure(figsize=(10, 6))
country_counts = df['Country Code'].value_counts()
sns.barplot(x=country_counts.index, y=country_counts.values, palette='viridis')
plt.title("Country Code Distribution")
plt.xlabel("Country Code")
plt.ylabel("Number of Restaurants")
plt.xticks(rotation=45)
plt.show()

# City Distribution (Top 10 Cities)
plt.figure(figsize=(12, 6))
city_counts = df['City'].value_counts().head(10)
sns.barplot(x=city_counts.values, y=city_counts.index, palette='viridis')
plt.title("Top 10 Cities with Most Restaurants")
plt.xlabel("Number of Restaurants")
plt.ylabel("City")
plt.show()

# Cuisines Distribution (Top 10 Cuisines)
plt.figure(figsize=(12, 6))
top_cuisines = df['Cuisines'].value_counts().head(10)
sns.barplot(x=top_cuisines.values, y=top_cuisines.index, palette='viridis')
plt.title("Top 10 Cuisines")
plt.xlabel("Number of Restaurants")
plt.ylabel("Cuisines")
plt.show()

# LEVEL 1 - TASK 3

# Create a folium map centered around the average location of restaurants
'''m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)

# Create a MarkerCluster
marker_cluster = MarkerCluster().add_to(m)

# Add markers for each restaurant to the cluster
for i, row in df.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"{row['Restaurant Name']}, Rating: {row['Aggregate rating']}",
        icon=folium.Icon(color='blue')
    ).add_to(marker_cluster)

# Display the map
m.save('restaurants_map.html')'''


# Count the number of restaurants in each city
city_distribution = df['City'].value_counts()

# Combine lesser cities into "Other"
threshold = 50  # determines how many restaurants a city must have to be included as an individual bar. 
#Cities with fewer restaurants are aggregated into an "Other" category.

top_cities = city_distribution[city_distribution >= threshold]
other_cities = city_distribution[city_distribution < threshold].sum()
top_cities['Other'] = other_cities

# Create a horizontal bar plot for city distribution
plt.figure(figsize=(12, 8))
sns.barplot(x=top_cities.values, y=top_cities.index, palette='viridis')
plt.title("Distribution of Restaurants Across Cities")
plt.xlabel("Number of Restaurants")
plt.ylabel("City")
plt.show()



# Scatter plot of Latitude vs. Aggregate Rating
'''plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Longitude', y='Latitude', hue='Aggregate rating', palette='viridis', size='Votes', sizes=(20, 200), alpha=0.7)
plt.title("Location of Restaurants vs. Aggregate Rating")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(title='Aggregate Rating', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()'''


# Calculate the correlation coefficients
correlation_matrix = df[['Longitude', 'Latitude', 'Aggregate rating']].corr()

# Print the correlation matrix
print("Correlation Matrix Between Location (Longitude & Latitude) and Aggregate Rating :")
print(correlation_matrix)

# Scatter plot of Longitude vs. Aggregate Rating
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Longitude', y='Latitude', hue='Aggregate rating', palette='viridis', size='Votes', sizes=(20, 200), alpha=0.7)
plt.title("Location of Restaurants vs. Aggregate Rating")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(title='Aggregate Rating', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# LEVEL 1 ENDED

# LEVEL 2 - TASK 1 



# Calculate percentages of restaurants offering table booking and online delivery
table_booking_percentage = (df['Has Table booking'].mean() * 100)
online_delivery_percentage = (df['Has Online delivery'].mean() * 100)

# Calculate average ratings of restaurants with and without table booking
avg_rating_table_booking = df[df['Has Table booking']]['Aggregate rating'].mean()
avg_rating_no_table_booking = df[~df['Has Table booking']]['Aggregate rating'].mean()

# Prepare data for online delivery analysis by price range
price_delivery_analysis = df.groupby('Price range').agg({
    'Has Online delivery': 'mean',
    'Aggregate rating': 'mean'
}).reset_index()

# Calculate percentage of restaurants offering online delivery by price range
price_delivery_analysis['Percentage of Restaurants Offering Online Delivery'] = price_delivery_analysis['Has Online delivery'] * 100

# Visualization 1: Percentage of Restaurants Offering Table Booking and Online Delivery
percentages = [table_booking_percentage, online_delivery_percentage]
labels = ['Table Booking', 'Online Delivery']

plt.figure(figsize=(8, 5))
bars = sns.barplot(x=labels, y=percentages, palette='viridis')
plt.title("Percentage of Restaurants Offering Table Booking and Online Delivery")
plt.ylabel("Percentage (%)")
plt.ylim(0, 100)  # Set y-axis limit

# Add data labels
for bar in bars.patches:
    bars.annotate(f'{bar.get_height():.2f}%', 
                  (bar.get_x() + bar.get_width() / 2., bar.get_height()), 
                  ha='center', va='bottom', fontsize=12)

plt.show()

# Visualization 2: Average Ratings of Restaurants With and Without Table Booking
avg_ratings = [avg_rating_table_booking, avg_rating_no_table_booking]
rating_labels = ['With Table Booking', 'Without Table Booking']

plt.figure(figsize=(8, 5))
bars = sns.barplot(x=rating_labels, y=avg_ratings, palette='mako')
plt.title("Average Ratings of Restaurants With and Without Table Booking")
plt.ylabel("Average Rating")
plt.ylim(0, 5)  # Assuming ratings are on a scale of 0 to 5

# Add data labels
for bar in bars.patches:
    bars.annotate(f'{bar.get_height():.2f}', 
                  (bar.get_x() + bar.get_width() / 2., bar.get_height()), 
                  ha='center', va='bottom', fontsize=12)

plt.show()

# Visualization 3: Availability of Online Delivery Among Different Price Ranges
plt.figure(figsize=(12, 6))
bars = sns.barplot(data=price_delivery_analysis, x='Price range', y='Percentage of Restaurants Offering Online Delivery', palette='plasma')
plt.title("Availability of Online Delivery Among Different Price Ranges")
plt.ylabel("Percentage of Restaurants Offering Online Delivery (%)")
plt.xlabel("Price Range")
plt.ylim(0, 100)  # Set y-axis limit
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Add data labels
for bar in bars.patches:
    bars.annotate(f'{bar.get_height():.2f}%', 
                  (bar.get_x() + bar.get_width() / 2., bar.get_height()), 
                  ha='center', va='bottom', fontsize=12)

plt.show()
 
# LEVEL 2 - TASK 2 
# Price Range Analysis
# Determine the most common price range
most_common_price_range = df['Price range'].mode()[0]

# Calculate the average rating for each price range
average_rating_by_price_range = df.groupby('Price range')['Aggregate rating'].mean().reset_index()

# Identify the color that represents the highest average rating
average_rating_with_color = df.groupby(['Price range', 'Rating color'])['Aggregate rating'].mean().reset_index()
highest_avg_rating_row = average_rating_with_color.loc[
    average_rating_with_color['Aggregate rating'].idxmax()
]

highest_avg_rating = highest_avg_rating_row['Aggregate rating']
color_representing_highest_rating = highest_avg_rating_row['Rating color']

# Display results
print(f"\nMost Common Price Range: {most_common_price_range}")
print(f"Average Rating by Price Range:\n{average_rating_by_price_range}")
print(f"Color Representing Highest Average Rating: {color_representing_highest_rating} with an average rating of {highest_avg_rating:.2f}")

# Visualization of Average Rating by Price Range
plt.figure(figsize=(10, 6))
bars = sns.barplot(data=average_rating_by_price_range, x='Price range', y='Aggregate rating', palette='viridis')
plt.title("Average Rating by Price Range")
plt.ylabel("Average Rating")
plt.xlabel("Price Range")

# Add data labels to each bar
for bar in bars.patches:
    bars.annotate(f'{bar.get_height():.2f}', 
                  (bar.get_x() + bar.get_width() / 2., bar.get_height()), 
                  ha='center', va='bottom', fontsize=12)

plt.show()


# LEVEL 2 - TASK 3

# Display initial dataset information
print("Initial Dataset Info:")
print(df.info())

# Feature Engineering

# 1. Extract the length of the restaurant name
df['Name Length'] = df['Restaurant Name'].apply(len)

# 2. Extract the length of the address
df['Address Length'] = df['Address'].apply(len)

# 3. Count the number of different cuisines offered by each restaurant
df['Number of Cuisines'] = df['Cuisines'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)

# 4. Create a popularity score
# Popularity Score: Combine Votes and Aggregate Rating
df['Popularity Score'] = df['Votes'] * df['Aggregate rating']

# 5. Create a binary feature indicating if a restaurant has multiple cuisines
df['Has Multiple Cuisines'] = df['Number of Cuisines'].apply(lambda x: 1 if x > 1 else 0)

# Display the modified dataset with new features
print("\nModified Dataset with New Features:")
print(df[['Name Length', 'Address Length', 'Number of Cuisines', 'Popularity Score', 'Has Multiple Cuisines']].head())

# Optionally, save the modified dataset to a new CSV file
#df.to_csv('modifieddataset.csv', index=False)
#print("\nModified dataset saved as 'Restaurent_Modified_Dataset.csv'")

# Display dataset information to check types and missing values
print("\nModified Dataset Information:")
print(df.info())

'''df1 = pd.read_csv("Restaurent_Modified_Dataset.csv")

print(df1.info())'''
# LEVEL 2 COMPLETED 

# LEVEL 3 - TASK 1

# Load the dataset
df1 = pd.read_csv("modifieddataset.csv")

# Data Preprocessing
# Convert categorical variables to numerical using Label Encoding
label_encoder = LabelEncoder()
df1['Currency'] = label_encoder.fit_transform(df1['Currency'])
df1['Rating color'] = label_encoder.fit_transform(df1['Rating color'])
df1['Rating text'] = label_encoder.fit_transform(df1['Rating text'])
df1['Has Table booking'] = label_encoder.fit_transform(df1['Has Table booking'])
df1['Has Online delivery'] = label_encoder.fit_transform(df1['Has Online delivery'])
df1['Is delivering now'] = label_encoder.fit_transform(df1['Is delivering now'])
df1['Switch to order menu'] = label_encoder.fit_transform(df1['Switch to order menu'])

# Define feature variables and target variable
X = df1.drop(columns=['Aggregate rating', 'Restaurant ID', 'Restaurant Name', 'City', 'Address', 'Locality', 'Locality Verbose', 'Cuisines','Name Length','Address Length'])
y = df1['Aggregate rating']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to evaluate and compare models
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_mape = mean_absolute_percentage_error(y_train, y_pred_train)
    test_mape = mean_absolute_percentage_error(y_test, y_pred_test)
    
    print(f"{model_name} Performance:")
    print(f"Train MSE: {train_mse:.2f}, Test MSE: {test_mse:.2f}")
    print(f"Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}")
    print(f"Train MAPE: {train_mape:.2f}, Test MAPE: {test_mape:.2f}")
    print(f"Train R2: {train_r2:.2f}, Test R2: {test_r2:.2f}")
    print("-" * 50)
    
    return model, test_mse, test_r2

# Linear Regression
linear_reg = LinearRegression()
linear_model, linear_mse, linear_r2 = evaluate_model(linear_reg, X_train, X_test, y_train, y_test, "Linear Regression")

# Decision Tree Regressor
decision_tree = DecisionTreeRegressor(random_state=42)
tree_model, tree_mse, tree_r2 = evaluate_model(decision_tree, X_train, X_test, y_train, y_test, "Decision Tree")

# Random Forest Regressor
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
forest_model, forest_mse, forest_r2 = evaluate_model(random_forest, X_train, X_test, y_train, y_test, "Random Forest")

# Support Vector Machine Regressor
svm_reg = SVR()
svm_model, svm_mse, svm_r2 = evaluate_model(svm_reg, X_train, X_test, y_train, y_test, "SVM Regressor")

# K-Nearest Neighbors Regressor
knn_reg = KNeighborsRegressor()
knn_model, knn_mse, knn_r2 = evaluate_model(knn_reg, X_train, X_test, y_train, y_test, "KNN Regressor")

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(random_state=42)
gbr_model, gbr_mse, gbr_r2 = evaluate_model(gbr, X_train, X_test, y_train, y_test, "Gradient Boosting Regressor")

# XGBoost Regressor
xgb_reg = XGBRegressor(random_state=42)
xgb_model, xgb_mse, xgb_r2 = evaluate_model(xgb_reg, X_train, X_test, y_train, y_test, "XGBoost Regressor")

# Compare Models
models = ['Linear Regression', 'Decision Tree', 'Random Forest', 'SVM', 'KNN', 'Gradient Boosting', 'XGBoost']
mse_scores = [linear_mse, tree_mse, forest_mse, svm_mse, knn_mse, gbr_mse, xgb_mse]
r2_scores = [linear_r2, tree_r2, forest_r2, svm_r2, knn_r2, gbr_r2, xgb_r2]

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.barplot(x=models, y=mse_scores)
plt.xticks(rotation=45)
plt.title("Model Comparison - Mean Squared Error")
plt.ylabel("MSE")

plt.subplot(1, 2, 2)
sns.barplot(x=models, y=r2_scores)
plt.xticks(rotation=45)
plt.title("Model Comparison - R2 Score")
plt.ylabel("R2 Score")

plt.tight_layout()
plt.show()


# LEVEL 3 - TASK 2 

df2 = pd.read_csv("modifieddataset.csv")
# Fill missing values in 'Cuisines' with the mode
df2['Cuisines'].fillna(df2['Cuisines'].mode()[0], inplace=True)

# Explode the 'Cuisines' column if it contains multiple cuisines
df2['Cuisines'] = df2['Cuisines'].str.split(', ')
df_exploded = df.explode('Cuisines')

# Define cuisine categories with all mentioned cuisines covered
cuisine_categories = {
    'Asian': [
        'Japanese', 'Korean', 'Chinese', 'Indian', 'Thai', 'Filipino', 'Sushi', 
        'Vietnamese', 'Malaysian', 'Indonesian', 'Singaporean', 'Burmese', 
        'Modern Indian', 'South Indian', 'Pakistani', 'Afghani', 'Hyderabadi', 
        'Rajasthani', 'Sri Lankan', 'Biryani', 'Street Food', 'Goan', 'Naga', 
        'Mughlai', 'Chettinad', 'Kerala', 'Tibetan', 'Dim Sum', 'Teriyaki'
    ],
    'European': [
        'French', 'Italian', 'Spanish', 'German', 'Belgian', 'British', 
        'Irish', 'Portuguese', 'Greek', 'Dutch', 'Scandinavian'
    ],
    'American': [
        'American', 'Mexican', 'Brazilian', 'Cajun', 'Tex-Mex', 'New American', 
        'Southern', 'Soul Food', 'Fast Food', 'Burger', 'Diner', 'Bar Food', 
        'Grill', 'BBQ', 'Gourmet Fast Food'
    ],
    'Mediterranean': [
        'Mediterranean', 'Lebanese', 'Turkish', 'Middle Eastern', 'Egyptian'
    ],
    'Desserts': [
        'Desserts', 'Ice Cream', 'Bakery', 'Patisserie', 'Mithai'
    ],
    'Fusion': [
        'Fusion', 'Asian Fusion', 'World Cuisine'
    ],
    'Others': [
        'Seafood', 'Cafe', 'Buffet', 'Deli', 'Drinks Only', 'Tapas', 
        'Healthy Food', 'Salad', 'Snacks', 'Street Food', 'Fast Casual', 
        'Beverages', 'Finger Food', 'Pub Food', 'Charcoal Grill'
    ],
    'African': [
        'African', 'Moroccan', 'South African', 'Kenyan', 'Ethiopian'
    ],
    'Latin American': [
        'Cuban', 'Peruvian', 'Argentine', 'Latin American'
    ],
    'Pacific': [
        'Australian', 'Kiwi', 'Hawaiian', 'Modern Australian'
    ]
}

# Function to categorize cuisines
def categorize_cuisine(cuisine):
    for category, cuisines in cuisine_categories.items():
        if cuisine in cuisines:
            return category
    return 'Other'  # Default category for uncategorized cuisines

# Apply categorization
df_exploded['Cuisine Category'] = df_exploded['Cuisines'].apply(categorize_cuisine)

# Calculate average rating for each cuisine category
category_rating = df_exploded.groupby('Cuisine Category')['Aggregate rating'].mean().reset_index()

# Sort the categories by average rating
category_rating = category_rating.sort_values(by='Aggregate rating', ascending=False)

# Visualize the average ratings by cuisine category
plt.figure(figsize=(12, 8))
sns.barplot(x='Aggregate rating', y='Cuisine Category', data=category_rating, palette='viridis')
plt.title('Average Restaurant Ratings by Cuisine Category')
plt.xlabel('Average Rating')
plt.ylabel('Cuisine Category')
plt.xticks(rotation=0)

# Add average ratings on top of the bars
for index, value in enumerate(category_rating['Aggregate rating']):
    plt.text(value + 0.02, index, f"{value:.2f}", va='center')

plt.tight_layout()
plt.show()

# Print the average ratings for each cuisine category
print(category_rating)



# Calculate total votes for each cuisine
popular_cuisines = df_exploded.groupby('Cuisines')['Votes'].sum().reset_index()

# Sort cuisines by the number of votes
popular_cuisines = popular_cuisines.sort_values(by='Votes', ascending=False)

# Visualize the most popular cuisines
plt.figure(figsize=(12, 8))
sns.barplot(x='Votes', y='Cuisines', data=popular_cuisines.head(35), palette='coolwarm')  # Show top 35 cuisines
plt.title('Most Popular Cuisines Based on Number of Votes')
plt.xlabel('Total Votes')
plt.ylabel('Cuisine')
plt.xticks(rotation=45)

# Add total votes on top of the bars
for index, value in enumerate(popular_cuisines['Votes'].head(35)):
    plt.text(value + 0.02, index, f"{value}", va='center')

plt.tight_layout()
plt.show()

# Print the most popular cuisines
print(popular_cuisines.head(35))  # Print top 35 popular cuisines



# Set the style of seaborn
sns.set(style="whitegrid")

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1. Histogram for Aggregate Rating
sns.histplot(df2['Aggregate rating'], bins=10, kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Distribution of Aggregate Ratings')
axes[0].set_xlabel('Aggregate Rating')
axes[0].set_ylabel('Frequency')

# 2. Bar Plot for Average Count of Ratings
rating_counts = df2['Aggregate rating'].value_counts().sort_index()
sns.barplot(x=rating_counts.index, y=rating_counts.values, ax=axes[1], palette='viridis')
axes[1].set_title('Count of Restaurants by Aggregate Rating')
axes[1].set_xlabel('Aggregate Rating')
axes[1].set_ylabel('Count of Restaurants')

# Show the plots
plt.tight_layout()
plt.show()


# Convert boolean columns to 'Yes'/'No' and numeric (1/0)
df2['Has Table booking'] = df2['Has Table booking'].replace({True: 1, False: 0})
df2['Has Online delivery'] = df2['Has Online delivery'].replace({True: 1, False: 0})

# Check the data after conversion
print(df2[['Has Table booking', 'Has Online delivery']].head())

# Create ranges for Average Cost for Two in Thousands
bins = [0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000, np.inf]
labels = ['0-20K', '20K-40K', '40K-60K', '60K-80K', '80K-100K', '100K-120K', 
          '120K-140K', '140K-160K', '160K-180K', '180K-200K', 'Above 200K']
df2['Average Cost Range'] = pd.cut(df2['Average Cost for two'], bins=bins, labels=labels, right=False)

# Check the new ranges
print(df2[['Average Cost for two', 'Average Cost Range']].head())

# Set the style of seaborn
sns.set(style="whitegrid")


#  Box Plot for Aggregate Rating by Has Table Booking
plt.figure(figsize=(8, 6))
sns.boxplot(x='Has Table booking', y='Aggregate rating', data=df2, palette='Set2')
plt.title('Aggregate Rating by Table Booking')
plt.xlabel('Has Table Booking (1=Yes, 0=No)')
plt.ylabel('Aggregate Rating')
plt.show()

#  Box Plot for Aggregate Rating by Has Online Delivery
plt.figure(figsize=(8, 6))
sns.boxplot(x='Has Online delivery', y='Aggregate rating', data=df2, palette='Set2')
plt.title('Aggregate Rating by Online Delivery')
plt.xlabel('Has Online Delivery (1=Yes, 0=No)')
plt.ylabel('Aggregate Rating')
plt.show()

#  Box Plot for Average Cost Range and Aggregate Rating
plt.figure(figsize=(10, 6))
sns.boxplot(x='Average Cost Range', y='Aggregate rating', data=df2, palette='Set2')
plt.title('Aggregate Rating by Average Cost Range')
plt.xlabel('Average Cost Range (in Thousands)')
plt.ylabel('Aggregate Rating')
plt.xticks(rotation=45)
plt.show()

# Strip Plot for Average Cost Range vs. Aggregate Rating
plt.figure(figsize=(10, 6))
sns.stripplot(x='Average Cost Range', y='Aggregate rating', data=df2, color='blue', alpha=0.6)
plt.title('Aggregate Rating by Average Cost Range')
plt.xlabel('Average Cost Range (in Thousands)')
plt.ylabel('Aggregate Rating')
plt.xticks(rotation=45)
plt.show()

#  Heatmap for Correlation Matrix (using only numeric data)
numeric_data = df2.select_dtypes(include=[np.number])  # Select only numeric columns
correlation = numeric_data.corr()  # Calculate the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Heatmap')
plt.show()

# LEVEL 3 FINISHED 
# ----COMPLETED ----