import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# load the dataset into a pandas DataFrame
rent_data = pd.read_csv('Project 1\House_Rent_Dataset.csv')

# print the first 5 rows of the dataset
print(rent_data.head())

# print the graph of the number of houses in each city
sns.countplot(x='City', data=rent_data)
plt.show()

# print the count of rent values
print(rent_data['Rent'].value_counts())

# delete the rows with rent values that count less than 10
rent_data = rent_data[rent_data['Rent'].map(rent_data['Rent'].value_counts()) > 10]

# print the count of size values
print(rent_data['Size'].value_counts())

# delete the rows with size values that count less than 10
rent_data = rent_data[rent_data['Size'].map(rent_data['Size'].value_counts()) > 10]

# print the count of BHK values
print(rent_data['BHK'].value_counts())

# delete the rows with BHK values that count less than 10
rent_data = rent_data[rent_data['BHK'].map(rent_data['BHK'].value_counts()) > 10]

# print the count of rent values
print(rent_data['Rent'].value_counts())

# delete the rows with rent values that count less than 10
rent_data = rent_data[rent_data['Rent'].map(rent_data['Rent'].value_counts()) > 10]

# drop all columns except 'Size', 'BHK', 'City', 'Rent'
rent_data = rent_data.drop(['Posted On', 'Floor', 'Area Type', 'Area Locality', 'Furnishing Status', 'Tenant Preferred', 'Bathroom', 'Point of Contact'], axis=1)

# print the first 5 rows of the dataset
print(rent_data.head())

# prepare features for moddel fitting
predictors = ['Size', 'BHK', 'City']

data = rent_data[predictors]

target_name = 'Rent'
target = rent_data[target_name]

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

city_encoder = OrdinalEncoder(categories=[['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai']])
nominal_preprocessor = OneHotEncoder(handle_unknown='ignore')
numerical_preprocessor = StandardScaler()

from sklearn.compose import ColumnTransformer

city_attributes = ['City']
nominal_attributes = ['BHK']
numerical_attributes = ['Size']

preprocessor = ColumnTransformer([
    ('city', city_encoder, city_attributes),
    ('one-hot-enconder', nominal_preprocessor, nominal_attributes),
    ('standard-scaler', numerical_preprocessor, numerical_attributes)
])

data_prepared = preprocessor.fit_transform(data)

# separate the data into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_prepared, target, test_size=0.2, random_state=424)


# fit the model
from sklearn import linear_model

# Create linear regression object
reg = linear_model.LinearRegression()

# Train the model
reg.fit(X_train, y_train)


# check the performance of the model
from sklearn.metrics import mean_absolute_error, r2_score

# Performance on training data
predicted_y = reg.predict(X_train)
train_mae = mean_absolute_error(y_train, predicted_y)

from sklearn.metrics import mean_absolute_error

# Performance on test data
predicted_y = reg.predict(X_test)
test_mae = mean_absolute_error(y_test, predicted_y)

print("the mean absolute error on the training data is: %.2f" % train_mae)
print("the r-squared score is: %.2f" % r2_score(y_test, predicted_y))
print(f"and it's coefficients are: {reg.coef_}")

# plot the test error vs the training error
plt.plot(y_test, predicted_y, 'o')
plt.xlabel('Actual Rent')
plt.ylabel('Predicted Rent')
plt.show()


# make predictions

bhk = input('Enter the number of bedrooms/hallways/kitchens: ')
size = input('Enter the size of the house/apartment/flat in square feet: ')
city = input('Enter the city: ')

input_data = pd.DataFrame([(size, bhk, city)], columns= predictors)
input_data_prepared = preprocessor.transform(input_data)

predicted_rent = reg.predict(input_data_prepared)

print('Predicted rent: ', predicted_rent)



