# predict the rent of a house/apartment/flat based on its size, number of bedrooms/hallways/kitchens, and city
# Alberto Matute Beltran - A01704584
# 2/March/2023

# Dataset comes from: https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset

# import the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_error


# load the dataset into a pandas DataFrame
rent_data = pd.read_csv('LinearRegressionProject1\Project 1\House_Rent_Dataset.csv')

# print the first 5 rows of the dataset
print(rent_data.head())

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

# Create linear regression object
reg = linear_model.LinearRegression()

# Train the model
reg.fit(X_train, y_train)


# check the performance of the model


# Performance on training data
predicted_y = reg.predict(X_train)
train_mae = mean_absolute_error(y_train, predicted_y)



# Performance on test data
predicted_y = reg.predict(X_test)
test_mae = mean_absolute_error(y_test, predicted_y)

print("the mean absolute error on the training data is: %.2f" % train_mae)
print("the r-squared score is: %.2f" % r2_score(y_test, predicted_y))
print(f"and it's coefficients are: {reg.coef_}")

# plot the results of the model
plt.scatter(y_test, predicted_y)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.show()

# reduce the data_perpared to only 1000 rows
data_prepared = data_prepared[:1000]

target = target[:1000]

X_train, X_test, y_train, y_test = train_test_split(data_prepared, target, test_size=0.2, random_state=424)

# Train the model again
reg.fit(X_train, y_train)


# check the performance of the model
# Performance on training data
predicted_y = reg.predict(X_train)
train_mae = mean_absolute_error(y_train, predicted_y)



# Performance on test data
predicted_y = reg.predict(X_test)
test_mae = mean_absolute_error(y_test, predicted_y)

print("the mean absolute error on the training data is: %.2f" % train_mae)
print("the r-squared score is: %.2f" % r2_score(y_test, predicted_y))
print(f"and it's coefficients are: {reg.coef_}")

# plot the results of the model
plt.scatter(y_test, predicted_y)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.show()

# use cross validation to check the performance of the model
from sklearn.model_selection import cross_val_score

scores = cross_val_score(reg, data_prepared, target, scoring='neg_mean_absolute_error', cv=10)
print("the mean absolute error on the cross validation is: %.2f" % -scores.mean())

# make predictions until the user wants to stop
while True:
    bhk = input('Enter the number of bedrooms/hallways/kitchens (more than 1): ')
    size = input('Enter the size of the house/apartment/flat in square feet (more than 200 feet): ')
    city = input('Enter the city name: \n 1. Bangalore \n 2. Chennai \n 3. Delhi \n 4. Hyderabad \n 5. Kolkata \n 6. Mumbai \n')

    input_data = pd.DataFrame([(size, bhk, city)], columns= predictors)
    input_data_prepared = preprocessor.transform(input_data)

    predicted_rent = reg.predict(input_data_prepared)

    print("the predicted rent is: %.2f" % predicted_rent)

    # ask the user if they want to make another prediction
    another_prediction = input('Do you want to make another prediction? (y/n): ')
    if another_prediction == 'n':
        break







