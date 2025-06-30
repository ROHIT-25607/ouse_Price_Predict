import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

house_data = {
    'Area': [1200, 1500, 1000, 1800, 2000, 1100],
    'Location': ['Downtown', 'Suburb', 'Downtown', 'Suburb', 'Countryside', 'Downtown'],
    'Rooms': [3, 4, 2, 4, 5, 2],
    'Price': [250000, 300000, 200000, 320000, 270000, 210000]
}

dataframe = pd.DataFrame(house_data)

input_features = dataframe[['Area', 'Location', 'Rooms']]
target_prices = dataframe['Price']

number_columns = ['Area', 'Rooms']
category_columns = ['Location']

scale_numbers = StandardScaler()
encode_categories = OneHotEncoder(drop='first')

preprocessing_steps = ColumnTransformer([
    ('scale', scale_numbers, number_columns),
    ('encode', encode_categories, category_columns)
])

price_model = Pipeline([
    ('prepare', preprocessing_steps),
    ('predictor', LinearRegression())
])

train_features, test_features, train_prices, test_prices = train_test_split(
    input_features, target_prices, test_size=0.2, random_state=42
)

price_model.fit(train_features, train_prices)

estimated_prices = price_model.predict(test_features)

print("Estimated Prices:", estimated_prices)
print("MSE:", mean_squared_error(test_prices, estimated_prices))
print("R2 Score:", r2_score(test_prices, estimated_prices))

