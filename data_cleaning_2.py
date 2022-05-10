import pandas as pd

df = pd.read_csv('City.csv')

df.to_csv("city_data_cleaned.csv")
