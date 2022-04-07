import pandas as pd
df1= pd.read_csv('Cab_Data.csv')
df2= pd.read_csv('Transaction_ID.csv')
df3= pd.read_csv('Customer_ID.csv')

inner = pd.merge(df1, df2) 

new_dataframe = pd.merge(inner, df3)








