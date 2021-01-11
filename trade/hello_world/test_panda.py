import pandas as pd

#Read csv file
df = pd.read_csv("data/aapl.csv")
#Get max price
max_price = df['Close'].max() #
print("max_price: ", max_price);
#Get avg. volume
avg_volume = df['Volume'].mean()
print("avg_volume: ", avg_volume);
#first 5 days
first_5_days = df.head(5);
print("first 5 days: ",first_5_days)
last_5_days = df.tail(5);
print("last 5 days: ",last_5_days)
#range 10-20
rows = df[10:21]
print("rows ",rows)



