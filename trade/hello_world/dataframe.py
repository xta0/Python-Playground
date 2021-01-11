import pandas as pd
import matplotlib.pyplot as plt

def test_run():
    start_date = '2017-03-10'
    end_date = '2018-03-30'
    dates = pd.date_range(start_date, end_date)
    # <class 'pandas.core.indexes.datetimes.DatetimeIndex'>
    # 2010-01-22 00:00:00
    #empty dataframe
    #index = date
    df1 = pd.DataFrame(index=dates)
    #指定主键
    dfSPY = pd.read_csv("./data/SPY.csv",
                        index_col="Date",
                        parse_dates=True,
                        usecols=['Date','Adj Close'],
                        na_values=['nan'])
    #Rename Column name
    dfSPY = dfSPY.rename(columns={'Adj Close':'SPY'})
    
    #Inner Join SP&500
    df1 = df1.join(dfSPY,how="inner");

    #Read symbols in ./data
    symbols = ['GOOG','FB','AAPL'];
    for name in symbols:
        df_tmp = pd.read_csv(f"./data/{name}.csv",
                         index_col="Date",
                         parse_dates=True,
                         usecols=['Date','Adj Close'],
                         na_values=['nan']);

        df_tmp = df_tmp.rename(columns={'Adj Close':name})
        df1 = df1.join(df_tmp) #how="left";
        
        #normalized data
        df1 = df1 / df1.ix[0,:]

        # ax = df1.plot(title='Stock Prices', fontsize=6)
        # ax.set_xlabel("Date")
        # ax.set_ylabel("Price")    
        # plt.show()

        ## slice
        # slice by row range(dates) using Datafrome.ix[] 
        print(df1.ix['2018-03-01':'2018-03-06']) #first 6 days
        print(type(df1.values))
        # row slice
        # print(df1['GOOG'])
        # print(df1[['GOOG','FB']])
        # print(df1.ix['2018-03-01':'2018-03-06',['SPY','GOOG']])

if __name__ == '__main__':
    test_run()