import pandas as pd
import plot
%matplotlib inline
def test_run():
    df = pd.read_csv("data/aapl.csv")
    
    df['Close'].plot()
    plt.show() 

    # print (df['High'])
    # df['High'].plot()
    # plt.show() 


if __name__ == "__main__":
    test_run()
