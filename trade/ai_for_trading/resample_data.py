import numpy as np
import pandas as pd

dates = pd.date_range('10/10/2018', periods=11, freq='D')
close_prices = np.arange(len(dates))

close = pd.Series(close_prices, dates)
print(close)
# Let's say we want to bucket these days into 3 day periods.
# This returns a DatetimeIndexResampler object. 
close.resample('3D')
close.resample('3D').first()
# So, why use the resample function instead of .iloc[::3] or the groupby function?
# The resample function shines when handling time and/or date specific tasks. In fact, # you can't use this function if the index isn't a time-related class.

pd.DataFrame({
    'days': close,
    'weeks': close.resample('W').first()})

def days_to_weeks(open_prices, high_prices, low_prices, close_prices):
    """Converts daily OHLC prices to weekly OHLC prices.
    
    Parameters
    ----------
    open_prices : DataFrame
        Daily open prices for each ticker and date
    high_prices : DataFrame
        Daily high prices for each ticker and date
    low_prices : DataFrame
        Daily low prices for each ticker and date
    close_prices : DataFrame
        Daily close prices for each ticker and date

    Returns
    -------
    open_prices_weekly : DataFrame
        Weekly open prices for each ticker and date
    high_prices_weekly : DataFrame
        Weekly high prices for each ticker and date
    low_prices_weekly : DataFrame
        Weekly low prices for each ticker and date
    close_prices_weekly : DataFrame
        Weekly close prices for each ticker and date
    """
    
    open_prices_weekly = open_prices.resample('W').first()
    high_prices_weekly = high_prices.resample('W').max()
    low_prices_weekly = low_prices.resample('W').min()
    close_prices_weekly = close_prices.resample('W').last()
    
    return open_prices_weekly, high_prices_weekly, low_prices_weekly, close_prices_weekly
