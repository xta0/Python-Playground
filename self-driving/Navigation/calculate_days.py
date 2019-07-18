import calendar 

def is_leap_year(year):
    return  calendar.isleap(year)

def is_31days(month):
    return month in {1,3,5,7,8,10,12}
    
def total_days(y,m,d):
    days = 0
    for _ in range(0,y):
        days+=365
    
    for month in range(0,m):
        if month == 1:
            if is_leap_year(y+1):
                days+=29
            else:
                days+=28
        else:
            if is_31days(month+1):
                days+=31
            else:
                days+=30
    
    days += d

    return days

def days_between_dates(y1, m1, d1, y2, m2, d2):
    """
    Calculates the number of days between two dates.
    """       
    return total_days(y2,m2,d2) - total_days(y1,m1,d1)

def test_days_between_dates():
    
    # test same day
    assert(days_between_dates(2017, 12, 30,
                              2017, 12, 30) == 0)
    # test adjacent days
    assert(days_between_dates(2017, 12, 30, 
                              2017, 12, 31) == 1)
    # test new year
    assert(days_between_dates(2017, 12, 30, 
                               2018, 1,  1)  == 2)
    # test full year difference
    assert(days_between_dates(2012, 6, 29,
                              2013, 6, 29)  == 365)
    
    print("Congratulations! Your days_between_dates")
    print("function is working correctly!")
    
test_days_between_dates()
print(days_between_dates(1985,10,22,2018,9,5))

