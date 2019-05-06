
import math

books = [100, 200, 300, 400, 500, 600, 700, 800, 900]  # 书页数
k = 3  # 抄写员人数

def check(x):
    cur_pages = 0
    piles = 0
    for i in reversed(books):
        if (cur_pages + i > x):
            piles += 1
            cur_pages = i
        else:
            cur_pages += i
    if cur_pages > 0:
        piles += 1

    return  piles<=k


#二分查找
l = max(books)
r = sum(books)
while(l <= r):
    mid = math.ceil((l+r)/2)
    result = check(mid)
    if(result):
        r = mid-1
    else:
        l = mid + 1

print(l, r)
