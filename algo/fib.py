#Naive recurrsion
def fib1(n):
    if n==0 or n==1:
        return n
    else:
        return fib1(n-1)+fib1(n-2)

#Recursion + Memoization
def fib2(n,memo):
    if n in memo:
        return memo[n]
    else:
        if n==0 or n==1:
            return n
        else:
            v = fib2(n-1,memo)+fib2(n-2,memo)
            memo[n] = v
            return v

#DP
def fib3(n):
    fib ={} 
    fib[0] = 0
    fib[1] = 1
    for i in range(2,n+1):
        fib[i] = fib[i-1]+fib[i-2]

    return fib[n]

print(fib1(30))
print(fib2(30,{}))
print(fib3(30))
