
count = 0
def move(n,src,mid,dst):
    
    global count
    count = count+1
    if n == 1:
        print("from",src,"to",dst)
        return
    else:
        move(n-1,src,dst,mid)
        print("from",src,"to",dst)
        move(n-1,mid,src,dst)
    
move(3,'A','B','C')
print("Count:",count)

#using stack
class Problem:
    def __init__(self,n,src,mid,dst):
        self.n = n
        self.start = src
        self.mid = mid
        self.end = dst

stack = list()
stack.append(Problem(3,'A','B','C'))
while( len(stack) != 0 ):
    problem = stack.pop()
    if(problem.n == 1):
        print("from",problem.start,"to",problem.end)
    else:
        #2 B A C
        stack.append(Problem(problem.n-1,problem.mid,problem.start,problem.end))
        #1 A B C
        stack.append(Problem(1,problem.start,problem.mid,problem.end))
        #2 A C B
        stack.append(Problem(problem.n-1,problem.start,problem.end,problem.mid))
