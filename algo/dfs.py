#permutation

class String:
    def __init__(self,text):
        self.text = text

    #combination
    def dfs2(self,curr,depth,arr,index,choose,result):
        if depth == curr:
            result.append("".join(choose))
            return 
        # print(result)
            # return
        for i in range(index,len(arr)):
            x = arr[i]
            # print("(i,x) ",(i,x))
            choose.append(x)
            # result.append("".join(choose))
            curr += 1
            self.dfs2(curr,depth,arr,i+1,choose,result)
            curr-=1
            choose.pop()
        

    #permutation
    def dfs1(self,arr,choose,result):
        if len(arr) == 0:
            result.append("".join(choose))
            print("result: ",result)
            return

        for i in range(0,len(arr)):
            #choose
            x = arr[i]
            choose.append(x)
            arr.pop(i)
            #dfs
            self.dfs1(arr,choose,result)
            #unchoose
            choose.pop()
            arr.insert(i,x)

    def permutation(self):
        arr = list(self.text)
        choose = []
        result = []
        self.dfs1(arr,choose,result)
        return result

    def combination(self,n):
        arr = list(self.text)
        choose = []
        result = []
        self.dfs2(0,n,arr,0,choose,result)
        return result

#permutation
# print(String("abc").permutation())
#combination
print(String("Google").combination(2))

class Maze:
