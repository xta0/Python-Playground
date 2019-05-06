class PT:
    prev = None
    def __init__(self,i,j):
        self.i = i
        self.j = j 
    def children(self,w,h):
        ret = []
        if self.j-1>=0:
            pt = PT(self.i,self.j-1)
            pt.prev = self
            ret.append(pt)
        if self.j+1<w:
            pt = PT(self.i,self.j+1)
            pt.prev = self
            ret.append(pt)
        if self.i-1>=0:
            pt = PT(self.i-1,self.j)
            pt.prev = self
            ret.append(pt)
        if self.i+1<h:
            pt = PT(self.i+1,self.j)
            pt.prev = self
            ret.append(pt)
        return ret
    def __eq__(self,other):
        return self.i == other.i and self.j == other.j
    def __str__(self):
        return f"({self.i},{self.j})"


#geeksforgeeks:https://www.geeksforgeeks.org/shortest-path-in-a-binary-maze/
def bfs(w,h,matrix,marks,start,dst):
    marks[start.i][start.j] = 1
    queue = []
    queue.append(start)
    while(len(queue)):
        sz = len(queue)
        for _ in range(0,sz):
            pt = queue[0]
            queue.pop(0)
            if pt == dst:
                return pt
            marks[pt.i][pt.j] = 1
            for c in pt.children(w,h):
                if matrix[c.i][c.j] == 1 and marks[c.i][c.j] == 0:
                    queue.append(c)
                    
    return None


def pathToDst(matrix,start,dst):
    w = len(matrix)
    h = len(matrix[0])
    marks = []
    for _ in range(0,h):
        marks.append([0]*w)

    pt = bfs(w,h,matrix,marks,start,dst)
    while pt:
        print(pt)
        pt = pt.prev
    
    
matrix =   [[1, 0, 1, 1, 1, 1, 0, 1, 1, 1 ],
            [1, 0, 1, 0, 1, 1, 1, 0, 1, 1 ],
            [1, 1, 1, 0, 1, 1, 0, 1, 0, 1 ],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 1 ],
            [1, 1, 1, 0, 1, 1, 1, 0, 1, 0 ],
            [1, 0, 1, 1, 1, 1, 0, 1, 0, 0 ],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1 ],
            [1, 0, 1, 1, 1, 1, 0, 1, 1, 1 ],
            [1, 1, 0, 0, 0, 0, 1, 0, 0, 1 ]]
start = PT(0,0)
dst = PT(3,4)
pathToDst(matrix,start,dst)
