class TreeNode:
    def __init__(self,x):
        self.val = x
        self.left = None
        self.right = None

def constructBST(root,val):
    if root.val == val:
        return 

    if val < root.val:
        if not root.left:
            root.left = TreeNode(val)
        else:
            constructBST(root.left,val)
    else:
        if not root.right:
            root.right = TreeNode(val)
        else:
            constructBST(root.right,val)

def inorderTraversal(root):
    if not root:
        return
    inorderTraversal(root.left)
    print(root.val)
    inorderTraversal(root.right)

def search(root,val,path):
    path.append(root.val)
    if val < root.val:
        search(root.left,val,path)
    elif val > root.val:
        search(root.right,val,path)
    else:
        return

def distanceBetweenTwoNode(arr,p,q):
    if len(arr) == 0:
        return 0
    if p == q:
        return 0

    #construct BST
    root = TreeNode(arr[0])
    for i in range(1,len(arr)):
        constructBST(root,arr[i])

    arr1 = []
    arr2 = []
    search(root,p,arr1)
    search(root,q,arr2)

    index = 0
    while index<len(arr1) and index<len(arr2) and arr1[index] == arr2[index]:
        index += 1
    
    l1 = len(arr1)-index
    l2 = len(arr2)-index

    return l1+l2

    
arr = [6,2,8,0,4,7,9,3,5]
n = distanceBetweenTwoNode(arr,5,7)
print(n)




