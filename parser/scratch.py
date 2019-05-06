import re

# find
def myfirst_yoursecond(p,q):
    myfirst = p[0:p.find(" ") ]
    yourSecond = q[q.find(" ")+1:]
    return myfirst == yourSecond

print(myfirst_yoursecond("c b", "b a"))

# split
str1 = "abc102#)*gdf"
print(str1.split('b'))
l = re.compile(r"[a-zA-Z]+").split(str1);
print(l)

#find all

#match function
reg1 = r'[a-z]+\( *[0-9]+ *\)'
str1 = "cos(1)"
print(re.findall(reg1,str1))

#match "I say, \"hello\""
reg2 = r'(?:[^\\]|(?:\\.))*'
str2 = "I say, \"hello\""
print(re.findall(reg2, str2))

def sumnums(sentence): 
    return sum([int(num) for num in re.findall(r'[0-9]+',sentence)])






