#balanced brackets

def isValid(s):
    stack = []
    for c in s:
        if c=='(':
            stack.append(c)
        elif c==')':
            stack.pop(-1)
        
    return len(stack) == 0


print(isValid("("))
print(isValid("()"))
print(isValid("(())"))
print(isValid("(()"))
print(isValid("()()"))
            
