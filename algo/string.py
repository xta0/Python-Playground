#- Given a String find all the k long substrings which contains k different characters.

#brutal force O(n^2)
def findkstrings(text, k):
    if not text or len(text) == 0 or k <= 0 or k>len(text) :
        return []

    choosen = []
    result = []
    for i in range(0,len(text)-k+1):
        for j in range(i,i+k):
            x = text[j]
            if not (x in choosen):
                choosen.append(x)

        if len(choosen) == k:
            result.append("".join(choosen))
        choosen.clear()

    return result



print(findkstrings("aba",2))
    
