class SeqItor:
    def __init__(self, seq):
        self.seq = seq
        self.index = 0
    def __next__(self):
        if self.index >= len(self.seq):
            raise StopIteration()
        else:
            item = self.seq[self.index]
            self.index += 1
            return item
    def __iter__(self):
        return self


class MyList:
    def __init__(self, seq):
        self.seq = seq
    def __iter__(self):
        return SeqItor(self.seq)
    


x = MyList([1, 2, 3])
# for n in x:
    # print(n)
it = iter(x)
n = next(it)
print(n)
