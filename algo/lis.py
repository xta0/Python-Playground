import random

class Solution:
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        if n == 0:
            return 0
        if n == 1:
            return 1
        
        l = [0]*n;
        l[0]=1
        
        for i in range(1,n):
            tmp = 0
            for j in range(0,i):
                if nums[j] < nums[i] and l[j] > tmp:
                    tmp = l[j]
            
            l[i] = tmp + 1
            
        
        return max(l)


# array = random.sample(range(-10,10),10)
array = [5,7,4,-3,9,1,10,4,5,8,9,3]
s = Solution()
print(s.lengthOfLIS(array))
# print(array)