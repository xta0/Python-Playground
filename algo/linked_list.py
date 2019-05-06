class ListNode():
    def __init__(self,val):
        self.val = val
        self.next = None
    
    
def merge_two_sorted_list(l1, l2):
    head = tmp = None
    while l1 and l2:
        node = None
        if l1.val < l2.val:
            node = ListNode(l1.val)
            l1 = l1.next
        else:
            node = ListNode(l2.val)
            l2 = l2.next
        if not head:
            head = node
            tmp = head
        else:
            tmp.next = node
            tmp = tmp.next
        
    #append left
    while l1:
        if not head:
            head = ListNode(l1.val)
            tmp = head
        else:
            tmp.next = ListNode(l1.val)
            tmp = tmp.next
        
        l1 = l1.next
    
    #append right
    while l2:
        if not head:
            head = ListNode(l2.val)
            tmp = head
        else:
            tmp.next = ListNode(l2.val)
            tmp = tmp.next
        l2 = l2.next

    return head


                


def reverse(root):
    