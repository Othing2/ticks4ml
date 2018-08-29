#!/usr/bin/python

def qsorted(l):
    """
    quick sorted
    """
    if len(l) <= 1:
        return l
    else:
        base = l[0]
        left = [i for i in l[1:] if i<=base]
        right = [j for j in l[1:] if j>base]
        return qsorted(left)+[base]+qsorted(right)

def msorted(l):
    """
    merge sorted
    """
    def merge(a,b):
        ll =[]
        while a and b:
            if a[0]<b[0]:
                ll.append(a.pop(0))
            else:
                ll.append(b.pop(0))
        return ll+a+b
    
    if len(l)<=1:
        return l
    return merge(msorted(l[:len(l)//2]), msorted(l[len(l)//2:]))

def hsorted(l):
    """
    heap sorted
    """
    from heapq import heappush, heappop
    rs = []
    for i in l:
        heappush(rs, i)
    return [heappop(rs) for _ in range(len(rs))]
    
def kth_max(l, k):
    """
    the k-th max num
    """
    if len(l)<k:
        return -9999999999999
    base = l[0]
    bet = [i for i in l[1:] if i>=base]
    if len(bet) == k-1:
        return base
    elif len(bet)>k-1:
        return kth_max(bet, k)
    else:
        les = [j for j in l[1:] if j<base]
        return kth_max(les, k-len(bet))
        
def k_max(l, k):
    """
    the k max nums
    """
    if len(l)<k:
        return l 
    base = l[0]
    bet = [i for i in l[1:] if i>=base]
    if len(bet) == k:
        return bet
    elif len(bet)>k:
        return k_max(bet, k)
    else:
        les = [j for j in l[1:] if j<base]
        return k_max(les, k-len(bet)) + bet

a = [6,5,4,3,2,1]
print("Quick Sorted:   ", a, '==>', qsorted(a))
#print("Merge Sorted:   ", a, '==>', msorted(a))
#print("Heap Sorted:    ", a, '==>', hsorted(a))
#print("The 2-th Max:   ", a, '==>', kth_max(a, 2))
#print("The 2-max Nums: ", a, '==>', k_max(a, 2))
