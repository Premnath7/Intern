def table(str):
  k=len(str)
  T=[]
  T.append(-1)
  T.append(0)
  p=2
  c=0
  while p<k:
    if(str[p-1]==str[c]):
      T.append(c+1)
      c=c+1
      p=p+1
    elif c>0:
      c=T[c]
    else:
      T.append(0)
      p=p+1
  return T


def KMPMatch(A,B):
  
  a=len(A)
  b=len(B)
  c=0
  i=0
  T = table(B)
  while i+c<a:
    if(A[c+i]==B[i]):
      if (i==b-1):
        return c
      i=i+1
    else:
      if(T[i]>-1):
        i=T[i]
        c=c+i-T[i]
      else:
        i=0
        c=c+1
  return -1
A=raw_input("Source String")
B=raw_input("Search String")
print(KMPMatch(A,B))
