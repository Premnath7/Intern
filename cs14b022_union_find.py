
class Node:

  def __init__(self, ops):
    self.parent = self
    self.ops = ops
    self.rank=0


  def get_root(self):
    self.ops.increment()

    # Implement path compression
    if (self.parent != self):
      self.parent = self.parent.get_root()
    return self.parent


  def union(self, other):
    self.ops.increment()
    xr = self.get_root()
    yr = other.get_root()
    if xr == yr :
       return
    if xr.rank < yr.rank :
         xr.parent = yr
    elif xr.rank > yr.rank :
         yr.parent = xr
    else :
         yr.parent = xr
         xr.rank = xr.rank + 1


    # Implement rank based union
   
