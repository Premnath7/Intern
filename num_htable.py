from htable import HashTable
from numpy import random

class NumberHashTable(HashTable):
  """
  A pair-wise independent hash for numbers
  """

  def __init__(self, p):
	
    """
    Initializes a pair-wise independent hash table to store numbers.
    The parameter p is a prime number.  The hash function is:

      h(a, b) (x) =  (ax + b) mod p

    where a is non-zero random number (between 1 and p-1 incl.) 
    and b is a random number between 0 and p-1 incl.  These hash functions
    are pair-wise independent and satisfy useful properties as we will see.

    Remark:  a and b are picked at random only ONCE when the hashtable is initialized and remains fixed when the hash table is used.
    """
    super(HashTable, NumberHashTable).__init__(self)
    self.p = p
    self.a = random.randint(1,p)
    self.b = random.randint(0,p)
    hashtable=[[]for x in range(p)]
    self.hashtable = hashtable

  def insert(self, element):
    index = self.hashf(element)
    if(self.query(element))==False:
      self.hashtable[index].append(element)
      return len(self.hashtable[index]) 	
    return len(self.hashtable[index])

    """
    Inserts an element into the HashTable.  For our purposes, the method should return the number of basic operations performed.
    """
    #raise NotImplementedError("Abstract Base class instantiated.")

  def delete(self, element):
    index = self.hashf(element)
    for x in self.hashtable[index]:
      if (x==element):
        self.hashtable[index].remove(element)
        return len(self.hashtable[index])
    return len(self.hashtable[index])
	
    """
    Deletes an element from the HashTable.  For our purposes, the method should return the number of basic operations performed.
    """
    #raise NotImplementedError("Abstract Base class instantiated.")

  def query(self, element):
    index = self.hashf(element)
    for x in self.hashtable[index]:
      if (x==element):
        return True
    return False
    """
    Checks if an element exists in the HashTable
    """
    #raise NotImplementedError("Abstract Base class instantiated.")

  def hashf(self,v):
    fvalue = (self.a*v+self.b)%self.p
    return fvalue

