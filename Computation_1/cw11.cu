#include <stdio.h>
__global__ void removeDiv(int x, int y, int z) {
  // assert(x==y || x==z);
  if (x == y)
    x = z;
  else
    x = y;
  // 1. we can remove the else part
  // can we predicate it?
  bool cond = x == y;
  x = cond * z + (1 - cond) * y;
  // OR
  x = (x == y) ? z : y;
}