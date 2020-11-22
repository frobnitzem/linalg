#!/usr/bin/env python3

import numpy as np
rand = np.random

def main(argv):
    X = rand.random((4,3))-0.5
    Y = rand.random((4,3))-0.5
    Z = np.dot(X, Y.transpose())
    print( gen_tile('X', X, 4) )
    print( gen_tile('Y', Y, 3) )
    print( gen_tile('Z', Z, 4) )

# generate C code for creating a CPU tile corresponding to this matrix
def gen_tile(v, x, stride=None):
    if stride is None:
        stride = x.shape[0]
    data_name = "%s_data"%v
    s = gen_mat(data_name, x, stride)
    s += "auto %s = std::make_shared<Linalg::Tile<T> >(%d, %d, %d, %s, Linalg::Place::Host);\n"%(
            v,x.shape[0],x.shape[1],stride,data_name)
    return s

# generate C code for declaring the matrix x as variable v
def gen_mat(v, x, stride, nrow=6):
    m, n = x.shape
    lst = []
    for j in range(n):
        lst.extend(x[i,j] for i in range(m))
        lst.extend(0.0 for i in range(m,stride))
    s = "double %s[%d] = {"%(v,stride*n)
    newl = ',\n' + ' '*len(s)
    s += newl.join(
        ', '.join('%f'%lst[j] for j in range(i,min(i+nrow,len(lst))))
           for i in range(0,len(lst),nrow))
    return s + '}\n'

if __name__=="__main__":
    import sys
    main(sys.argv)
