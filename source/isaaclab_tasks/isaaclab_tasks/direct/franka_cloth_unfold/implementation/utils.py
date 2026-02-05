import numpy as np

def createMesh(interval, npx, npy, f1, f2, f3):
    #se crea a partir de una parametrizacion de la forma (f1(x,y),f2(x,y),f3(x,y))
    #donde (x,y) estan en el rectangulo [ax,bx]x[ay,by] and interval = [ax bx ay by]
    # Allocate space for the nodal coordinates matrix
    X = np.zeros((npx*npy, 3))
    xs = np.linspace(interval[0], interval[1], npx).reshape(-1, 1)
    unos = np.ones((npx, 1))
    # Nodes' coordinates
    yys = np.linspace(interval[2], interval[3], npy)
    for i in range(npy):
        ys = yys[i] * unos
        posi = np.arange((i * npx), ((i + 1) * npx))
        X[posi, :] = np.column_stack((f1(xs, ys), f2(xs, ys), f3(xs, ys)))
    # Elements (quadrilaterals)
    nx = npx - 1
    ny = npy - 1
    T = np.zeros((nx * ny, 4), dtype=int)
    for a in range(1, ny + 1):
        for b in range(1, nx + 1):
            ielem = (a - 1) * nx + b - 1
            inode = (a - 1) * npx + b - 1
            T[ielem, :] = [inode, inode + 1, inode + npx + 1, inode + npx]
    return X, T

def createRectangularMesh(a,b,na,nb,h = 0.5):
    #coordinate function for a flat cloth
    def f1(x, y):
        return x
    def f2(x, y):
        return y
    def f3(x, y):
        return h*np.sqrt(1.5-x**2) #avoid singular case
    #rectangle; a and b are the sides of the rectangle and na nb the number of nodes
    rect = [-a/2, a/2, -b/2, b/2]
    #create the mesh
    X, T = createMesh(rect, na, nb, f1, f2, f3)   
    return X, T  

def createRectangularCurvedMesh(a,b,na,nb):
    #coordinate function for a flat cloth
    def f1(x, y):
        return x
    def f2(x, y):
        return y
    def f3(x, y):
        return 1.5*np.sqrt(4.5-x**2-0*y**2) - 2 #avoid singular case
    #rectangle; a and b are the sides of the rectangle and na nb the number of nodes
    rect = [-a/2, a/2, -b/2, b/2]
    #create the mesh
    X, T = createMesh(rect, na, nb, f1, f2, f3)   
    return X, T  