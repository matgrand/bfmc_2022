from cmath import sin
import numpy as np
import matplotlib.pyplot as plt
PI = np.pi

R = 1.03     # big radius
r = 0.65    # small radius
L = 3.0    # horizontal distance

d = 0.01    # distance between points

m = 0.25     # border from the edge of the map

assert r < R
assert (R + L + r) <= 6.0 - 2*m 
assert (2*R + 0.4) <= 3.0 - 2*m

if __name__ == '__main__':
    CR = PI * R
    cr =  PI * r * 0.5

    # big arc
    th = np.linspace(0, -PI, round((PI*R/d)))
    g12 = np.array([1.5 + R*np.cos(th) , m+R + R*np.sin(th)])[:,:-1]
    print(f'g12 shape = {g12.shape}')

    g23x = (1.5 - R)*np.ones(round(L/d))
    g23y = np.linspace(m+R, m+R+L, round(L/d))
    g23 = np.vstack((g23x, g23y))[:,:-1]
    print(f'g23 shape = {g23.shape}') 

    th = np.linspace(-PI, -1.5*PI, round((0.5*PI*r/d)))
    g34x = (1.5-R+r) + r*np.cos(th)
    g34y = (R+m+L) + r*np.sin(th)
    g34 = np.vstack((g34x, g34y))[:,:-1]
    print(f'g34 shape = {g34.shape}') 

    g45x = np.linspace(1.5-R+r, 1.5+R-r, round((2*(R-r))/d))
    g45y = (R+L+m+r)*np.ones(round((2*(R-r))/d))
    g45 = np.vstack((g45x, g45y))[:,:-1]
    print(f'g45 shape = {g45.shape}') 

    th = np.linspace(-1.5*PI, -2*PI, round((0.5*PI*r/d)))
    g56x = (1.5+R-r) + r*np.cos(th)
    g56y = (R+m+L) + r*np.sin(th)
    g56 = np.vstack((g56x, g56y))[:,:-1]
    print(f'g56 shape = {g56.shape}') 

    g61x = (1.5 + R)*np.ones(round(L/d))
    g61y = np.linspace(m+R+L, m+R, round(L/d))
    g61 = np.vstack((g61x, g61y))[:,:-1]
    print(f'g61 shape = {g61.shape}') 

    g = np.hstack((g12,g23,g34,g45,g56,g61)).T
    print(f'g shape = {g.shape}') 

    diff = np.linalg.norm(g[1:]-g[:-1], axis=1)
    print(f'diff = {diff}')

    

    #plots
    g = g.T
    plt.plot(g[0],g[1])
    # plt.plot(diff)
    plt.axis('equal')
    # plt.ylim([-0.01,0.02])
    plt.show()


    #save the path
    np.save('sparcs_path.npy',g)
    print('path saved... exiting.')


