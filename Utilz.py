import numpy as np
from scipy import stats
from scipy.special import erfcx

def rho(x, y, eps=10):
    t1 = np.exp(-eps**2 * (x-y)**2, dtype=np.float64)
    t2 = np.exp(-eps**2 * (x**2 + y**2), dtype=np.float64)
    denum = 1 - np.exp(-2*eps**2, dtype=np.float64)
    num_p1 = np.exp(-eps**2 * (1-y)**2, dtype=np.float64) - np.exp(-eps**2 * (1+y**2), dtype=np.float64)
    num_p2 = np.exp(-eps**2 * (x-1)**2, dtype=np.float64) - np.exp(-eps**2 * (1+x**2), dtype=np.float64)
    num = num_p1 * num_p2
    t3 = num / denum
    return t1 - t2 - t3

def rho_x(x, y, eps=10):
    t1 = -2 * eps**2 * (x-y) * np.exp(-eps**2 * (x-y)**2, dtype=np.float64)
    t2 = 2 * eps**2 * x * np.exp(-eps**2 * (y**2 + x**2), dtype=np.float64)
    denum = 1 - np.exp(-2*eps**2, dtype=np.float64)
    num_p1 = np.exp(-eps**2 * (1-y)**2, dtype=np.float64) - np.exp(-eps**2 * (1+y**2), dtype=np.float64)
    num_p2 = -2 * eps**2 * (x-1) * np.exp(-eps**2 * (x-1)**2) + 2 * eps**2 * x * np.exp(-eps**2 * (1+x**2), dtype=np.float64)
    num = num_p1 * num_p2
    t3 = num / denum
    return t1 + t2 - t3

def rho_xx(x, y, eps=10):
    t1 = -2 * eps**2 * np.exp(-eps**2 * (x-y)**2, dtype=np.float64)
    t2 = 4 * eps**4 * (x-y)**2 * np.exp(-eps**2 * (x-y)**2, dtype=np.float64)
    t3 = 2 * eps**2 * np.exp(-eps**2 * (x**2 + y**2))
    t4 = -4 * eps**4 * x**2 * np.exp(-eps**2 * (x**2 + y**2))
    
    denum = 1 - np.exp(-2*eps**2)
    num_p1 = np.exp(-eps**2 * (1-y)**2) - np.exp(-eps**2 * (1 + y**2))
    num_p2_t1 = -2 * eps**2 * np.exp(-eps**2 * (x-1)**2, dtype=np.float64)
    num_p2_t2 = 4 * eps**4 * (x-1)**2 * np.exp(-eps**2 * (x - 1)**2, dtype=np.float64)
    num_p2_t3 = 2 * eps**2 * np.exp(-eps**2 * (1+x**2), dtype=np.float64)
    num_p2_t4 = -4 * eps**4 * x**2 * np.exp(-eps**2 * (1 + x**2), dtype=np.float64)
    num_p2 = num_p2_t1 + num_p2_t2 + num_p2_t3 + num_p2_t4
    num = num_p1 * num_p2
    t5 = num / denum
    return t1 + t2 + t3 + t4 - t5

def M(y):
    return erfcx(y/np.sqrt(2)) / np.sqrt(2) * np.sqrt(np.pi)

def F_v0 (x, t, sigma, v0, max_k=20):
    p = np.exp(-(v0*x)/(2*sigma) - (v0**2*t)/(4*sigma)) #
    s = 0
    for k in range(max_k+1):
        if k % 2 == 0:
            rk = k + x
            t1 = stats.norm.pdf(rk/np.sqrt(2*sigma*t))
        else:
            rk = k + 1 - x
            t1 = - stats.norm.pdf(rk/np.sqrt(2*sigma*t))
        # t1 = (-1)**(k) * stats.norm.pdf(rk/np.sqrt(2*sigma*t))
        t2 = M((rk-v0*t)/np.sqrt(2*sigma*t)) + M((rk+v0*t)/np.sqrt(2*sigma*t))
        s += t1*t2
        
    return p*s

def F_v0_x(x, t, sigma, v0, max_k=20):
    p = 2*np.exp(0.5*v0*(-x - v0*t)/(sigma)) #
    s = 0
    for k in range(max_k+1):
        if k % 2 == 0:
            rk = k + x
            t1 = - stats.norm.pdf(rk/np.sqrt(2*sigma*t))
            t2 = v0/(2*sigma) * M((rk-v0*t)/np.sqrt(2*sigma*t)) + 1/np.sqrt(2*sigma*t)
        else:
            rk = k + 1 - x
            t1 = stats.norm.pdf(rk/np.sqrt(2*sigma*t))
            t2 = v0/(2*sigma) * M((rk+v0*t)/np.sqrt(2*sigma*t)) - 1/np.sqrt(2*sigma*t)
        # t1 = (-1)**(k+1) * stats.norm.pdf(rk/np.sqrt(2*sigma*t))
        # t2 = v0/(2*sigma) * M((rk+v0*t*(-1)**(k+1))/np.sqrt(2*sigma*t)) + (-1)**k/np.sqrt(2*sigma*t)
        s += t1*t2
        
    return p*s

def diff(F, t):
    f = []
    dt = t[1] - t[0]
    for i in range(len(F)-1):
        f.append((F[i+1]-F[i])/dt)
    return f

# def eval_F(sol, x):    
#     F = 0
#     for i in range(1, sol['X'].shape[0]-1):
#         F += sol['Lambda'][i-1] * rho(x, sol['X'][i], sol['eps'])
#     return F

def eval_F(sol, x):    
    F = 0
    for i in range(sol['X'].shape[0]):
        F += sol['Lambda'][i] * rho(x, sol['X'][i], sol['eps'])
    return F

