import numpy as np
from Utilz import rho, rho_x, rho_xx

# def mesh_less(kappa, prms, mu, g, theta=0.5, M=9, N=10, max_k=20, eps=10):
    
#     # X = np.arange(1/(M+1), 1, 1/(M+1), dtype=np.float64)
#     X = np.linspace(0, 1, M+2)
#     dt = 1/N

#     Phi = np.empty((M, M), dtype=np.float64)
#     Omega = np.empty(Phi.shape, dtype=np.float64)
#     Xi = np.empty(Phi.shape, dtype=np.float64)
    
#     G = np.empty((M,), dtype=np.float64)

#     eta = dt * (theta - 1)
    
#     Lambda = []

#     for i in range(1, M+1):
#         for j in range(1, M+1):
#             Phi[i-1, j-1] = rho(X[i], X[j], eps=eps)
#             Omega[i-1, j-1] = rho(X[i], X[j], eps=eps) + eta*kappa*rho_xx(X[i], X[j], eps=eps) + eta*mu(X[i], dt, prms)*rho_x(X[i], X[j], eps=eps)

#     for i in range(1, M+1):
#         G[i-1] = dt*theta*g(X[i], 0, prms, max_k) - eta*g(X[i], dt, prms, max_k)

#     Lambda.append(np.linalg.solve(Omega, G))
    
#     n = 1

#     while n < N:

#         for i in range(1, M+1):
#             for j in range(1, M+1):
#                 Omega[i-1, j-1] = rho(X[i], X[j], eps=eps) + eta*kappa*rho_xx(X[i], X[j], eps=eps) + eta*mu(X[i], (n+1)*dt, prms)*rho_x(X[i], X[j], eps=eps)
#                 Xi[i-1, j-1] = rho(X[i], X[j], eps=eps) + dt*theta*kappa*rho_xx(X[i], X[j], eps=eps) + dt*theta*mu(X[i], (n)*dt, prms)*rho_x(X[i], X[j], eps=eps)

#         for i in range(1, M+1):
#             G[i-1] = dt*theta*g(X[i], n*dt, prms, max_k) - eta*g(X[i], (n+1)*dt, prms, max_k)

#         Lambda.append(np.linalg.solve(Omega, Xi @ Lambda[-1] + G))

#         n += 1
        
#     res = {'sol':Phi@Lambda[-1],
#            'X': X, 
#            'Lambda': Lambda[-1],
#            'eps': eps}
    
#     return res


def mesh_less(kappa, prms, mu, g, theta=0.5, M=10, N=10, max_k=20, eps=10):
    
    X = np.arange(1/(M+1), 1, 1/(M+1), dtype=np.float64)[:M]
    dt = 1/N

    Phi = np.empty((X.shape[0], X.shape[0]), dtype=np.float64)
    Omega = np.empty(Phi.shape, dtype=np.float64)
    Xi = np.empty(Phi.shape, dtype=np.float64)
    
    G = np.empty(X.shape[0], dtype=np.float64)

    eta = dt * (theta - 1)
    
    Lambda = []

    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            Phi[i, j] = rho(X[i], X[j], eps=eps)
            Omega[i, j] = rho(X[i], X[j], eps=eps) + eta*kappa*rho_xx(X[i], X[j], eps=eps) + eta*mu(X[i], dt, prms)*rho_x(X[i], X[j], eps=eps)

    for i in range(X.shape[0]):
        G[i] = dt*theta*g(X[i], 0, prms, max_k) - eta*g(X[i], dt, prms, max_k)

    Lambda.append(np.linalg.solve(Omega, G))
    
    n = 1

    while n < N:

        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                Omega[i, j] = rho(X[i], X[j], eps=eps) + eta*kappa*rho_xx(X[i], X[j], eps=eps) + eta*mu(X[i], (n+1)*dt, prms)*rho_x(X[i], X[j], eps=eps)
                Xi[i, j] = rho(X[i], X[j], eps=eps) + dt*theta*kappa*rho_xx(X[i], X[j], eps=eps) + dt*theta*mu(X[i], (n)*dt, prms)*rho_x(X[i], X[j], eps=eps)

        for i in range(X.shape[0]):
            G[i] = dt*theta*g(X[i], n*dt, prms, max_k) - eta*g(X[i], (n+1)*dt, prms, max_k)

        Lambda.append(np.linalg.solve(Omega, Xi @ Lambda[-1] + G))

        n += 1
        
    res = {'sol':Phi@Lambda[-1],
           'X': X, 
           'Lambda': Lambda[-1],
           'eps': eps}
    
    return res


def crank_nicolson_fd(kappa, prms, mu, g, theta=0.5, M=50, N=50, max_k=20):
    dx = 1/M
    dt = 1/N

    X = np.linspace(0, 1, M+1)
    t = np.linspace(0, 1, N+1)

    A = np.zeros((M-1, M-1))
    B = np.zeros((M-1, M-1))
    G = np.empty((M-1,))

    nu = dt*(theta-1)
    beta = kappa/dx**2

    F = []

    for n in range(N):
        for i in range(M-1):
            if i+1 < M-1:
                A[i, i+1] = nu*(beta + mu(X[i+1], (n+1)*dt, prms)/(2*dx))
                A[i+1, i] = nu*(beta - mu(X[i+1], (n+1)*dt, prms)/(2*dx))
            A[i, i] = 1 - 2*nu*beta
            
        for i in range(M-1):
            if i+1 < M-1:
                B[i, i+1] = dt*theta*(beta + mu(X[i+1], (n)*dt, prms)/(2*dx))
                B[i+1, i] = dt*theta*(beta - mu(X[i+1], (n)*dt, prms)/(2*dx))
            B[i, i] = 1 - 2*dt*theta*beta
            
        for i in range(1, M):
            G[i-1] = dt*(theta*g(X[i], n*dt, prms, max_k) + (1-theta)*g(X[i], (n+1)*dt, prms, max_k))
        
        if n == 0:
            F.append(np.linalg.inv(A) @ G)
        else:
            F.append(np.linalg.inv(A) @ (B @ F[-1] + G))

    res = {'sol': F[-1],
           'X': X}
    return res

def implicit_fd(kappa, prms, mu, g, M=50, N=50, max_k=20):
    dx = 1/M
    dt = 1/N

    X = np.linspace(0, 1, M+1)
    t = np.linspace(0, 1, N+1)

    A = np.zeros((M-1, M-1))
    G = np.empty((M-1,))

    beta_1 = dt*kappa/dx**2

    F = []

    for n in range(N):
        for i in range(M-1):
            if i+1 < M-1:
                A[i, i+1] = -beta_1
                A[i+1, i] = -beta_1 + dt*mu(X[i+1], (n+1)*dt, prms)/dx
            A[i, i] = 1 + 2*beta_1 - dt*mu(X[i+1], (n+1)*dt, prms)/dx
            
        for i in range(1, M):
            G[i-1] = dt*g(X[i], (n+1)*dt, prms, max_k)
        
        if n == 0:
            F.append(np.linalg.inv(A) @ G)
        else:
            F.append(np.linalg.inv(A) @ (F[-1] + G))

    res = {'sol': F[-1],
           'X': X}
    return res
