import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sparse_la

def cconj(A):
    return np.conj(A.T)

def c_distance(x1, x2):
    #  compute the distance between complex signals
    
    x1 = x1.reshape([-1,1])
    x2 = x2.reshape([-1,1])
    return np.linalg.norm(x1 - np.exp(-1j * np.angle(cconj(x1)@x2))*x2)

def modify_f(f, q):
    return f * (np.linalg.norm(f) ** q)

def make_rot_invariant_exp_family_ensemble(m, n, q):
    # creates complex matrices from the rotationally invariant exponential
    # family which is detailed in the manuscript
    
    coefs = np.random.normal(size=[2*n*n,m]) 
    coefs /= n # dividing prevents overflows. accounted for when computing a
    coefs *= np.linalg.norm(coefs, axis=0)**q
    
    ensemble = ((coefs[:n**2] + 1j*coefs[n**2:]).T).reshape([m,n,n])
        
    coefs = np.sum(coefs**2, axis=1)
    kappa_vals = (1/(coefs/m))**0.5
    
    return ensemble *(np.mean(kappa_vals))

def make_x(n):
    # creates a random  complex signal to recover
    
    x = np.random.random(size=(n,1, 2)).astype('complex128')
    x[:,:,1] *= 1j
    x = np.sum(x, axis=2)

#    x /= np.linalg.norm(x) # uncomment this line to have unit norm x
    
    return x

def get_measurements(ensemble, x):
    # gets quadratic measurements
    
    y = np.zeros(shape=[ensemble.shape[0], 1]).astype('complex128')
    for i in range(ensemble.shape[0]):
        y[i] = cconj(x) @ ensemble[i] @ x
    
    return y

def get_initialisation(ensemble, y):
    # compute spectral initialization
    # the leading left singular vector is used, however, the leading right
    # singular vector can also be used
    # the chosen singular vector is scaled by an estimate of the signal norm
    
    S = 0
    for i in range(ensemble.shape[0]):
        S += np.conj(y[i]) * ensemble[i]
    
    u, _, _ = sparse_la.svds(S/ensemble.shape[0], k=1)
    
    return ((np.mean(np.conj(y)*y)*0.5)**0.25) * u

def loss_grad(x_t, ensemble, y):
    # computes the gradient
    
    grad = 0
    for i in range(ensemble.shape[0]):        
        grad += (np.conj(y[i]) - x_t.T @ np.conj(ensemble[i]) @ np.conj(x_t)) * (-1* ensemble[i].T @ np.conj(x_t)) + (y[i] - cconj(x_t) @ ensemble[i] @ x_t) * (-1* np.conj(ensemble[i]) @ np.conj(x_t)) 
        
    return np.conj(grad)/ensemble.shape[0]

def grad_descent(n, m, ensemble, y, x_t, x_true):
    lr = 1e-1 # step size
    norm_estimate = np.real(((np.mean(np.conj(y)*y)*0.5)**0.25))
    lr /= norm_estimate**2 # scale the step size by the inverse of the norm squared as in the mathematical proofs
    
    diff = 1e6 # arbitrary initial distance to between consecutive iterates
    iterations = 0 # iterations counter
    max_iterations = 2500 # maximum gradient descent iterations
    diff_tol = 1e-6 # maximum allowable distance between consecutive iterates
    
    # gradient descent
    while (diff>diff_tol and iterations<max_iterations):
        grad = loss_grad(x_t, ensemble, y)
        x_t1 = x_t.copy()
        x_t = x_t - lr * grad
        diff = c_distance(x_t, x_t1) / norm_estimate
        iterations +=1

    # calculate error and print information
    error = c_distance(x_t, x_true)/np.linalg.norm(x_true)
    print ('Final error: ', error)
    print ('Iterations: ' + str(iterations) + ' - lr: ' + str(lr))
    
    return error, iterations, x_t
    
def solve(n, m, q, trials):
    # store the error and number of iterations from each trial
    errors = np.zeros(trials)
    iterations = np.zeros(trials)
    
    for i in range(trials):
        print ('Trial: ' + str(i+1))
        
        # create ensemble of measurement matrices and signal to recover, x
        ensemble = make_rot_invariant_exp_family_ensemble(m, n, q)
        x_true = make_x(n)
        
        # get measurements from the ensemble and signal
        y = get_measurements(ensemble, x_true)
        
        # get spectral initalization and calculate distance from optimum
        x_0 = get_initialisation(ensemble, y)
        print ('Initial distance: ', c_distance(x_0, x_true)/np.linalg.norm(x_true))
        
        # run gradient descent
        error, iterations_taken, recovered = grad_descent(n, m, ensemble, y, x_0, x_true)
        
        # save error and number of iterations taken
        errors[i] = error
        iterations[i] = iterations_taken
        
    # print mean iteration and error information
    print ('Mean iterations: ' + str(np.mean(iterations)))
    print ('Mean error: ' + str(np.mean(errors)))
    
    # plot and view errors
    plt.figure()
    plt.plot(errors)
    plt.title('trials=' +str(trials) +  ' - n=' + str(n) + ' - m=' + str(m))
    plt.show()

###############################################################################
if __name__ == "__main__":
    # the dimension of the signal to recover
    n = 100
    
    # the number of measurements
    m = 400
    
    # measurement matrices distribution family parameter. q=0 gives the
    # complex Gaussian measurement model
    q = 0

    # number of trials to try the algorithm
    trials = 100
    
    # solve using spectral initialization and gradient descent
    solve(n, m, q, trials)