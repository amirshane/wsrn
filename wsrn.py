'''
Simulation of Wide-Scale Random Noise [Helper Functions] (Algorithm by Paul Valiant)
Author: Amir Shanehsazzadeh
Date: 5/13/19
'''

import copy
import random

import numpy as np


import poly_coeffs as pc


# generates uniform variables for polynomial coefficients
def coeffs_generator(dim, bound):
    
    coeffs = []
    
    for i in range(dim):
        coeff = np.random.uniform(-bound, bound)
        coeffs.append(coeff)
        
    return coeffs


# generates a polynomial (list rep) from R^n to R^m with degree at most k and with coeffs
def poly_generator(n, m, k, coeffs):
    
    poly = []
    sols = pc.sols(n, k)
    dim = len(sols)
    
    for i in range(m):
        poly_i = []
        for j in range(i * dim, (i+1) * dim):
            term = [coeffs[j]] + [sols[j-i*dim]]
            poly_i.append(term)
        poly.append(poly_i)
        
    return poly


# generates polynomial (list rep) as above but generates uniform variables for coeffs
def uniform_bounded_poly_generator(n, m, k, bound):
    
    poly = []
    sols = pc.sols(n, k)
    dim = len(sols)
    dim_ = dim * m
    coeffs = coeffs_generator(dim_, bound)
    
    for i in range(m):
        poly_i = []
        for j in range(i * dim, (i+1) * dim):
            if j < dim_:
                term = [coeffs[j]] + [sols[j-i*dim]]
                poly_i.append(term)
        poly.append(poly_i)
        
    return poly

# multiplies polynomial coefficients by a constant
def poly_multiply(poly, const):

    poly_ = copy.deepcopy(poly)
    
    for i in range(len(poly_)):
        for j in range(len(poly_[i])):
            poly_[i][j][0] *= const

    return poly_


# returns polynomial in simplified list format
def easy_rep(poly):

    coeffs = []
    degrees = []
    m = len(poly)
    
    for i in range(len(poly[0])):
        i_coeffs = []
        degrees.append(poly[0][i][1])
        
        for j in range(m):
            i_coeffs.append(poly[j][i][0])
        
        coeffs.append(i_coeffs)
    
    return coeffs, degrees


# computes an individaul term: c(x_1^(a_1))* ... *(x_n^(a_n))
def prod(term, inputs):

    coeff = term[0]
    degs = term[1]
    output = 0
    
    for i in range(len(inputs)):
        output += np.power(inputs[i], degs[i])
    output *= coeff

    return output


# evaluates a polynomial from R^n to R
def single_poly_eval(poly, inputs):
    
    output = 0
    
    for term in poly:
        output += prod(term, inputs)

    return output
    
# evaluates polynomial from R^n to R^m
def poly_eval(poly, inputs):

    outputs = []
    m = len(poly)
    
    for i in range(m):
        y = single_poly_eval(poly[i], inputs)
        outputs.append(y)

    return outputs


# returns L^p loss between two vectors
def norm(vec1, vec2, p):
    
    c_norm = 0
    
    for i in range(len(vec1)):
        c_norm += np.power(np.abs(vec1[i]-vec2[i]), p)
    
    return c_norm
       
 
# generates inputs from distributions
# acceptable distributions include: 'normal', 'uniform', 'exponential', 'gamma', 'beta'
# for 'normal' include mean and variance (must be non-negative) as params
# for 'uniform' include lowwer and upper ends as params
# for 'exponential' include scale (must be positive)
# for 'gamma' include shape and scale (both positive)
# for 'beta' include alpha and beta (both positive) as params
# for 'weibull' include scale (must be positive) and shape (must be non-negative)
# can also include multiplicative and additive scales
# can also make distribution symmetric about mean by multiplying by Rademacher variable
def inputs_generator(distributions):
    
    num_vars = len(distributions)
    vars = []
    
    for i in range(num_vars):
        
        dist = distributions[i]
        dist_name = dist[0]
        dist_params = dist[1]
        
        dist_mult_scale = 1 if dist[2] == None else dist[2]
        dist_add_scale = 0 if dist[3] == None else dist[3]
        dist_symmetric = False if dist[4] == None else dist[4]
        
        if dist_name == 'normal':
            X = np.random.normal(dist_params[0], dist_params[1])
    
        elif dist_name == 'uniform':
            X = np.random.uniform(dist_params[0], dist_params[1])
  
        elif dist_name == 'exponential':
            X = np.random.exponential(dist_params[0])
            
        elif dist_name == 'gamma':
            X = np.random.gamma(dist_params[0], dist_params[1])
        
        elif dist_name == 'beta':
            X = np.random.beta(dist_params[0], dist_params[1])
  
        elif dist_name == 'weibull':
            X = np.random.exponential(dist_params[0])
            X = np.power(X, dist_params[1])
            
        else:
            X = np.random.standard_normal()
        
        X *= dist_mult_scale
        X += dist_add_scale
            
        if dist_symmetric == True:
            X = X * (2 * np.random.binomial(1, 0.5) - 1)
                
        vars.append(X)

    return vars
        

# computes empirical expected loss between two polynomials
def expected_loss(poly1, poly2, n, iters, distributions, c_norm = 2):
    
    loss = 0

    for i in range(iters):
        inputs = inputs_generator(distributions)

        X, Y = poly_eval(poly1, inputs), poly_eval(poly2, inputs)

        l = norm(X, Y, c_norm)
        loss += l
        
    loss /= iters
    
    return loss


# computes bounds for wsrn (see paper)
def bounds(const, eps, dim):
    
    rad = np.divide(eps, 6 * const * np.sqrt(dim))
    essential = np.divide(rad * eps, 6 * const)
    lower_bound = essential * (eps**4)
    upper_bound = essential / (eps**4)
    
    return (lower_bound, upper_bound)


# wide-scale random noise step (see paper)
# generates random variable rho uniformly from range (log_2(lower_bound, log_2(upper_bound))
# generates random point vec on dim-dimensional-unit sphere using iid normals
# scales point vec by 2^rho and returns
def wsrn(lower_bound, upper_bound, dim):
    
    rho = np.random.uniform(np.log2(lower_bound), np.log2(upper_bound))
    scale = np.power(2, rho)
    vec = []
    sum_of_squares = 0
    
    for i in range(dim):
        
        Z = np.random.normal()
        vec.append(Z)
        sum_of_squares += np.square(Z)
    
    vec = np.divide(np.asarray(vec), (np.sqrt(sum_of_squares)/scale))
    
    return vec
 

# mutator (see paper)
# computes const for bounds on WSRN
# computes threshold according to paper
# runs WSRN and adds output to polynomial coefficients
# applies selection rule and returns new loss and new polynomial appropriately
def mutator(target_poly, poly, gens, children, iters, eps, distributions, n, m, k, p = 2):
    
    for _ in range(gens):
       
        dim = len(poly[0])
        dim_ = m * dim
        
        const = 0
    
        for i in range(m):
            
            target_poly_i = target_poly[i]
            poly_i = poly[i]
            
            for j in range(len(target_poly_i)):
                const += np.power((np.abs(target_poly_i[j][0] - poly_i[j][0])), p)
    
        threshold = np.power(eps, 2)/(18 * const * k)
        
        lower_bound, upper_bound = bounds(const, eps, dim_)
        
        bene = []
        neut = [poly]
        
        for l in range(children):
            
            updates = wsrn(lower_bound, upper_bound, dim_)
        
            new_poly = copy.deepcopy(poly)
                    
            for v in range(m):
                for w in range(v*dim, (v+1)*dim):
                    if w < (v+1) * dim:
                        new_poly[v][w-v*dim][0] += updates[w]
    
            loss_1 = expected_loss(target_poly, new_poly, n, iters, distributions, p)
            loss_2 = expected_loss(target_poly, poly, n, iters, distributions, p)
            loss_delta = loss_1 - loss_2
            
            if loss_delta <= -threshold:
                bene.append(new_poly)
                
            elif loss_delta > -threshold and loss_delta < threshold:
                neut.append(new_poly)
                
        if len(bene) != 0:
            poly = random.choice(bene)
            
        else:
            poly = random.choice(neut)
    
    final_loss = expected_loss(target_poly, poly, n, iters, distributions, p)
    
    return final_loss, poly
