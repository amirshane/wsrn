'''
Simulation of Wide-Scale Random Noise (Algorithm by Paul Valiant)
Author: Amir Shanehsazzadeh
Date: 5/13/19
'''


import wsrn


# computes and returns averages in every index of list
def list_average(lists):
    
    average = [0 for i in range(len(lists[0]))]
    
    for i in range(len(lists)):
        for j in range(len(lists[0])):
            
            average[j] += lists[i][j] / len(lists)
    
    return average


# runs WSRN-Mutator on hypothesis h until it evolves target h for all errors
def evolve(t, h, n, m, k, r, c, p, iters, distributions,  errors):
        
    gens = [0 for i in range(len(errors))]

    initial_loss = wsrn.expected_loss(t, h, n, iters, distributions, p)
    loss = initial_loss
        
    for j in range(len(errors)):
            
        eps = errors[j]
            
        while(loss > eps):

            gens[j] += 1
            loss, h = wsrn.mutator(t, h, 1, c, iters, eps, distributions, n, m, k, p)
            
        if j+1 < len(gens):
            gens[j+1] += gens[j]
    
    return gens
                

# for given parameters generates targets and hypotheses and evolves them
# repeats samples many times and prints and returns averaged results
def analysis(n, m, k, bounds, c, p, samples):
    
    r = bounds[0]
    
    iters = 1000
    
    distribution = ['normal',  [0, 1], None, None, None]
    distributions = [distribution for i in range(n)]
    
    errors = [0.01, 0.0075, 0.005, 0.0025, 0.001, 0.00075, 0.0005, 0.00025]
    
    generations = [[] for i in range(len(bounds))]
    targets = [[] for i in range(len(bounds))]
    hypotheses = [[] for i in range(len(bounds))]
    
    for i in range(samples):
        
        t = wsrn.uniform_bounded_poly_generator(n, m, k, r)
        h = wsrn.uniform_bounded_poly_generator(n, m, k, r)
                
        for j in range(len(bounds)):
            
            targets[j].append(wsrn.poly_multiply(t, bounds[j]/r))
            hypotheses[j].append(wsrn.poly_multiply(h, bounds[j]/r))
        
    for i in range(len(bounds)):
        
        for j in range(samples):
        
            t = targets[i][j]
            h = hypotheses[i][j]
            
            gens = evolve(t, h, n, m, k, r, c, p, iters, distributions, errors)
            generations[i].append(gens)
            
    generations_ = [list_average(generations[i]) for i in range(len(bounds))]
    
    for i in range(len(bounds)):
        
        print(str([n, m, k, bounds[i], c, p]))
        print(str(generations_[i]))

    return generations_


'''
# low-dimensional iteration of analysis with 10 samples
n = 1
m = 1
k = 1
bounds = [1]
c = 1
p = 2
samples = 10
analysis(n, m, k, bounds, c, p, samples)
'''
