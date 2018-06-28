import warnings

import numpy as np

from scipy import stats
from pykdtree.kdtree import KDTree

import matplotlib.pyplot as plt
from tqdm import tqdm_notebook


class SpatialDistribution():
    ''' Uses Monte Carlo simulations to test whether
    two spatial datasets are distributed evenly.
    
    Source:
    https://gis.stackexchange.com/questions/4484/comparing-two-spatial-point-patterns
    '''
    
    def __init__(self):
        return None
    
    def fit(self, a, b):
        
        # build dataset from input
        self.x = np.vstack([a, b])
        self.y = np.hstack([np.ones(a.shape[0]), np.zeros(b.shape[0])])
        
        # size ratio of inputted datasets
        self.frac = np.mean(self.y)
        
        # calculate mean distance over input dataset
        self.orig_dist = np.mean(
            KDTree(self.x[self.y == 1]).query(self.x[self.y == 0])[0])
        
        return None
        
    def simulate(self, n_iter):
        
        # run n_iter simulations
        self.sim_dists = []
        for _ in tqdm_notebook(range(n_iter)):
            
            # rearrange labels
            self.y = (np.random.rand(self.x.shape[0]) < self.frac).astype(int)
            
            # calculate mean nearest neigbor distance
            self.sim_dists.append(
                np.mean(KDTree(self.x[self.y == 1]).query(self.x[self.y == 0])[0]))
        
        # raise warning if simulated distances not normally distributed
        z, p = stats.mstats.normaltest(self.sim_dists, axis=0)
        if p > 0.05:
            warnings.warn('Simulated distances are not normally distributed ' + \
                          '(p={:.4f}). Run more simulation iterations.'.format(p))

        return self.orig_dist, self.sim_dists
    
    def summary(self):
        
        # calculate hypothesis test statistics
        mu, sigma = stats.norm.fit(self.sim_dists)
        z = (self.orig_dist - mu) / sigma
        p = stats.norm.sf(abs(z)) * 2
        
        # plot summary histogram with hypothesis test statistics
        plt.hist(self.sim_dists, bins=30, color='b', edgecolor='k')
        plt.axvline(self.orig_dist, color='red', linewidth=2)
        plt.title('Z-score: {:.4f} - p-value: {:.4f}'.format(z, p), loc='right')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.show()
        
        return None
   
