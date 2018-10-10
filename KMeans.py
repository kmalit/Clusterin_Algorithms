# K - MEANS CLUSTERING
# K - MEANS CLUSTERING
'''
This is conceptually simple, easy to implement but expressively limited
It works with arbitrary distance measures 
The goal is to:
    1. Divide the data into clusters
    2. Find representatives for each of the K clusters

The algorithm:
input instances X1:n
Initialize K centres C1:k
Repeat until convergence:
    # Assignment phase
    for i in 1:k
        Ci = {x:x in D ^ ci = argmin C (dist(x,c))}
    end
    # maximization phase
    for i in 1:k
        centres = sum(x in Ci)/size of Ci
    end

return: C i:k and centeres i:k
'''

class K_means:
    
    # Check if packages are installed
    import importlib.util
    import sys
    req_packages = list(['pandas', 'numpy','progressbar'])
    install_pckg = []
    for i in range(req_packages):
        spec = importlib.util.find_spec(req_packages[i])
        if spec is None:
            install_pckg.append(req_packages[i])
    if len(install_pckg) != 0:
        raise Exception("Please install the following packages: "+install_pckg)
    
    # Intitialize the function(s) inputs
    def __init__ (self,data,K,method,max_iterations = 50):
        import numpy as np
        self.K = K
        self.max_iterations = max_iterations
        if method == None:
            raise Exception ("Please provide method for centre initialization as either 'random' or 'KMeans++'.")
        else:
            self.method = method
        self.data = np.array(data)
        self.n = len(data)
        self.d = data.shape[1]
        self.mu = None
    
    def _dist_from_centers(self):
        import numpy as np
        D2 = np.array([min([np.linalg.norm(x-c)**2 for c in self.mu]) for x in self.data])
        self.D2 = D2
        return self.D2
    
    def _choose_next_center(self):
        import numpy as np
        self.probs = self.D2/self.D2.sum()
        self.cumprobs = self.probs.cumsum()
        r = np.random.random()
        ind = np.where(self.cumprobs >= r)[0][0]
        return self.data[ind]

    def K_meansplusplus(self):
        import numpy as np
        self.mu = list(self.data[np.random.choice(self.n,1,replace=False)])
        while len(self.mu) < self.K:
                self._dist_from_centers()
                self.mu.append(self._choose_next_center())
        return self.mu
       
    def K_means_train (self):
        import numpy as np
        from progressbar import ProgressBar
        pbar = ProgressBar()
        # initialize the centers either randomly (uniform from data set) or by KMeans++
        if self.method == 'random':
            C = self.data[np.random.choice(self.n,self.K,replace=False)]
        else:
            C = self.K_meansplusplus()
        # remember old assignment
        old_assignment = np.zeros(self.n)
        # Perform Iterations
        for iteration in pbar(range(self.max_iterations)):
            # Assignment Phase:
            dist = np.zeros((self.n,self.K)) # holds distances
            for i in range(self.n): # for each data point
                for k in range(self.K): # for each cluster / center
                    # compute distance and save it
                    dist[i,k] = np.linalg.norm(self.data[i]-C[k])
            # assignment is the id of the closest center
            assignment = dist.argmin(1)
            # convergence
            if np.all(assignment == old_assignment):
                print("Done @ iteration",iteration)
                break
            else:
                old_assignment = assignment.copy()
            # maximization phase
            for k in range(self.K):
                C[k] = self.data[assignment == k].mean(0)
        # Compute Schwarz criterion:
        schwarz_crit = sum (np.amin(dist,1)) + (self.K * self.d * np.log(self.n))
        return C,assignment,schwarz_crit

    # Create prediction function which is a static method since it uses non of the class variables
    @staticmethod
    def K_means_predict(centres,data):
        import pandas as pd
        from progressbar import ProgressBar
        import numpy as np
        pbar = ProgressBar() 
        #Check if dimensions of data and the centres are the same
        if data == None:
            raise Exception('Please provide data')
        else:
            df = pd.DataFrame(data)
        if centres == None:
            raise Exception ("Please provide cluster representatives from trained model as a df of K centres of d - dimensions (K * d)")
        else:
            model = pd.DataFrame(centres)
        n = df.shape[0]
        K = model.shape[0]
        d = model.shape[1]
        if df.shape[1] != d:
            raise Exception("Please provide cluster representatives from trained model as a df of K centres of d - dimensions (K * d)")
        else:
            dist = np.zeros((n,K)) # holds distances
            for i in pbar(range(n)): # for each data point
                for k in range(K): # for each cluster / center
                    # compute distance and save it
                    dist[i,k] = np.linalg.norm(df.loc[i]-model.loc[k])
            # assignment is the id of the closest center
            assignment = dist.argmin(1)
            df['cluster_assignment'] = assignment
        return df