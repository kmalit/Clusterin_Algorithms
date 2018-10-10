class MixedGaussians:
    def __init__(self, data, K, max_iterations = 50):
        self.K = K
        self.max_iterations = max_iterations
        if data == None:
            raise Exception ("Please provide data")
        else: 
            self.data = data
        self.n = len(data)
        self.d = data.shape[1]
        self.mu = None