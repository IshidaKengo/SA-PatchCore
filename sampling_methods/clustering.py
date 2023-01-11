import torch as th

"""
Author: Josue N Rivera (github.com/JosueCom)
Date: 7/3/2021
Description: Snippet of various clustering implementations only using PyTorch
Full project repository: https://github.com/JosueCom/Lign (A graph deep learning framework that works alongside PyTorch)
"""

def randomize_tensor(tensor):
    return tensor[th.randperm(len(tensor))]

def distance_matrix(x, y=None, p = 2): #pairwise distance of vectors
    
    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    
    dist = th.pow(x - y, p).sum(2)
    
    return dist

class NN():

    def __init__(self, X = None, Y = None, p = 2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")
        
        dist = distance_matrix(x, self.train_pts, self.p) ** (1/self.p)
        labels = th.argmin(dist, dim=1)
        return self.train_label[labels]

class KNN(NN):

    def __init__(self, X = None, Y = None, k = 3, p = 2):
        self.k = k
        super().__init__(X, Y, p)
    
    def train(self, X, Y):
        super().train(X, Y)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")
        
        dist = distance_matrix(x, self.train_pts, self.p) ** (1/self.p)

        knn = dist.topk(self.k, largest=False)
        votes = self.train_label[knn.indices]

        winner = th.zeros(votes.size(0), dtype=votes.dtype, device=votes.device)
        count = th.zeros(votes.size(0), dtype=votes.dtype, device=votes.device) - 1

        for lab in self.unique_labels:
            vote_count = (votes == lab).sum(1)
            who = vote_count >= count
            winner[who] = lab
            count[who] = vote_count[who]

        return winner


class KMeans(NN):

    def __init__(self, X = None, k=2, n_iters = 10, p = 2):

        self.k = k
        self.n_iters = n_iters
        self.p = p

        if type(X) != type(None):
            self.train(X)

    def train(self, X):

        self.train_pts = randomize_tensor(X)[:self.k]
        self.train_label = th.LongTensor(range(self.k))

        for _ in range(self.n_iters):
            labels = self.predict(X)

            for lab in range(self.k):
                select = labels == lab
                self.train_pts[lab] = th.mean(X[select], dim=0)

if __name__ == '__main__':
    a = th.Tensor([
        [1, 1],
        [0.88, 0.90],
        [-1, -1],
        [-1, -0.88]
    ])

    b = th.LongTensor([3, 3, 5, 5])

    c = th.Tensor([
        [-0.5, -0.5],
        [0.88, 0.88]
    ])

    knn = KNN(a, b)
    print(knn(c))