
import torch
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.cluster import KMeans
import pygmo as pg
import numpy as np

from .utils import *
class node:
    def __init__(self, 
                 nobjs: int, 
                 ref_point = None,
                 parent = None , 
                 dims : int = 0,
                 Cp: float = 0.1,
                 id: int = 0, 
                 weight_method : str = "hv",
                 cluster_type : str = "dominant",
                 kernel_type : str = "poly") -> None:
        
        self.nobjs = nobjs
        self.dims = dims
        self.parent = parent
        self.ref_point = ref_point
        self.Cp = Cp
        self.id = id

        self.n = 0
        self.weight = float('inf')
        self.weight_method = weight_method
        self.hv = 0
        self.is_splittable = False
        self.classifer = SVC(kernel=kernel_type, gamma="scale")
        # self.classifer = RandomForestClassifier()
        self.kids = [] # Length == nobjs
        self.X = torch.tensor([])
        self.Y = torch.tensor([])

        if cluster_type == "kmeans":
            self.learn_cluster = self.learn_kmeans_clusters
        elif cluster_type == "dominant":
            self.learn_cluster = self.learn_dominance_clusters
        elif cluster_type == "hybrid":
            self.learn_cluster = (self.learn_dominance_clusters if self.parent is None 
                else self.learn_kmeans_clusters)
        else:
            raise NotImplementedError
        return

    def clear_data(self):

        self.X = torch.tensor([])
        self.Y = torch.tensor([])
        return
    
    def update_bag(self, X, Y, global_pareto = None):

        self.X = X
        self.Y = Y
        self.is_splittable = self.is_svm_splittable()
        self.hv = get_hv(self.Y, self.ref_point)
        if self.weight_method == "global-hv":
            self.weight = self.get_exp_hv_gain(global_pareto)
        elif self.weight_method == "hv":
            self.weight = self.get_weight()
        else:
            raise NotImplementedError
        return

    def get_weight(self):
        if self.parent is None:
            return float('inf')
        else:
            ucb = self.Cp * self.hv * np.sqrt(2 * np.log(self.parent.X.shape[0]) / self.X.shape[0])
            return self.hv + ucb
        
    def get_exp_hv_gain(self, curr_pareto: torch.Tensor):

        if self.parent is None:
            return float('inf')
        
        ucb = (curr_pareto.max(dim=0).values * 
            self.Cp * 
            np.sqrt(2 * np.log(self.parent.X.shape[0]) / self.X.shape[0]))

        self.obj_ucb = (self.Y.mean(dim=0) + ucb).unsqueeze(0)
        test_Y = torch.cat((self.obj_ucb, curr_pareto))

        if is_non_dominated(test_Y)[0]:
            return get_hv(test_Y, self.ref_point)
        else:
            distIdx = torch.norm(curr_pareto - self.obj_ucb, dim=1).argmin()
            # curr_pareto[distIdx] = self.obj_ucb
            # return get_hv(curr_pareto, self.ref_point) 
            dist = curr_pareto[distIdx] - self.obj_ucb
            neg_hv = torch.prod(dist)
            # print(neg_hv)
            return get_hv(curr_pareto, self.ref_point) - neg_hv

    def learn_dominance_clusters(self, mode = "even"):

        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(self.Y.cpu().numpy())

        # Good Samples - 0, Bad Samples - 1
        if mode == "uneven":
            # Uneven Cluster
            plabel = torch.tensor(ndr < ndr.mean(), dtype=int)
        elif mode == "even":
            # Even Cluster
            indice = np.argsort(dc)
            ranks = np.empty_like(indice)
            ranks[indice] = np.arange(len(dc))
            plabel = torch.tensor(ranks < self.Y.shape[0]//2, dtype=int)

        return plabel
    
    def learn_kmeans_clusters(self):
        kmeans = KMeans(self.nobjs, n_init='auto', max_iter=1000)
        if (self.Y.shape[0] < self.nobjs):
            return torch.zeros(self.Y.shape[0])
        kmeans.fit(self.Y.cpu().numpy())
        plabel = kmeans.predict(self.Y.cpu().numpy())
        return torch.tensor(plabel)

    def split_data(self):
        assert len(self.X) > 0 and len(self.Y) > 0
        # plabel = self.learn_dominance_clusters()
        plabel = self.learn_cluster()

        self.classifer.fit(self.X.cpu().numpy(), plabel.cpu().numpy())
        self.is_splittable = self.is_svm_splittable()
        partitions = []
        if self.is_splittable:
            for i in plabel.unique():
                partitions.append((self.X[plabel==i], self.Y[plabel==i]))
        else:
            print("[DDSP] Node not splittable!")
        return partitions

    def update_kids(self, kids):
        self.kids = kids

    def select_kid(self):
        best_kid = max(self.kids, key=lambda x: x.weight)
        for label in range(len(self.kids)):
            if self.kids[label] == best_kid:
                return best_kid, label
        return None

    def is_svm_splittable(self):
        # plabel = self.learn_dominance_clusters()
        if len(self.Y) < 2:
            return False
        plabel = self.learn_cluster()
        if len(plabel.unique()) < 2:
            # print("[DDSP] Clustering Not Splittable!")
            return False
        self.classifer.fit(self.X.cpu().numpy(), plabel.cpu().numpy())
        plabel = torch.tensor(self.classifer.predict(self.X.cpu().numpy()))
        if len(plabel.unique()) < 2:
            # print("[DDSP] Not SVM Not Splittable!")
            return False
        return True

    def is_leaf(self):
        return len(self.kids) == 0

    def depth(self):
        dp = 0
        p = self.parent
        while p is not None:
            dp += 1
            p = p.parent
        return dp
    
    def __str__(self):
        
        node_name = " "*self.depth() + "└" if self.parent is not None else ""
        node_name += "node " + str(self.id)
        node_name += " "*(16 - len(node_name))
        
        node_name += "leaf: " + str(self.is_leaf())
        node_name += " "*(32 - len(node_name))

        node_name += "sp: " + str(self.X.shape[0]) 
        node_name += "/" + str(self.parent.X.shape[0]) if self.parent is not None else ""
        node_name += " "*(48 - len(node_name))
        
        node_name += f"hv: {self.hv:.4f}" + " "*4
        node_name += f"weight: {self.weight:.4f}" + " "*4

        return node_name