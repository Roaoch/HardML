import math
import pickle
import random
from typing import List, Tuple

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tqdm.auto import tqdm


class Solution:
    def __init__(self, n_estimators: int = 100, lr: float = 0.5, ndcg_top_k: int = 10,
                 subsample: float = 0.6, colsample_bytree: float = 0.9,
                 max_depth: int = 5, min_samples_leaf: int = 8):
        self._prepare_data()

        self.ndcg_top_k = ndcg_top_k
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.trees = []
        self.best_ndcg = -2000,
        self.best_ndcg_i = -1

    def _get_data(self) -> List[np.ndarray]:
        train_df, test_df = msrank_10k()

        X_train = train_df.drop([0, 1], axis=1).values
        y_train = train_df[0].values
        query_ids_train = train_df[1].values.astype(int)

        X_test = test_df.drop([0, 1], axis=1).values
        y_test = test_df[0].values
        query_ids_test = test_df[1].values.astype(int)

        return [X_train, y_train, query_ids_train, X_test, y_test, query_ids_test]

    def _prepare_data(self) -> None:
        (X_train, y_train, self.query_ids_train,
            X_test, y_test, self.query_ids_test) = self._get_data()
        self.X_train = torch.FloatTensor(self._scale_features_in_query_groups(X_train, self.query_ids_train))
        self.X_test = torch.FloatTensor(self._scale_features_in_query_groups(X_test, self.query_ids_test))
        self.ys_train = torch.reshape(torch.FloatTensor(y_train), (-1, 1))
        self.ys_test = torch.reshape(torch.FloatTensor(y_test), (-1, 1))

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        scaler = StandardScaler()
        ids_indexes = []
        temp_i_val = -1
        temp_i = 0
        for i, i_val in enumerate(inp_query_ids):
            if (temp_i_val != i_val and i != 0):
                ids_indexes.append((temp_i, i))
                temp_i = i
            temp_i_val = i_val
        ids_indexes.append((temp_i, len(inp_query_ids)))
        self.ids_indexes = torch.FloatTensor(ids_indexes)
        
        for start, end in ids_indexes:
            start = int(start)
            end = int(end)
            
            group = inp_feat_array[start:end]
            scaler.fit(group)
            inp_feat_array[start:end] = scaler.transform(group)
        return inp_feat_array

    def _train_one_tree(self, cur_tree_idx: int, train_preds: torch.FloatTensor) -> Tuple[DecisionTreeRegressor, np.ndarray]:
        np.random.seed(cur_tree_idx)
        
        x_len = np.shape(self.X_train)[1]
        idd = torch.randperm(x_len)[:int(self.subsample * x_len)]
        y_len = np.shape(self.X_train)[0]
        idn = torch.randperm(y_len)[:int(self.colsample_bytree * y_len)]
        
        with torch.no_grad():
            pos_pairs_score_diff = 1.0 + torch.exp((train_preds - train_preds.t()))
            
            decay_diff = (1.0 / torch.log2(self.rank_order + 1.0)) - (1.0 / torch.log2(self.rank_order.t() + 1.0))
            delta_ndcg = torch.abs(self.N * self.gain_diff * decay_diff)
            lambda_update =  (0.5 * (1 - self.Sij) - 1 / pos_pairs_score_diff) * delta_ndcg
            lambda_update = torch.sum(lambda_update, dim=1, keepdim=True)
            return DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf).fit(self.X_train[idn][:, idd], self.ys_train[idn]), idd
        

    def _calc_data_ndcg(self, queries_list: np.ndarray,
                        true_labels: torch.FloatTensor, preds: torch.FloatTensor) -> float:
        ndcg = []
        for start, end in self.ids_indexes:
            start = int(start)
            end = int(end)
            
            true_labels_batch = true_labels[start:end]
            preds_batch = preds[start:end]
            ndcg.append(self._ndcg_k(true_labels_batch, preds_batch, 10))
        return np.mean(ndcg)

    def fit(self):
        np.random.seed(0)
        
        ideal_dcg = self._ideal_dcg(self.ys_train)
        self.N = 1 / ideal_dcg
        rel_diff = self.ys_train - self.ys_train.t()
        self.Sij = (rel_diff > 0).type(torch.float32) - (rel_diff < 0).type(torch.float32)
        self.gain_diff = torch.pow(2.0, self.ys_train) - torch.pow(2.0, self.ys_train.t()) 
        _, rank_order = torch.sort(self.ys_train, descending=True, axis=0)
        self.rank_order = rank_order + 1
        
        sum_of_pred = torch.zeros(np.shape(self.ys_train))
        sum_of_test_pred = torch.zeros(np.shape(self.ys_train))
        for i in range(self.n_estimators):
            clf, idd = self._train_one_tree(i, sum_of_pred)
            self.trees.append((clf, idd))
            preds = torch.reshape(torch.FloatTensor(clf.predict(self.X_train[:, idd])), (-1,1))
            sum_of_pred += preds*self.lr
            
            test_pred = torch.reshape(torch.FloatTensor(clf.predict(self.X_test[:, idd])), (-1,1))
            sum_of_test_pred += test_pred*self.lr
            ndcg = self._calc_data_ndcg(self.ids_indexes, self.ys_test, sum_of_test_pred)
            if ndcg > self.best_ndcg:
                self.best_ndcg = ndcg
                self.best_ndcg_i = i
        self.trees = self.trees[:self.best_ndcg_i + 1]
            

    def predict(self, data: torch.FloatTensor) -> torch.FloatTensor:
        res = torch.zeros((np.shape(data)[0], 1))
        for tree, idd in self.trees:
            res += torch.reshape(torch.Tensor(tree.predict(data[:, idd])), (-1, 1)) * self.lr
        return res

    def _compute_lambdas(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        pos_pairs_score_diff = 1.0 + torch.exp((y_pred - y_pred.t()))
        decay_diff = (1.0 / torch.log2(self.rank_order + 1.0)) - (1.0 / torch.log2(self.rank_order.t() + 1.0))
        delta_ndcg = torch.abs(self.N * self.gain_diff * decay_diff)
        lambda_update =  (0.5 * (1 - self.Sij) - 1 / pos_pairs_score_diff) * delta_ndcg
        return torch.sum(lambda_update, dim=1, keepdim=True)

    def _ndcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor, ndcg_top_k: int) -> float:
        def dcg(ys_true: torch.Tensor, ys_pred: torch.Tensor, ndcg_top_k: int) -> float:
            res = 0
            ys_true = ys_true.double()
            for i, i_true in enumerate(torch.sort(ys_pred, descending=True, axis=0).indices[:ndcg_top_k]):
                value = self._compute_gain(float(ys_true[i_true]))
                res += value / (math.log2(i + 2))
            return float(res)
        
        dc = dcg(ys_true, ys_pred, ndcg_top_k)
        idc = dcg(ys_true, ys_true, ndcg_top_k)
        return float(dc / idc)
    
    def _compute_gain(self, y_value: float) -> float:
            return (2 ** y_value) - 1
    
    def _ideal_dcg(self, ys_true: torch.Tensor) -> float:
        ys_true = ys_true.double()
        idcg = []
        for start, end in self.ids_indexes:
            start = int(start)
            end = int(end)
            
            ys = ys_true[start:end]
            res = 0
            for i, i_true in enumerate(torch.sort(ys, descending=True, axis=0).indices):
                value = self._compute_gain(float(ys[i_true]))
                res += value / (math.log2(i + 2))
            idcg.append(float(res))
        return np.mean(idcg)

    def save_model(self, path: str):
        state = {
            "ndcg_top_k":self.ndcg_top_k,
            "lr":self.lr, 
            "trees":self.trees, 
        }
        f = open(path, 'wb')
        pickle.dump(state, f)

    def load_model(self, path: str):
        f = open(path, 'rb')
        state = pickle.load(f)
        
        self.ndcg_top_k = state["ndcg_top_k"]
        self.lr = state["lr"]
        self.trees = state["trees"]
