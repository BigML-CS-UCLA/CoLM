import time
import gc

from submodlib.submodlib import FacilityLocationFunction
import numpy as np
import torch
from torchmetrics.functional import pairwise_cosine_similarity

from colm.train.utils import convert_to_ordered_range, increase_array_to_threshold, decrease_array_to_threshold, increase_array_to_threshold_v2


def similarity(X, metric):
    '''Computes the similarity between each pair of examples in X.

    Args
    - X: np.array, shape [N, d]
    - metric: str, one of ['cosine', 'euclidean']

    Returns
    - S: np.array, shape [N, N]
    '''
    start = time.time()
    # Convert X to float32
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).to(torch.float32)
    else:
        X = X.to(torch.float32)

    if metric == 'cosine':
        S = pairwise_cosine_similarity(X, X)
    elif metric == 'euclidean' or metric == 'l1':
        dists = torch.cdist(X, X, p=1 if metric == 'l1' else 2)
        m = torch.max(dists)
        S = m - dists
    else:
        raise ValueError(f'unknown metric: {metric}')
    elapsed = time.time() - start
    # If similarity is NaN, do not select that example
    if torch.isnan(S).sum() > 0:
        print("Handle NaN in similarity")
        S = torch.nan_to_num(S, nan=-0.95)

    return S.cpu().to(torch.float32).numpy(), elapsed


def get_orders_and_weights(B, X, metric, y=None, per_class_start="floor", strategy="proportional"):
    '''
    Ags
    - X: np.array, shape [N, d]
    - B: int, number of points to select
    - metric: str, one of ['cosine', 'euclidean'], for similarity
    - y: np.array, shape [N], integer class labels for C classes
    - strategy: str, one of ['none', 'proportional', 'balanced']

    Returns
    - order_mg/_sz: np.array, shape [B], type int32
      - *_mg: order points by their marginal gain in FL objective (largest gain first)
      - *_sz: order points by their cluster size (largest size first)
    - weights_mg/_sz: np.array, shape [B], type float32, sums to 1
    '''
    N = X.shape[0]
    if y is None:
        y = np.zeros(N, dtype=np.int32)  # assign every point to the same class
        assert strategy == "none", f"Strategy {strategy} is not supported when the class label is not available."
    else:
        y = convert_to_ordered_range(y)
    classes = np.unique(y)
    
    if strategy == 'balanced':
        min_num_per_class = np.int32(np.floor(np.divide([sum(y == i) for i in classes], N) * B))
        max_num_per_class = np.int32(np.ceil(np.divide([sum(y == i) for i in classes], N) * B))
        num_per_class = increase_array_to_threshold_v2(min_num_per_class, max_num_per_class, B)
    elif strategy == 'proportional':
        if per_class_start == "floor":
            num_per_class = np.int32(np.floor(np.divide([sum(y == i) for i in classes], N) * B))
            num_per_class = increase_array_to_threshold(num_per_class, B)
        elif per_class_start == "ceil":
            num_per_class = np.int32(np.ceil(np.divide([sum(y == i) for i in classes], N) * B))
            num_per_class = decrease_array_to_threshold(num_per_class, B)
    elif strategy == "none":
        num_per_class = np.int32([B])
    else:
        raise ValueError(f"Strategy {strategy} is not supported.")
    
    assert num_per_class.sum() == B
    
    orders_all, weights_all = [], []
    
    for c in classes:
        class_indices = np.where(y == c)[0]
        
        if num_per_class[c] == 0:
            orders_all = np.append(orders_all, np.array([]))
            weights_all = np.append(weights_all, np.array([]))
        elif len(class_indices) == 1:
            orders_all = np.append(orders_all, class_indices)
            weights_all = np.append(weights_all, np.ones(1))
        elif len(class_indices) == num_per_class[c]:
            orders_all = np.append(orders_all, class_indices)
            weights_all = np.append(weights_all, np.ones_like(class_indices))
        else:
            S, _ = similarity(X[class_indices], metric=metric)
            flf = FacilityLocationFunction(n=len(class_indices), sijs=S, separate_rep=False, mode="dense", metric=metric)
            greedy_indices = flf.maximize(budget=num_per_class[c], optimizer="LazyGreedy", stopIfZeroGain=False, stopIfNegativeGain=False, show_progress=False)
            # print(greedy_indices)
            orders = np.array([x[0] for x in greedy_indices], dtype=np.int32)
            weights = np.zeros(num_per_class[c], dtype=np.float32)
            
            for i in range(len(class_indices)):
                # Ensure that each selected sample has positive weight
                if i in orders:
                    weights[np.where(orders == i)[0][0]] += 1
                else:
                    weights[np.argmax(S[i, orders])] += 1
            
            orders_all = np.append(orders_all, class_indices[orders])
            weights_all = np.append(weights_all, weights)
    
    orders_all = np.array(orders_all, dtype=np.int32)
    weights_all = np.array(weights_all, dtype=np.float32)
    
    return orders_all, weights_all