import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    # Your code here
    l = []
    for i in range(n_samples):
        l.append(i)
    

    if shuffle == True :
        np.random.shuffle(l)


    n = n_samples / k
    
    full = []
    y = 0
    for _ in range(k):
        t = ()
        l1 = []
        l2 = []
        temp = 0
        
        x = 0 
        while x < n_samples :
            if x == y and temp < n:
                l1.append(l[x])
                temp+=1
                y+=1
                
            else : 
                l2.append(l[x])
            x+=1

        full.append( (l2 , l1) )



    return full