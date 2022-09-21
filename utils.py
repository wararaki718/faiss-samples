from time import time
from typing import List

import numpy as np

from ann import BaseANN


def show(recommender: BaseANN,
         x: np.ndarray,
         query_index: int,
         names: List[str],
         k: int=5,
         verbose: bool=False):
    build_start_tm = time()
    recommender.add(x)
    build_tm = time() - build_start_tm

    search_start_tm = time()
    distances, indices = recommender.search(x[query_index].reshape((1, -1)), k=k+1)
    search_tm = time() - search_start_tm

    if verbose:
        print(indices.shape)
        print(distances.shape)
        print(indices)
        print(distances)
        print()
    
    print(f"build_tm={build_tm}, search_tm={search_tm}")
    print(f"query: {names[query_index]} [id={query_index}]")
    for i, (ind, distance) in enumerate(zip(indices[0][1:], distances[0][1:]), start=1):
        print(f"{i}-th: {names[ind]} [id={ind}, distance={distance}]")
    print()
