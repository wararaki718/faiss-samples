import gc
from pathlib import Path

from ann import (
    LinearIP,
    LinearL2,
    HNSW,
    IVF,
    PQ,
    LSH
)
from loader import load_pokemon
from utils import show
from vectorizer import vectorize


def main():
    n_nn = 16
    n_bits = 128
    
    df = load_pokemon(Path("data/Pokemon.csv"))
    print(f"pokemon data: {df.shape}")
    target = 30 # pika
    
    names = df.Name.tolist()
    x = vectorize(df)
    d = x.shape[1]
    print(f"the number of feature: {x.shape[1]}")
    print()

    print("linear ip")
    ip = LinearIP(d)
    show(ip, x, target, names)
    del ip
    gc.collect()
    
    print("linear l2")
    l2 = LinearL2(d)
    show(l2, x, target, names)
    del l2
    gc.collect()
    
    print("hnsw")
    hnsw = HNSW(d, n_nn)
    show(hnsw, x, target, names)
    del hnsw
    gc.collect()

    print("ivf")
    ivf = IVF(d, n_nn)
    show(ivf, x, target, names)
    del ivf
    gc.collect()

    print("pq")
    pq = PQ(d, 5, n_bits)
    show(pq, x, target, names)
    del pq
    gc.collect()

    print("lsh")
    lsh = LSH(d, n_bits)
    show(lsh, x, target, names)
    del lsh
    gc.collect()

    print("DONE")


if __name__ == "__main__":
    main()
