import pandas as pd
import numpy as np
import hashlib
import multiprocessing
import csv
from tqdm import tqdm


CORES = multiprocessing.cpu_count()
print("using cores:", CORES)

def hash1_(x):
    md5hash = hashlib.md5()
    md5hash.update(str(x).encode('utf-8'))
    return md5hash.hexdigest()
    

def hash_phone_number(phones):
    md5phones = list(map(lambda x: hash1_(x), phones))
    return md5phones


def hash_phone_numbers(phone_numbers):
    pool = multiprocessing.Pool(CORES)
    chunks = np.array_split(phone_numbers, CORES)
    
    c = []
    for i in range(CORES):
        res = pool.apply_async(hash_phone_number, args=(chunks[i],))
        c.append(res)
    pool.close()
    pool.join()
    
    results = []
    for i in c:
        results += i.get()
        
    return results

data = np.arange(10000 * 10000, 110000000)
data2 = np.arange(10000 * 10000)
print(len(data))

md5s = hash_phone_numbers(data2)

df = pd.DataFrame(data=md5s, columns=['id'])
print(df)

df.to_csv('psi1yi.csv', index=None)

md5s_1kw = hash_phone_numbers(data)
md5s_5kw = md5s[:4000 * 10000] + md5s_1kw

df2 = pd.DataFrame(data=md5s_5kw, columns=['id'])
df2.to_csv('psi5kw_inter4kw.csv', index=None)

df3 = df[:100000]

df3.to_csv('psi10w.csv', index=None)