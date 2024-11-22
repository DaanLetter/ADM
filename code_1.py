import numpy as np
import matplotlib.pyplot as plt
import scipy 

file  = np.load('user_movie_rating.npy')
user_id = file[:,0]
movie_id = file[:,1]
rating = file[:,2]

similarity_threshold = 0.5

print(file)

def jaccard_similarity(user1, user2):
    intersection = len(np.intersect1d(user1, user2))
    union = len(np.union1d(user1, user2))
    return intersection / union
    # return np.dot(user1, user2) / (np.linalg.norm(user1) * np.linalg.norm(user2)) # cosine similarity

def minhash(data,num_hashes):
    num_movies = data.shape[1]
    permutations = [np.random.permutation(num_movies) for _ in range(num_hashes)]
    minhash_signature = np.full((data.shape[0],num_hashes), np.inf)

    for i in range(data.shape[0]):
        for j in range(num_movies):
            if data[i,j] == 1:
                for k in range(num_hashes):
                    if permutations[k][j] < minhash_signature[i,k]:
                        minhash_signature[i,k] = permutations[k][j]
    return minhash_signature

num_hashes = 100
minhash_signature = minhash(file,num_hashes)
print(minhash_signature)


def lsh(minhash_signature, num_hashes, similarity_threshold):
    num_users = minhash_signature.shape[0]
    num_bands = 50
    band_size = num_hashes // num_bands
    candidate_pairs = set()
    for i in range(num_bands):
        band = minhash_signature[:,i*band_size:(i+1)*band_size]
        hash_band = [hash(tuple(row)) for row in band]
        hash_table = {}
        for j in range(num_users):
            if hash_band[j] in hash_table:
                hash_table[hash_band[j]].append(j)
            else:
                hash_table[hash_band[j]] = [j]
        for key in hash_table:
            if len(hash_table[key]) > 1:
                for pair in itertools.combinations(hash_table[key],2):
                    candidate_pairs.add(pair)
    return candidate_pairs

candidate_pairs = lsh(minhash_signature, num_hashes, similarity_threshold)
print(candidate_pairs)









