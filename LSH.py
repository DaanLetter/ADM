import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from optparse import OptionParser

def create_matrix(file):
    user_id = file[:,0]
    movie_id = file[:,1]
    rating = file[:,2]
    data = np.ones(len(rating))
    user_movie_matrix = csc_matrix((data, (movie_id, user_id))) #heb movie en user omgedraaid, omdat we de users als colommen willen hebben
    return user_movie_matrix

#make a minhash function for the csc matrix, I realise now that a csr matrix might have been more efficient
def minhash(data, num_permutations,seed=None):
    if not isinstance(data, csc_matrix):
        raise ValueError("Input data must be a scipy.sparse CSC matrix.")
    
    num_movies = data.shape[1]  # Number of columns (features)
    num_users = data.shape[0]   # Number of rows (samples)
    
    # Create random permutations
    permutations = [np.random.permutation(num_movies) for _ in range(num_permutations)]
    perm_signature = np.full((num_users, num_permutations), np.inf)
    
    # Iterate over columns (movies/features) of the CSC matrix
    for movie_index in range(num_movies):
        # Get all non-zero row indices for the current column
        row_indices = data.indices[data.indptr[movie_index]:data.indptr[movie_index + 1]]
        
        for k in range(num_permutations):
            # Compute the hash value for the current column (movie_index)
            hash_value = permutations[k][movie_index]
            
            # Update the MinHash signature for the rows with non-zero values in this column
            for row_index in row_indices:
                perm_signature[row_index, k] = min(perm_signature[row_index, k], hash_value)

    return perm_signature#.transpose()

def jaccard_similarity(signature1, signature2):
    return np.sum(signature1 == signature2) / np.sum(signature1 != np.inf)

def lsh(signatures, num_hashes, num_bands, similarity_threshold):
    
    band_size = num_hashes // num_bands

    buckets = {}
    for user_id, signature in enumerate(signatures):
        for band in range(num_bands):
            # Group the portion of the signature belonging to this band
            band_hash = tuple(signature[band * band_size:(band + 1) * band_size])
            if band_hash not in buckets:
                buckets[band_hash] = []
            buckets[band_hash].append(user_id)

    print(buckets)

    candidate_pairs = set()
    for bucket_users in buckets.values():
        for i in range(len(bucket_users)):
            for j in range(i + 1, len(bucket_users)):
                candidate_pairs.add((bucket_users[i], bucket_users[j]))
    
    print(candidate_pairs)
    similar_users = []
    for user1, user2 in candidate_pairs:
        sim = jaccard_similarity(signatures[user1], signatures[user2])
        if sim >= similarity_threshold:
            similar_users.append((user1, user2))

    return similar_users

def write_to_txt(similar_users):
    with open('similar_users.txt', 'w') as f:
        for user1, user2 in similar_users:
            f.write(f"{user1},{user2}\n")

def main(seed):
    data  = np.load('user_movie_rating.npy')
    similarity_threshold = 0.5

    user_movie_matrix = create_matrix(data)
    #delete the first row and column since they are empty
    user_movie_matrix = user_movie_matrix[1:,1:]

    if not os.path.exists('minhash.npy'):
        print('Creating minhash signature...')
        testhash = minhash(user_movie_matrix, 100)
        np.save('minhash.npy', testhash)
        print('Created minhash signature.')
    else:
        testhash = np.load('minhash.npy')
        print('Loaded minhash signature.')

    similar_users = lsh(testhash, 100, 50, 0.5)
    print(similar_users)

    #todo check what exactly the numbers are, are they the user id's or movie id's or what are they and why are they the same number.
    # It could have something to do with the fact that when I ran this code I had not yet transposed the minhash signature,
    # so the users and movies were switched.

    print(len(similar_users))

    write_to_txt(similar_users)

def new_option_parser():
    result = OptionParser()
    result.add_option("--seed", dest="seed", type="int", default=None, help="random seed [%default]")

    return result

if __name__ in ('__main__', '__plot__'):
    o, arguments = new_option_parser().parse_args()
    main(**o.__dict__)