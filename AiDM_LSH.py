import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from optparse import OptionParser
from collections import defaultdict
from itertools import combinations

def create_matrix(file):
    "Function to create a sparse binary user-movie matrix from a file"
    user_id = file[:,0]
    movie_id = file[:,1]
    rating = file[:,2]
    data = np.ones(len(rating))
    user_movie_matrix = csc_matrix((data, (movie_id, user_id))) #heb movie en user omgedraaid, omdat we de users als colommen willen hebben
    return user_movie_matrix

#make a minhash function for the csc matrix, I realise now that a csr matrix might have been more efficient
def minhash(data, num_permutations,seed=None):
    "Function to create a MinHash signature for each user in the input data"
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

#define functions to use in the LSH function

def jaccard_similarity(signature1, signature2):
    "Function to calculate the Jaccard similarity between two MinHash signatures"
    return np.sum(signature1 == signature2) / np.sum(signature1 != np.inf)

def split_vector(signature, num_bands):
    "Function to split the MinHash signature into bands"
    column_size = signature.shape[0] // num_bands

    #validate that the number of columns is divisible by the number of bands
    if signature.shape[1] % num_bands != 0:
        raise ValueError(f"The number of columns {signature.shape[0]} must be divisible by the number of bands {num_bands} for exact band splitting.")
    
    split_arrays = np.hsplit(signature, signature.shape[0]//column_size)
    print([sub_array.shape for sub_array in split_arrays])
    return split_arrays


def Hash_band(column, num_buckets):
    "Function to hash the bands into buckets"
    #convert the array to a string and hash it
    column_str = ",".join(map(str, column))
    return hash(column_str) % num_buckets

def LSH(signature, num_bands, num_buckets, similarity_threshold):
    "Function to perform Locality Sensitive Hashing on the MinHash signatures"
    #split the signature matrix into bands
    split_arrays = split_vector(signature, num_bands)
    #create a dictionary for the buckets
    hashbucketdict = {}

    #populate the dictionary with the bucket its hashed in for each band
    for i in range(len(split_arrays)):
        hashbucketdict[f'hash{i}'] = np.apply_along_axis(Hash_band, 1, split_arrays[i], num_buckets=num_buckets)
    
    #populate a bucket dictionary with the entire hashbucketdict
    for i in range(len(split_arrays)):
        bucket_groups = defaultdict(list)
        for idx, bucket in enumerate(hashbucketdict[f'hash{i}']):
            bucket_groups[bucket].append(idx)

    #filter out any bucket with only a single index occupying it
    filtered_buckets = {bucket: indices for bucket, indices in bucket_groups.items() if len(indices) > 1 }
    print(f'the amount of buckets with multiple values: {len(filtered_buckets)}')

    #create a dictionary to store the jaccard similarity between the users
    jaccard_similarity_dict = {}
    similar_users = []

    #iterate over the filtered buckets and calculate the jaccard similarity between the users
    for bucket, indices in filtered_buckets.items():

        #get all possible combinations of the indices in the bucket and calculate the jaccard similarity
        for set1, set2 in combinations(indices, 2):
            similarity = jaccard_similarity(signature[set1,:], signature[set2,:])

            #allow only similarities higher than the set threshold to be stored
            if similarity > similarity_threshold:
                jaccard_similarity_dict[f'users: {set1}, {set2}'] = similarity
                similar_users.append((set1, set2))

    return jaccard_similarity_dict, similar_users


 

def write_to_txt(similar_users):
    with open('similar_users.txt', 'w') as f:
        for user1, user2 in similar_users:
            f.write(f"{user1},{user2}\n")

def main(seed):
    np.random.seed(seed)

    data  = np.load('user_movie_rating.npy')
    
    user_movie_matrix = create_matrix(data)
    #delete the first row and column since they are empty
    user_movie_matrix = user_movie_matrix[1:,1:]

    filename = 'minhash.npy'

    if not os.path.exists(filename):
        print('Creating minhash signature...')
        hash = minhash(user_movie_matrix, 100)
        np.save('minhash.npy', testhash)
        print('Created minhash signature.')
    else:
        hash = np.load(filename)
        print('Loaded minhash signature.')

    bands = 10
    buckets = 500
    similarity_threshold = 0.5

    jaccard_similarity_dict, similar_users = LSH(hash, bands, buckets, similarity_threshold)
    print(f'There are {len(similar_users)} similar user pairs')

    write_to_txt(similar_users)

def new_option_parser():
    result = OptionParser()
    result.add_option("--seed", dest="seed", type="int", default=None, help="random seed [%default]")

    return result

if __name__ in ('__main__', '__plot__'):
    o, arguments = new_option_parser().parse_args()
    main(**o.__dict__)