from DB.database import Database, get_data, get_entities
from DB.model import *
import numpy as np
from multiprocessing import Pool
import pickle
import src.sc as sc
import src.mc as mc
from collections import Counter
import math

valid_inds_cache = {}
def get_valid_inds(relations, min_reviews=10):
    if min_reviews not in valid_inds_cache:
        valid_inds_cache[min_reviews] = [relations[:,i].nnz > min_reviews for i in range(relations.shape[1])]
    return valid_inds_cache[min_reviews]

def get_friended_users(g, relations, friends, min_reviews=10):
    valid_inds = get_valid_inds(relations, min_reviews)
    possible_users = np.arange(relations.shape[1])[valid_inds]
    u = np.random.choice(np.where(np.logical_and(friends.sum(0) > (g//2 - 1), valid_inds))[1])
    friend_ids = np.append(np.random.choice(friends[u].rows[0], g-1), u)
    return friend_ids

def get_random_users(g, relations, min_reviews=10):
    valid_inds = get_valid_inds(relations, min_reviews)
    possible_users = np.arange(relations.shape[1])[valid_inds]
    return possible_users[np.random.randint(0, len(possible_users), g)]

def get_nearest_users(g, U):
    neigh = NearestNeighbors()
    neigh.fit(U)
    u = get_random_users(1)[0]
    nearest_ids = neigh.kneighbors([U[u]], g, return_distance=False)[0]
    return nearest_ids

def collaborative_filtering(user_ids, U, V, means):
    return V.dot(U[user_ids].T) + means[user_ids]

def clip(scores, min_val=1.0000001, max_val=4.9999999):
    return np.clip(scores, min_val, max_val)

def scale_linearly(scores, min_val=1, max_val=5):
    new_scores = scores.copy()
    s_max = np.max(new_scores, 0)
    s_min = np.min(new_scores, 0)
    s_range = s_max - s_min
    new_scores = new_scores - s_min
    new_scores = new_scores / s_range
    new_scores = new_scores * (max_val - min_val)
    new_scores = new_scores + min_val
    return new_scores

def fill_real_ratings(scores, user_reviews):
    nnz = np.where(user_reviews != 0)
    scores = scores.copy()
    scores[nnz] = user_reviews[nnz]
    return scores

def filter_cities_and_categories(scores, businesses, cities, categories):
    inds = []
    original_inds = {}
    for i, b in enumerate(businesses):
        b_cats = np.array([c.category for c in b.categories])
        if np.isin(b.city, cities).any() and np.isin(categories, b_cats).all():
            original_inds[len(inds)] = i
            inds.append(i)
    return scores[np.array(inds)], original_inds

def filter_city(scores, businesses, city):
    return filter_cities_and_categories(scores, businesses, [city], [])

def filter_misery(scores, n, ind_transition=None):
    if ind_transition is None:
        ind_transition = {}
    perc_ind = math.ceil((n/scores.shape[0]) * (scores.shape[0] - 1))
    perc_ind = np.clip(perc_ind, 0, scores.shape[0] - 1)
    p = np.sort(scores, 0)[::-1][perc_ind, :]
    n = min(n, perc_ind + 1)
    result = np.ndarray((n, scores.shape[1]))
    original_inds = {}
    for i in range(scores.shape[1]):
        mask = np.where(scores[:,i] >= p[i])
        result[:,i] = scores[mask, i][:,:n]
        for j in range(n):
            ind = mask[0][j]
            original_inds[(i, j)] = ind_transition[ind] if ind in ind_transition else ind
    return result, original_inds

def order(scores, ind_transition=None):
    M, g = scores.shape
    permutations = []
    ratings = []
    for i in range(g):
        rat, perm = zip(*sorted(zip(scores[:,i], range(M))))
        perm = list(reversed(perm))
        if ind_transition != None: # Evin added this part. 
            perm = [ind_transition[(i, p)] for p in perm]
        rat = list(reversed(rat))
        permutations.append(perm)
        ratings.append(rat)
    return np.array(ratings), np.array(permutations)


def get_recommendations(user_ids, N, M, city='Las Vegas', factorization_file='factorized_location.pickle', misery_perc = 0.75):
    db = Database()
    db.__enter__()
    users, businesses, reviews, categroy_names, cities = get_entities(db, N, M)
    friends, relations, business_attributes = get_data(N, M)
    with open(factorization_file, 'rb') as f:
        U, V, W = pickle.load(f)
    scores = collaborative_filtering(user_ids, U, V)
    scaled_scores = scale_linearly(scores, 1.00001, 4.99999)
    filled_scores = fill_real_ratings(scaled_scores, relations[:,user_ids].todense())
    filtered_scores, ind_transition = filter_city(filled_scores, businesses, city)
    n = int(filtered_scores.shape[0] * misery_perc)
    no_misery_scores, original_inds = filter_misery(filtered_scores, n, ind_transition)
    ratings, rankings = order(no_misery_scores, original_inds)
    result = mc.simulate_markov_chains(n, n * 2, rankings)
    return result

def get_most_rated_city(user_ids, R, businesses):
    rated = R[:,user_ids].nonzero()[0]
    return Counter([b.city for b in np.array(businesses)[rated]]).most_common(1)[0][0]

class GroupRecommender:
    def __init__(self, N=100000, M=100000, factorization_file='factorized_location.pickle', db=None):
        if db is None:
            self.db = Database()
            self.db.__enter__()
        else:
            self.db = db
        self.N, self.M = N, M
        self.users, self.businesses, self.reviews, self.category_names, self.cities = get_entities(self.db, N, M)
        self.friends, self.relations, self.business_attributes = get_data(N, M)
        with open(factorization_file, 'rb') as f:
            self.U, self.V, self.W = pickle.load(f)
            
    def get_recommendation(self, user_ids, number_rec, city=None, misery_perc=0.75):
        if city is None:
            city = get_most_rated_city(user_ids, self.relations, self.businesses)
        scores = collaborative_filtering(user_ids, self.U, self.V)
        scaled_scores = scale_linearly(scores, 1.00001, 4.99999)
        filled_scores = fill_real_ratings(scaled_scores, self.relations[:,user_ids].todense())
        filtered_scores, ind_transition = filter_city(filled_scores, self.businesses, city)
        n = int(filtered_scores.shape[0] * misery_perc)
        n = 1000
        no_misery_scores, original_inds = filter_misery(filtered_scores, n, ind_transition)
        ratings, rankings = order(no_misery_scores, original_inds)
        result = mc.simulate_markov_chains(2, 2000, rankings, n=number_rec)
        return result, ratings, rankings
    

if __name__ == '__main__':
    a = np.random.randn(4, 5)
    print(a)
    filtered, ind_transition = filter_misery(a, 5, {0 : 10, 1 : 4, 2:3, 3: 1, 4: 11})
    print(filtered)
    print(ind_transition)
    print(order(filtered, ind_transition))
