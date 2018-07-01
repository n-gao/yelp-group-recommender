import sqlalchemy
from .model import *
from multiprocessing import Pool
import os.path
import pickle
import scipy.sparse as sp
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, PorterStemmer, WordNetLemmatizer
import numpy as np

class Database:
    def __init__(self, address = 'sqlite:///../yelp.db'):
        self.db_engine = sqlalchemy.create_engine(address)
        self.DBSession = sessionmaker(bind=self.db_engine)

    def __enter__(self):
        self.session = sqlalchemy.orm.scoped_session(self.DBSession)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.session.close()
        self.session = None
        if exception_type is not None:
            raise exception_type(exception_value)
            
    def _query_wrapper(args):
        start, end, query, processing = args
        with Database() as db:
            result = query(db.session).offset(start).limit(end-start).all()
            if processing is not None:
                result = list(map(processing, result))
            return result
    
    def multithread_query(processes, step_size, count, query, processing = None):
        with Pool(processes=processes) as pool:
            result = pool.map(Database._query_wrapper,
                              [(i*step_size, (i+1)*step_size, query, processing) for i in range(count//step_size)])
            return flatten(result)
            
def flatten(to_flatten):
    result = []
    for l in to_flatten:
        for item in l:
            result.append(item)
    return result

default_dir = 'DB/pickle/'

def get_entities(db, N, M):
    users = db.session.query(User).order_by(sqlalchemy.desc(User.review_count * User.friend_count)).limit(N).all()
    print('Got users')
    businesses = db.session.query(Business).order_by(sqlalchemy.desc(Business.review_count)).limit(M).all()
    print('Got businesses')
    u_ind = set([u.id for u in users])
    b_ind = set([b.id for b in businesses])
    if N == db.session.query(User).count() and M == db.session.query(Business).count():
        reviews = db.session.query(Review).all()
    else:
        reviews = db.session.query(Review).filter(Review.business_id.in_(b_ind), Review.user_id.in_(u_ind)).all()
    print('Got reviews')
    category_names = db.session.query(Category.category).distinct().all()
    cities = db.session.query(Business.city).distinct().all()
    return users, businesses, reviews, category_names, cities   


def word_preprocess(reviews, businesses, users, M, N):
    all_words = []
    table = str.maketrans('', '', string.punctuation)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    bw, uw = {}, {}
    print("Stemming, stopword removing...")
    for r in reviews:
        text = r.text.lower()
        tokens = word_tokenize(text)
        words_stem = [stemmer.stem(word) for word in tokens]
        stripped = [w.translate(table) for w in words_stem]
        words = [word for word in stripped if word.isalpha()]
        words = [w for w in words if not w in stop_words]
        all_words += words
        bw[r.business_id] = words
        uw[r.user_id] = words

    all_words_set = set(all_words)
    word_labels = list(all_words_set)
    Z = len(all_words_set)
    print("total number of words:", Z)

    BW = sp.lil_matrix((M, Z))
    UW = sp.lil_matrix((N, Z))

    print("Indexing the words...")
    business_index, word_index = {}, {}
    for ind, bus in enumerate(businesses):
        business_index[bus.id] = ind
    for ind, word in enumerate(set(all_words)):    
        word_index[word] = ind
    user_index = {}
    for ind, user in enumerate(users):
        user_index[user.id] = ind    
    for k, words_list in bw.items():
        business_ind = business_index[k]
        for w in words_list:
            word_ind = word_index[w]
            BW[business_ind, word_ind] += 1 
    for k, words_list in uw.items():
        user_ind = user_index[k]
        for w in words_list:
            word_ind = word_index[w]
            UW[user_ind, word_ind] += 1 
            
    uw_csr = UW.tocsr()
    for i in range(uw_csr.shape[0]):
        s = np.sum(uw_csr[i,:])
        if s != 0:
            uw_csr[i,:] = uw_csr[i,:]/s
    
    bw_csr = BW.tocsr()
    for i in range(bw_csr.shape[0]):
        s = np.sum(bw_csr[i,:])
        if s != 0:
            bw_csr[i,:] = bw_csr[i,:]/s
            
    return bw_csr.tolil(), uw_csr.tolil(), word_labels

def get_matrices(users, businesses, reviews, category_names, cities, need_business=False, add_cities=False, add_words=False, buss_conn_threshold=4, buss_conn_min_users=1):
    
    N = len(users)
    M = len(businesses)
    
    file_name = default_dir + 'data_N%i_M%i_B%i_BT%i_BU%i_C%i_W%i.pickle' % (N, M, int(need_business), int(buss_conn_threshold), int(buss_conn_min_users), int(add_cities), int(add_words))
    if not os.path.exists(default_dir):
        os.makedirs(default_dir)
    if os.path.isfile(file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    
    # Create friendship matrix
    user_index = {}
    for ind, user in enumerate(users):
        user_index[user.id] = ind
    friends = sp.lil_matrix((N, N))
    for i in range(N):
        for f in users[i].friended:
            if f.user_id in user_index:
                friends[i, user_index[f.user_id]] = 1
                friends[user_index[f.user_id], i] = 1
    
    # Create Business attribute matrix
    business_index = {}
    for ind, bus in enumerate(businesses):
        business_index[bus.id] = ind
    category_index = {}
    c = 0
    for cat in category_names:
        category_index[cat[0]] = c
        c += 1
    if add_cities:
        for city in cities:
            category_index[city[0]] = c
            c += 1
    D = len(category_names) #+ len(cities)
    business_attributes = sp.lil_matrix((M, D))
    for ind, business in enumerate(businesses):
        for cat in business.categories:
            business_attributes[ind, category_index[cat.category]] = 1
        if add_cities:
            business_attributes[ind, category_index[business.city]] = 1
    
    # Create review matrix
    relations = sp.lil_matrix((M, N))
    for r in reviews:
        if r.business_id in business_index and r.user_id in user_index:
            relations[business_index[r.business_id], user_index[r.user_id]] = r.stars
            
    to_save = [friends, relations, business_attributes]
    
    if need_business:
        business_conn = get_buss_conn_mat(M, relations, buss_con_threshold, buss_conn_min_users)
        business_conn = sp.lil_matrix((M,M))
        for row, data in zip(relations.transpose().rows, relations.transpose().data):
            for i, entry_1 in enumerate(row):
                for j, entry_2 in enumerate(row[i+1:]):
                    if abs(data[i] - data[j + i]) <= buss_conn_threshold:
                        business_conn[entry_1, entry_2] += 1
                        business_conn[entry_2, entry_1] += 1
        business_conn[business_conn >= buss_conn_min_users] = -1
        business_conn[business_conn >= 1] = 0
        business_conn[business_conn == -1] = 1
        to_save.append(business_conn)
    
    if add_words:
        BW, UW, word_labels = word_preprocess(reviews, businesses, users, M, N)
        to_save.append(BW)
        to_save.append(UW)
        to_save.append(word_labels)
        print('Queried words')
    
    to_save = tuple(to_save)
    with open(file_name, 'wb') as f:
        pickle.dump(to_save, f)
    return to_save

def get_buss_conn_mat(M, relations, threshold=4, min_users=1):
    business_conn = sp.lil_matrix((M,M))
    for row, data in zip(relations.transpose().rows, relations.transpose().data):
        for i, entry_1 in enumerate(row):
            for j, entry_2 in enumerate(row[i+1:]):
                if abs(data[i] - data[j + i]) <= threshold:
                    business_conn[entry_1, entry_2] += 1
                    business_conn[entry_2, entry_1] += 1
    business_conn[business_conn >= min_users] = -1
    business_conn[business_conn >= 1] = 0
    business_conn[business_conn == -1] = 1
    return business_conn

def get_data(N=100000, M=100000, need_business=False, add_cities=False, add_words=False, buss_conn_threshold=4, buss_conn_min_users=1):
    file_name = default_dir + 'data_N%i_M%i_B%i_BT%i_BU%i_C%i_W%i.pickle' % (N, M, int(need_business), int(buss_conn_threshold), int(buss_conn_min_users), int(add_cities), int(add_words))
    if not os.path.exists(default_dir):
        os.makedirs(default_dir)
    if os.path.isfile(file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)
        
    # Query data
    db = Database()
    db.__enter__()
    users, businesses, reviews, category_names, cities = get_entities(db, N, M)
    print('Queried entries')
    
    to_save = get_matrices(users, businesses, reviews, category_names, cities, need_business=need_business, add_cities=add_cities, add_words=add_words, buss_conn_threshold=buss_conn_threshold, buss_conn_min_users=buss_conn_min_users)
    
    db.__exit__(None, None, None)
    return to_save
