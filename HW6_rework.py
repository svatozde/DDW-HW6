import scipy.stats as st
import math
import numpy as np
import pandas as pd
import pickle
import os


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)


def pearson(x, y):
    assert len(x) == len(y)
    n = len(x)
    if n == 0:
        return 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff

    return diffprod / math.sqrt(xdiff2 * ydiff2)


def pearson_dist_reduced(person1, person2):
    p1 = []
    p2 = []
    for i in range(0, len(person1)):
        if person1[i] != 0 and person2[i] != 0:
            p1.append(person1[i])
            p2.append(person2[i])
    np1 = np.array(p1)
    np2 = np.array(p2)
    p_ret = pearson(np1,np2)
    if math.isnan(p_ret):
        return (0,len(np1))
    return (p_ret, len(np1))


def user_cosine_reduced(person1, person2):
    p1 = []
    p2 = []
    for i in range(0, len(person1)):
        if person1[i] != 0 and person2[i] != 0:
            p1.append(person1[i])
            p2.append(person2[i])
    np1 = np.array(p1)
    np2 = np.array(p2)
    return (user_cosine_dist(np1,np2),len(np1))

def user_cosine_dist(person1, person2):
    prod = sum(person1 * person2)
    sqr1 = sum(person1 ** 2) ** (1 / 2)
    sqr2 = sum(person2 ** 2) ** (1 / 2)
    if sqr1 == 0 or sqr2 == 0:
        return 0
    return (prod / (sqr1 * sqr2))


def round_to_nearest(num):
    if num <= 0.24:
        return 0.0
    if num <= 0.74:
        return 0.5
    if num <= 1.24:
        return 1.0
    if num <= 1.74:
        return 1.5
    if num <= 2.24:
        return 2.0
    if num <= 2.74:
        return 2.5
    if num <= 3.24:
        return 3.0
    if num <= 3.74:
        return 3.5
    if num <= 4.24:
        return 4.0
    if num <= 4.74:
        return 4.5
    return 5


movies_df = pd.read_csv('data/movies.csv', header=0)
m_arr = movies_df.as_matrix()
del movies_df

genres = set()
for row in m_arr:
    # print(row)
    gen = row[2]
    gen_arr = str(gen).split('|')
    # print(gen_arr)
    for g in gen_arr:
        genres.add(g)

genre_map_dict = {}
genres = list(genres)
genres.sort()
for i, g in enumerate(genres):
    genre_map_dict[g] = i

num_genres = len(genres)
movie_genre_map = {}
movie_name_map = {}
movie_index_map = {}
for i, row in enumerate(m_arr):
    m_id = row[0]
    m_name = row[1]
    movie_index_map[m_id] = i
    movie_name_map[m_id] = m_name
    gen_arr = str(row[2]).split('|')
    if  m_id not in movie_genre_map:
        movie_genre_map[m_id] = np.zeros(num_genres)

    for gg in gen_arr:
        movie_genre_map[m_id][genre_map_dict[gg]] = 1


del m_arr

# print(movie_genre_map)

ratigs_df = pd.read_csv('data/ratings.csv', header=0)
tmp_arr = ratigs_df.as_matrix()

print(len(tmp_arr))

del ratigs_df

u_arr = []
test_arr = []

for i, r in enumerate(tmp_arr):
    if i % 2 == 0:
        u_arr.append(r)
    else:
        test_arr.append(r)

del tmp_arr

test_map = dict()
for row in test_arr:
    u_id = row[0]
    m_id = row[1]
    rating = row[2]
    if not u_id in test_map:
        test_map[u_id] = dict()
    test_map[int(u_id)][int(m_id)] = rating

print('test_map size: ' + str(len(test_map)))
# prepare set of movie ids and their maping to comaprable vector


vector_header = list(movie_genre_map.keys())
vector_header.sort()
movie_map = {}
reverse_movie_map = {}
for i, v in enumerate(vector_header):
    movie_map[v] = i
    reverse_movie_map[i] = v

user_map_of_ratings = {}
user_map_of_genres = {}
user_map_of_rated_movies = {}
user_map_of_genre_ratings = {}
user_map_of_genre_counts ={}

movie_num_ratings = {}
for row in u_arr:
    u_id = row[0]
    m_id = row[1]
    rating = row[2]
    if not u_id in user_map_of_genres:
        user_map_of_genres[u_id] = np.zeros(num_genres)
    if rating >= 3:
        user_map_of_genres[u_id] += movie_genre_map[m_id]

    if not u_id in user_map_of_rated_movies:
        user_map_of_rated_movies[u_id] = set()
    user_map_of_rated_movies[u_id].add(m_id)

    if not u_id in user_map_of_ratings:
        user_map_of_ratings[u_id] = np.zeros(len(movie_map))
    user_map_of_ratings[u_id][movie_map[m_id]] = rating

    if not u_id in user_map_of_genre_ratings:
        user_map_of_genre_ratings[u_id] = np.zeros(num_genres)
    if not u_id in user_map_of_genre_counts:
        user_map_of_genre_counts[u_id] = np.zeros(num_genres)

    genres = movie_genre_map[m_id]
    user_map_of_genre_ratings[u_id]+= genres*rating
    user_map_of_genre_counts[u_id]+=genres

    if m_id not in movie_num_ratings:
        movie_num_ratings[m_id] = 0
    movie_num_ratings[m_id] += rating

print(movie_num_ratings)

"""
for u_id in user_map_of_genres:
    print(user_map_of_genres[u_id])
    user_map_of_genres[u_id]= np.round(user_map_of_genres[u_id]/max(user_map_of_genres[u_id]))
    print(user_map_of_genres[u_id])
"""

for u_id in user_map_of_genre_ratings:
   sum_ratings = user_map_of_genre_ratings[u_id]
   counts = user_map_of_genre_counts[u_id]
   user_map_of_genre_ratings[u_id]=sum_ratings/counts
   user_map_of_genre_ratings[u_id][np.isnan(user_map_of_genre_ratings[u_id])]=0
   #print(user_map_of_genre_ratings[u_id])


user_size = len(user_map_of_genres.keys())
sorted_keys = list(user_map_of_ratings.keys())
sorted_keys = sorted(sorted_keys)

reduced_cosine_similarity_matrix = {}
pearson_similarity_matrix = {}

# CALCULATEE SIMILARITIES BETWEN USERS (N*(N-1))/2

if os.path.isfile('pickles/reduced_cosine_similarity_matrix.pickle'):
    with open('pickles/reduced_cosine_similarity_matrix.pickle', 'rb') as handle:
        reduced_cosine_similarity_matrix = pickle.load(handle)
    with open('pickles/pearson_similarity_matrix.pickle', 'rb') as handle:
        pearson_similarity_matrix = pickle.load(handle)
else:
    for i, ka in enumerate(sorted_keys):
        for j, kb in enumerate(sorted_keys):
            if i <= j :
                if not ka in reduced_cosine_similarity_matrix:
                    reduced_cosine_similarity_matrix[ka] = dict()
                if not kb in reduced_cosine_similarity_matrix:
                    reduced_cosine_similarity_matrix[kb] = dict()

                cos_sim = user_cosine_dist(user_map_of_ratings[ka], user_map_of_ratings[kb])
                reduced_cosine_similarity_matrix[ka][kb] = cos_sim
                reduced_cosine_similarity_matrix[kb][ka] = cos_sim

                """
                if not ka in pearson_similarity_matrix:
                    pearson_similarity_matrix[ka] = dict()
                if not kb in pearson_similarity_matrix:
                    pearson_similarity_matrix[kb] = dict()

                pearson_sim = pearson_dist_reduced(user_map_of_ratings[ka], user_map_of_ratings[kb])
                p_sim = sigmoid(pearson_sim[0]*pearson_sim[1])
                pearson_similarity_matrix[ka][kb] = p_sim
                pearson_similarity_matrix[kb][ka] = p_sim
                """


        print('computing similarities for:' + str(i)+':'+ str(j)+ ' _ ' + str(cos_sim))

    with open('pickles/reduced_cosine_similarity_matrix.pickle', 'wb') as handle:
        pickle.dump(reduced_cosine_similarity_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('pickles/pearson_similarity_matrix.pickle', 'wb') as handle:
        pickle.dump(pearson_similarity_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

# print(cosine_similarity_matrix)

# CALCULATEE SIMILARITIES BETWEN USERS AND MOVIES N*M
number_of_movies = len(movie_genre_map)
print('number of movies: ' + str(number_of_movies))

user_movie_similarities = {}
if os.path.isfile('pickles/user_movie_genre_similarity_matrix.pickle'):
    with open('pickles/user_movie_genre_similarity_matrix.pickle', 'rb') as handle:
        user_movie_similarities = pickle.load(handle)
else:
    for ka in sorted_keys:
        print('computing similarities for:' + str(ka))
        # print(user_movie_similarities)
        for m_id in movie_genre_map.keys():
            if not ka in user_movie_similarities:
                user_movie_similarities[ka] = {}
            user_movie_similarities[ka][m_id] = user_cosine_dist(user_map_of_genres[ka], movie_genre_map[m_id])

    with open('pickles/user_movie_genre_similarity_matrix.pickle', 'wb') as handle:
        pickle.dump(user_movie_similarities, handle, protocol=pickle.HIGHEST_PROTOCOL)

super_results = {}
supper_matrixes = {}



for ratios in [(0.0, 1.0),(0.3, 0.7),(0.5, 0.5), (0.7, 0.3),(1.0, 0.0)]:
    super_results[ratios] = {}
    supper_matrixes[ratios] = {}


    test_users= 0

    for i,u_id in enumerate(user_map_of_genres):
        if(i>400):
            break
        print(u_id)
        already_rated = user_map_of_rated_movies[u_id]

#        del pearson_similarity_matrix[u_id][u_id]
#        del reduced_cosine_similarity_matrix[u_id][u_id]
        sorted_similar_users = []

        for ou_id in reduced_cosine_similarity_matrix[u_id]:
            #p_sim = pearson_similarity_matrix[u_id][ou_id]
            c_sim = reduced_cosine_similarity_matrix[u_id][ou_id]
            sorted_similar_users.append((ou_id,c_sim))

        sorted_similar_users = sorted(sorted_similar_users, key=lambda e: e[1], reverse=True)
        #print('ORDERED USERS: ' + str(sorted_similar_users))
        # tuples  of similarity  to sort (expected_rating,sim,m_id)
        recomandations = []
        genre_sims = user_movie_similarities[u_id]
        genre_ratings = user_map_of_genre_ratings[u_id]
        for m_id in movie_genre_map:
            if not m_id in already_rated:
                g_sim = genre_sims[m_id]
                u_sim = 0
                guess_rating = 0
                num_users_avg = 0
                max_users= 5
                for ou_id, ou_sin in sorted_similar_users:
                    if m_id in user_map_of_rated_movies[ou_id]:
                        guess_rating += ou_sin*user_map_of_ratings[ou_id][movie_map[m_id]]
                        u_sim +=  ou_sin
                        num_users_avg+=1
                        if num_users_avg >= max_users:
                         break;

                genres = movie_genre_map[m_id]
                guess_arr =  genre_ratings*genres
                m_guess_rating = np.mean([x for x in guess_arr if x>0])
                if math.isnan(m_guess_rating):
                    m_guess_rating = 0.0
                #print(m_guess_rating)

                if guess_rating != 0:
                    guess_rating = guess_rating/u_sim
                if u_sim != 0:
                    u_sim = u_sim/num_users_avg

                g_sim =  ratios[0]*g_sim
                u_sim = ratios[1]*u_sim

                num_ratings = 0
                if m_id in movie_num_ratings:
                    num_ratings = movie_num_ratings[m_id]

                guess_rating = (ratios[0]*m_guess_rating+ratios[1]*guess_rating)


                #guess_rating = (g_sim * guess_rating + u_sim * guess_rating) / (g_sim + u_sim)
                guess_rating = round_to_nearest(guess_rating)
                #print(guess_rating)
                f_sim =  g_sim + u_sim
                rating_sigmoid = sigmoid(f_sim*guess_rating)
                recomandations.append((num_ratings*(f_sim**3)*guess_rating,rating_sigmoid,np.round(f_sim,3),num_ratings,guess_rating, m_id))



        #print('RECOMANDATIONS: ' + str(recomandations))
        test_movies = test_map[u_id]
        test_set = set(test_movies.keys())

        test_rec = sorted(recomandations,key=lambda e:e[1],reverse=True)
        print(test_rec[0:min(len(test_rec), 10)])

        recomandations.sort(reverse=True)
        print(recomandations[0:min(len(recomandations), 10)])

        for num_of_recomandations in [1,2,3,5,10,50,500]:
            tps = 0
            tns = 0
            fps = 0
            fns = 0
            m = 0
            g_g = 0

            if not num_of_recomandations in supper_matrixes[ratios]:
                supper_matrixes[ratios][num_of_recomandations] = {}
            rating_matrix = supper_matrixes[ratios][num_of_recomandations]

            guess_set = set([mid[-1] for mid in recomandations[0:min(len(recomandations), num_of_recomandations)]])
            not_guess_set = set([mid[-1] for mid in recomandations[min(len(recomandations), num_of_recomandations):]])



            #print([mid for mid in recomandations[0:min(len(recomandations), num_of_recomandations)]])
            #print(test_movies)
            #print(guess_set)
            #print(test_set)

            tps += len(test_set & guess_set )  # guessed and watched
            tns += len(not_guess_set - test_set)
            fps += len(guess_set - test_set)
            fns += len(not_guess_set &  test_set)


            for m_r in recomandations[0:min(len(recomandations), num_of_recomandations)]:
                if m_r[-1] in test_movies:
                    rounded_guess = round_to_nearest(m_r[-2])
                    real_rating = test_movies[m_r[-1]]

                    if rounded_guess == real_rating:
                        m+=1

                    if rounded_guess <= real_rating:
                        g_g+=1

                    if not rounded_guess in rating_matrix:
                        rating_matrix[rounded_guess] = {}
                    if not real_rating in rating_matrix[rounded_guess]:
                        rating_matrix[rounded_guess][real_rating] = 0

                    rating_matrix[rounded_guess][real_rating] += 1



            if not num_of_recomandations in super_results[ratios]:
                super_results[ratios][num_of_recomandations] = (0,0,0,0,0,0)

            (o_tps, o_tns, o_fps, o_fns, o_m, o_g_g) = super_results[ratios][num_of_recomandations]
            super_results[ratios][num_of_recomandations] = (o_tps+tps,o_tns+tns, o_fps+fps, o_fns+fns,o_m+m,o_g_g+g_g)
            #print(str(ratios) + '_' + str(num_of_recomandations)),
            #print(super_results)
            #print(rating_matrix)

            (o_tps, o_tns, o_fps, o_fns, o_m, o_g_g) = super_results[ratios][num_of_recomandations]
            print(str(u_id) + ' _ ' + str(ratios) + '_' + str(num_of_recomandations) + '_' + str(
                (o_tps/(o_tps+o_fps), o_tps/(o_tps+o_fns),(o_tps+o_tns)/(o_tps+o_tns+o_fns+o_fps),o_tps,o_fps,o_fns)))

with open('pickles/super_correct_results.pickle', 'wb') as handle:
    pickle.dump(super_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('pickles/super_correct_matric.pickle', 'wb') as handle:
    pickle.dump(supper_matrixes, handle, protocol=pickle.HIGHEST_PROTOCOL)
