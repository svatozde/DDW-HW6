import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os.path
import scipy.stats as st
import math as m
from scipy.special import expit
from sklearn.metrics import roc_curve, auc


def user_sim_pearson_corr(person1, person2):
    if len(person1) == 0 or len(person2) == 0.0:
        return 0
    dist, _ = st.pearsonr(person1, person2)
    if m.isnan(dist):
        return 0.0
    return dist


def pearson_dist_reduced(person1, person2):
    assert len(person1) == len(person2)
    p1 = []
    p2 = []
    for i in range(0, len(person1)):
        if person1[i] != 0 and person2[i] != 0:
            p1.append(person1[i])
            p2.append(person2[i])
    np1 = np.array(p1)
    np2 = np.array(p2)
    return (user_sim_pearson_corr(np1, np2), len(np1))


def user_cosine_dist(person1, person2):
    prod = sum(person1 * person2)
    sqr1 = sum(person1 ** 2) ** (1 / 2)
    sqr2 = sum(person2 ** 2) ** (1 / 2)
    if sqr1 == 0 or sqr2 == 0:
        return 0
    return (prod / (sqr1 * sqr2))


def get_bucket(num):
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
movie_genre_map = dict()
movie_name_map = dict()
for row in m_arr:
    m_id = row[0]
    m_name = row[1]
    movie_name_map[m_id] = m_name
    gen_arr = str(row[2]).split('|')
    if not m_id in movie_genre_map:
        movie_genre_map[m_id] = np.zeros(num_genres)
    for gg in gen_arr:
        movie_genre_map[m_id][genre_map_dict[gg]] = 1

# print(genres)
ratigs_df = pd.read_csv('data/ratings.csv', header=0)
tmp_arr = ratigs_df.as_matrix()

#tmp_arr = tmp_arr[:2000]

del ratigs_df

u_arr = []
test_arr = []

for i, r in enumerate(tmp_arr):
    if i % 2 == 0:
        test_arr.append(r)
    else:
        u_arr.append(r)

# reduce size fo rdevelopment

del tmp_arr

test_map = dict()
for row in test_arr:
    u_id = row[0]
    m_id = row[1]
    rating = row[2]
    if not u_id in test_map:
        test_map[u_id] = dict()
    test_map[u_id][m_id] = rating

print('test_map size: ' + str(len(test_map)))
# prepare set of movie ids and their maping to comaprable vector
movie_set = set()
for row in u_arr:
    movie_set.add(row[1])

vector_header = list(movie_set)
vector_header.sort()
movie_map = dict()

for i, v in enumerate(vector_header):
    movie_map[v] = i

print('Done movies')

user_map_of_ratings = dict()
user_map_of_genres = dict()
user_map_of_weighted_genres = dict()
user_map_of_rated_movies = dict()
user_map_of_ratings_sorted = dict()
for row in u_arr:
    u_id = row[0]
    m_id = row[1]
    rating = row[2]
    if not u_id in user_map_of_genres:
        user_map_of_genres[u_id] = np.zeros(num_genres)
    if rating >= 3:
        rating_modified = movie_genre_map[m_id]
        user_map_of_genres[u_id] += rating_modified

    if not u_id in user_map_of_weighted_genres:
        user_map_of_weighted_genres[u_id] = np.zeros(num_genres)

    rating_modified = movie_genre_map[m_id] * rating
    user_map_of_weighted_genres[u_id] += rating_modified

    if not u_id in user_map_of_rated_movies:
        user_map_of_rated_movies[u_id] = set()
    user_map_of_rated_movies[u_id].add(m_id)

    if not u_id in user_map_of_ratings:
        user_map_of_ratings[u_id] = np.zeros(len(movie_map))
    user_map_of_ratings[u_id][movie_map[m_id]] = rating

    if not u_id in user_map_of_ratings_sorted:
        user_map_of_ratings_sorted[u_id] = []
    user_map_of_ratings_sorted[u_id].append((m_id, rating))

for u_id in user_map_of_ratings_sorted:
    user_map_of_ratings_sorted[u_id].sort(key=lambda e: e[1], reverse=True)

print('Done users')

recomnadation_map = dict()
if os.path.isfile('pickles/recomnadation_map.pickle'):
    with open('pickles/recomnadation_map.pickle', 'rb') as handle:
        recomnadation_map = pickle.load(handle)

else:
    for u_id in user_map_of_genres:
        tmp = []
        for m_id in movie_genre_map:
            if not m_id in user_map_of_rated_movies[u_id]:
                u_vect = user_map_of_genres[u_id]
                m_vect = movie_genre_map[m_id]
                dist = user_cosine_dist(u_vect, m_vect)

                tmp.append((m_id, dist))
        recomnadation_map[u_id] = np.array(sorted(tmp, reverse=True))

    with open('pickles/recomnadation_map.pickle', 'wb') as handle:
        pickle.dump(recomnadation_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

for u_id in recomnadation_map:
    recommandations = recomnadation_map[u_id]
    recommandations = sorted(recommandations, key=lambda e: e[1], reverse=True)
    recomnadation_map[u_id] = recommandations

print('Done movie similrity')

user_size = len(user_map_of_genres.keys())
sorted_keys = list(user_map_of_ratings.keys())
sorted_keys = sorted(sorted_keys)

cosine_similarity_matrix = {}
pearson_reduced_similarity_matrix = {}
pearson_similarity_matrix = {}
genre_cosine_similarity = {}
genre_weighted_cosine_similarity = {}

if os.path.isfile('pickles/rating_similarity_matrix.pickle'):
    with open('pickles/rating_similarity_matrix.pickle', 'rb') as handle:
        cosine_similarity_matrix = pickle.load(handle)
    with open('pickles/rating_similarity_matrix_reduced.pickle', 'rb') as handle:
        pearson_reduced_similarity_matrix = pickle.load(handle)
    with open('pickles/pearson_similarity_matrix.pickle', 'rb') as handle:
        pearson_similarity_matrix = pickle.load(handle)
    with open('pickles/genre_cosine_similarity.pickle', 'rb') as handle:
        genre_cosine_similarity = pickle.load(handle)
    with open('pickles/genre_weighted_cosine_similarity.pickle', 'rb') as handle:
        genre_weighted_cosine_similarity = pickle.load(handle)
else:
    for i, ka in enumerate(sorted_keys):
        print('computing similarities for:' + str(i))
        for j, kb in enumerate(sorted_keys):
            if i <= j:
                if not ka in cosine_similarity_matrix:
                    cosine_similarity_matrix[ka] = dict()
                if not kb in cosine_similarity_matrix:
                    cosine_similarity_matrix[kb] = dict()

                cos_sim = user_cosine_dist(user_map_of_ratings[ka], user_map_of_ratings[kb])
                cosine_similarity_matrix[ka][kb] = cos_sim
                cosine_similarity_matrix[kb][ka] = cos_sim

                if not ka in pearson_reduced_similarity_matrix:
                    pearson_reduced_similarity_matrix[ka] = dict()
                if not kb in pearson_reduced_similarity_matrix:
                    pearson_reduced_similarity_matrix[kb] = dict()

                pear_sim = pearson_dist_reduced(user_map_of_ratings[ka], user_map_of_ratings[kb])
                pearson_reduced_similarity_matrix[ka][kb] = pear_sim
                pearson_reduced_similarity_matrix[kb][ka] = pear_sim

                if not ka in pearson_similarity_matrix:
                    pearson_similarity_matrix[ka] = dict()
                if not kb in pearson_similarity_matrix:
                    pearson_similarity_matrix[kb] = dict()

                pear_sim = user_sim_pearson_corr(user_map_of_ratings[ka], user_map_of_ratings[kb])
                pearson_similarity_matrix[ka][kb] = pear_sim
                pearson_similarity_matrix[kb][ka] = pear_sim

                if not ka in genre_cosine_similarity:
                    genre_cosine_similarity[ka] = dict()
                if not kb in genre_cosine_similarity:
                    genre_cosine_similarity[kb] = dict()

                pear_sim = user_cosine_dist(user_map_of_genres[ka], user_map_of_genres[kb])
                genre_cosine_similarity[ka][kb] = pear_sim
                genre_cosine_similarity[kb][ka] = pear_sim

                if not ka in genre_weighted_cosine_similarity:
                    genre_weighted_cosine_similarity[ka] = dict()
                if not kb in genre_weighted_cosine_similarity:
                    genre_weighted_cosine_similarity[kb] = dict()

                pear_sim = user_cosine_dist(user_map_of_weighted_genres[ka], user_map_of_weighted_genres[kb])
                genre_weighted_cosine_similarity[ka][kb] = pear_sim
                genre_weighted_cosine_similarity[kb][ka] = pear_sim



                # print('finish: ' + str(ka) + ' ful:' + str(cosine_similarity_matrix[ka]))
                # print('finish: ' + str(ka) + ' red:' + str(pearson_reduced_similarity_matrix[ka]))

    with open('pickles/rating_similarity_matrix.pickle', 'wb') as handle:
        pickle.dump(cosine_similarity_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('pickles/rating_similarity_matrix_reduced.pickle', 'wb') as handle:
        pickle.dump(pearson_reduced_similarity_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('pickles/pearson_similarity_matrix.pickle', 'wb') as handle:
        pickle.dump(pearson_similarity_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('pickles/genre_cosine_similarity.pickle', 'wb') as handle:
        pickle.dump(genre_cosine_similarity, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('pickles/genre_weighted_cosine_similarity.pickle', 'wb') as handle:
        pickle.dump(genre_weighted_cosine_similarity, handle, protocol=pickle.HIGHEST_PROTOCOL)

movie_count = len(movie_name_map)

map_result = dict()
map_result_count = dict()

# print(genre_weighted_cosine_similarity)
for similarity_threshold in [0.0]:
        map_result[round(similarity_threshold,1)] = []
        map_result_count[round(similarity_threshold, 1)] = 0
        print(str(similarity_threshold))
        rating_comparsion = []
        for u_id in user_map_of_rated_movies.keys():
            #print(str(similarity_threshold) + ' : ' + str(u_id) )
            cosine_similarities = cosine_similarity_matrix[u_id]
            reduced_pearson_similarities = pearson_reduced_similarity_matrix[u_id]
            pearson_similarities = pearson_similarity_matrix[u_id]
            genre_similarity = genre_cosine_similarity[u_id]
            genre_weighted_similarity = genre_weighted_cosine_similarity[u_id]
            result_similarity_users = []

            for ou_id in user_map_of_ratings:
                # print(str(u_id)+':'+str(ou_id))
                cosine_sim = cosine_similarities[ou_id]
                # print(cosine_sim)
                reduced_pearson_sim = reduced_pearson_similarities[ou_id]

                genre_sim = genre_similarity[ou_id]
                weighted_genre_sim = genre_weighted_similarity[ou_id]

                # print('w' + str(weighted_genre_sim)+': nw' +str(genre_sim))
                # print(pearson_sim)
                pearson_sim = expit(10*pearson_similarities[ou_id])

                pearson_coef = expit((reduced_pearson_sim[0]/2) * reduced_pearson_sim[1])
                result_sim = 0.0 * cosine_sim + 0* pearson_coef + 0.0 * pearson_sim + 0.0 * genre_sim + 1 * weighted_genre_sim

                result_similarity_users.append((ou_id, result_sim))

            result_similarity_users = sorted(result_similarity_users, key=lambda e: e[0], reverse=True)

            recomandations_map = dict()
            for ou_id, sim in result_similarity_users:
                if sim < similarity_threshold:
                    continue
                for m_id, rating in user_map_of_ratings_sorted[ou_id]:
                    if m_id in user_map_of_rated_movies[u_id]:
                        # print('skipping' + str(m_id))
                        continue
                    # print('adding' + str(m_id))
                    if not m_id in recomandations_map:
                        recomandations_map[m_id] = (1, rating, rating * sim, sim, m_id)
                    else:
                        o_count, o_rating, r_rating, o_sim, _ = recomandations_map[m_id]
                        recomandations_map[m_id] = (o_count + 1, o_rating + rating, r_rating + (rating * sim), o_sim + sim, m_id)

            final_recomandations = sorted(recomandations_map.values(),key=lambda e:e[4], reverse=True)
            avg_final_recomandations = []
            for count, rating, relative_rating, sim, m_id in final_recomandations[:5]:
                u_gen_m = user_cosine_dist(movie_genre_map[m_id], user_map_of_weighted_genres[u_id])
                if not m_id in user_map_of_rated_movies[u_id] and m_id in test_map[u_id]:
                    avg_final_recomandations.append((sim , rating / count, m_id))

            # avg_final_recomandations = [(user_cosine_dist(movie_genre_map[m_id],user_map_of_weighted_genres[u_id])*(sim/count)*(rating/count),rating/count,m_id) for count,rating,sim,m_id in final_recomandations]
            avg_final_recomandations = sorted(avg_final_recomandations, reverse=True)

            #print(str(u_id) + ' has ' + str(len(avg_final_recomandations)) + ' recomandations')

            for r in avg_final_recomandations[:min(20,len(final_recomandations))]:
                m_id = r[2]
                if u_id in test_map and m_id in test_map[u_id]:
                    # print(str(u_id)+':'+str(r))
                    real_rating = test_map[u_id][m_id]
                    guess_rating = get_bucket(r[1])
                    #print('u:' + str(u_id) + ' : ' + str(movie_name_map[m_id]) + ' with ' + str(r) + ' real_rating:' + str(real_rating) + ' guess_rating:' + str(guess_rating))

                    real_rating = round(real_rating, 1)
                    guess_rating = round(guess_rating, 1)

                    rating_comparsion.append((guess_rating, real_rating))




        print('number od recomandations: ' + str(len(rating_comparsion)))
        for threshold in [0.0,0.5,1.0,1.5, 2.0, 2.5, 3.0, 3.5, 4, 4.5]:
            print(str(threshold) + ':' + str(similarity_threshold))
            true_positive = 0
            true_negative = 0
            false_positive = 0
            false_negative = 0

            for guess_rating, real_rating in rating_comparsion:
                if guess_rating >= threshold and real_rating >= threshold:
                    true_positive += 1
                elif guess_rating >= threshold and real_rating < threshold:
                    false_positive += 1
                elif guess_rating < threshold and real_rating < threshold:
                    true_negative += 1
                elif guess_rating < threshold and real_rating >= threshold:
                    false_negative += 1

            print('Threshold:' + str(threshold) + ' and ' + str(similarity_threshold))

            print('tp:' + str(true_positive) + ' tn:' + str(true_negative) + ' fp:' + str(
                false_positive) + ' fn:' + str(false_negative))

            if true_positive + false_positive != 0.0:
                precision = true_positive / (true_positive + false_positive)
                print('precision: ' + str(precision))
            if(true_positive+true_negative+false_negative+false_positive)!=0:
                accuracy = (true_positive+true_negative)/(true_positive+true_negative+false_negative+false_positive)
                print('accuracy:' + str(accuracy) )

            if true_positive + false_negative != 0.0:
                recall = true_positive / (true_positive + false_negative)
                print('recall: ' + str(recall))

            map_result[similarity_threshold].append((threshold,precision,recall,accuracy))

            map_result_count[similarity_threshold] = len(rating_comparsion)

        confusion_matrix = dict()
        for guess_rating, real_rating in rating_comparsion:
            if not int(guess_rating * 10) in confusion_matrix:
                confusion_matrix[int(guess_rating * 10)] = dict()
            if not int(real_rating * 10) in confusion_matrix[int(guess_rating * 10)]:
                confusion_matrix[int(guess_rating * 10)][int(real_rating * 10)] = 0
            confusion_matrix[int(guess_rating * 10)][int(real_rating * 10)] += 1



        conf_arr = np.zeros((10,10))

        for x,k in enumerate(sorted(confusion_matrix.keys(), reverse=False)):
            for y,i in enumerate(sorted(confusion_matrix[k].keys(), reverse=False)):
                #print('g:' + str(k) + ' r:' + str(i) + ' = ' + str(confusion_matrix[k][i]))
                conf_arr[x][y]=int(confusion_matrix[k][i])

        with open( 'maps/' + str(similarity_threshold) + '__map', 'wb') as handle:
            pickle.dump(conf_arr, handle, protocol=pickle.HIGHEST_PROTOCOL)


        precision_arr = np.zeros((10, 10))
        racall_arr = np.zeros((10, 10))
        for i in range(0,10):

            min_i = max([0,i-1])
            max_i = min([i+1,10])
            precision_arr[i][i] =conf_arr[i,i]/ sum(conf_arr[:,i])
            racall_arr[i][i] = conf_arr[i,i] / sum(conf_arr[i,:])





        fig, (ax, ax1,ax2) = plt.subplots(ncols=3, figsize=(21, 10))
        # Using matshow here just because it sets the ticks up nicely. imshow is faster.
        ticks = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

        ax.matshow(conf_arr, cmap='Blues', interpolation='nearest')
        ax.set_xticks(np.arange(len(ticks)))
        ax.set_yticks(np.arange(len(ticks)))
        ax.set_xticklabels(ticks)
        ax.set_yticklabels(ticks)
        ax.set_xlabel('Computed rating')
        ax.set_ylabel('Real rating')
        ax.set_title('Confusion heatmap for rating with similarity > ' + str(similarity_threshold))
        for (i, j), z in np.ndenumerate(conf_arr):
            ax.text(j, i, str(int(z)), ha='center', va='center')

        ax1.matshow(precision_arr, cmap='Blues', interpolation='nearest')
        ax1.set_xticks(np.arange(len(ticks)))
        ax1.set_yticks(np.arange(len(ticks)))
        ax1.set_xticklabels(ticks)
        ax1.set_yticklabels(ticks)
        ax1.set_xlabel('Computed rating')
        ax1.set_ylabel('Real rating')
        ax1.set_title('Precission for similarity > ' + str(similarity_threshold))
        for i in range(0,10):
            ax1.text(i, i, "{0:.2f}".format( precision_arr[i][i]), ha='center', va='center')

        ax2.matshow(precision_arr, cmap='Blues', interpolation='nearest')
        ax2.set_xticks(np.arange(len(ticks)))
        ax2.set_yticks(np.arange(len(ticks)))
        ax2.set_xticklabels(ticks)
        ax2.set_yticklabels(ticks)
        ax2.set_xlabel('Computed rating')
        ax2.set_ylabel('Real rating')
        ax2.set_title('Recall for similarity > ' + str(similarity_threshold))
        for i in range(0, 10):
            ax2.text(i, i, "{0:.3f}".format(racall_arr[i][i]), ha='center', va='center')


        fig.savefig('figs/'+ str(similarity_threshold) + '.jpg')
        plt.cla()
        plt.clf()


with open('pickles/map_result.pickle', 'wb') as handle:
    pickle.dump(map_result, handle, protocol=pickle.HIGHEST_PROTOCOL)


fig, axes = plt.subplots(nrows=len(map_result), figsize=(250,250))
#fig.subplots_adjust(left  = 0.125 ,right = 0.9 ,bottom = 5,  top = 10 ,wspace = 10 ,hspace = 10 )

for i,key in enumerate(map_result.keys()):
    similarities = []
    precisions = []
    recals = []
    accc = []

    for s,p,r,a in map_result[key]:
        similarities.append(s)
        precisions.append(p)
        recals.append(r)
        accc.append(a)
    axes[i].plot(similarities,precisions,'r')
    axes[i].plot(similarities,recals, 'b')
    axes[i].plot(similarities, accc, 'g')
    axes[i].set_xticklabels(similarities)
    axes[i].set_xticks([0.5*a for  a in range(len(similarities))])
    axes[i].set_title('Precision and recal for similarity' + str(key), y=1.15)


plt.show()
fig.savefig('figs/recalls_and_precisions.jpg')








print('Done users matrix')

"""



for i,ka in enumerate(sorted_keys):
    for j, kb in enumerate(sorted_keys):
        if i <= j:
            genre_similarity_matrix[i][j]=genre_similarity_matrix[j][i] = user_cosine_dist(user_map_of_genres[ka], user_map_of_genres[kb])
            rating_similarity_matrix[i][j]=rating_similarity_matrix[j][i]=user_cosine_dist(user_map_of_ratings[ka], user_map_of_ratings[kb])




fig = plt.figure()
plt.imshow(genre_similarity_matrix, cmap='hot', interpolation='nearest')
plt.show()
fig.savefig('genre_similarity_matrix')

fig = plt.figure()
plt.imshow(rating_similarity_matrix, cmap='hot', interpolation='nearest')
plt.show()
fig.savefig('similarity_matrix')

"""
