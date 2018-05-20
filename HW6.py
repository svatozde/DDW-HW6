import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

movies_df = pd.read_csv('movies.csv',header=0)



def user_dist_reduced(person1, person2):
    assert len(person1)==len(person2)
    p1 = []
    p2 = []
    for i in range(0,len(person1)) :
        if person1[i]!=0 and person1[2]!=0:
            p1.append(person1[i])
            p2.append(person2[i])
    np1 = np.array(p1)
    np2 = np.array(p2)
    return user_cosine_dist(np1,np2)

def user_cosine_dist(person1, person2):
    prod = sum(person1*person2)
    sqr1 =  sum(person1**2)**(1/2)
    sqr2 =  sum(person2**2)**(1/2)
    if sqr1 == 0 or sqr2 == 0:
        return 1
    return  prod/(sqr1*sqr2)

m_arr = movies_df.as_matrix()

del movies_df

genres = set()
for row in m_arr:
    #print(row)
    gen =  row[2]
    gen_arr  =   str(gen).split('|')
    #print(gen_arr)
    for g in gen_arr:
        genres.add(g)

genre_map_dict = {}
genres = list(genres)
genres.sort()
for i,g in enumerate(genres):
    genre_map_dict[g]=i



num_genres = len(genres)
movie_genre_map = {}
for row in m_arr:
    m_id = row[0]
    gen_arr = str(gen).split('|')
    if not m_id in movie_genre_map:
        movie_genre_map[m_id]=np.zeros(num_genres)
    for gg in gen_arr:
        movie_genre_map[m_id][genre_map_dict[g]]=1

del m_arr
#print(genres)
ratigs_df = pd.read_csv('ratings.csv',header=0)
m_arr = ratigs_df.as_matrix()
#prepare set of movie ids and their maping to comaprable vector
movie_set = set()
for row in m_arr:
    movie_set.add(row[1])

vector_header = list(movie_set)
vector_header.sort()
movie_map = dict()

for i,v in enumerate(vector_header):
    movie_map[v]=i

user_map_of_ratings = dict()
user_map_of_genres = dict()

for row in m_arr:
    u_id = row[0]
    m_id = row[1]
    rating = row[2]
    if not u_id in user_map_of_genres:
        user_map_of_genres[u_id] = np.zeros(num_genres)

    user_map_of_genres[u_id]+=movie_genre_map[m_id]

    if not u_id in user_map_of_ratings:
       user_map_of_ratings[u_id] = np.zeros(len(movie_map))
    user_map_of_ratings[u_id][movie_map[m_id]] = rating


user_size = len(user_map_of_genres.keys())
rating_similarity_matrix = np.zeros([user_size, user_size])
genre_similarity_matrix = np.zeros([user_size, user_size])

sorted_keys = list(user_map_of_ratings.keys())
sorted_keys.sort()

for i,ka in enumerate(sorted_keys):
    for j, kb in enumerate(sorted_keys):
        if i <= j:
            genre_similarity_matrix[i][j]=genre_similarity_matrix[j][i] = user_cosine_dist(user_map_of_genres[ka], user_map_of_genres[kb])
            rating_similarity_matrix[i][j]=rating_similarity_matrix[j][i]=user_cosine_dist(user_map_of_ratings[ka], user_map_of_ratings[kb])

with open('similarity_map.pickle', 'wb') as handle:
    pickle.dump(genre_similarity_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('rating_similarity_matrix.pickle', 'wb') as handle:
    pickle.dump(rating_similarity_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)


fig = plt.figure()
plt.imshow(genre_similarity_matrix, cmap='hot', interpolation='nearest')
plt.show()
fig.savefig('genre_similarity_matrix')

fig = plt.figure()
plt.imshow(rating_similarity_matrix, cmap='hot', interpolation='nearest')
plt.show()
fig.savefig('similarity_matrix')




