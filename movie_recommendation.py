import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

df=pd.read_csv('ml-100k/u.data',sep='\t',names=['user_id','item_id','rating','timestamp'])
movie_title=pd.read_csv('ml-100k/u.item',sep='\|',header=None)
item_mapped_titles=movie_title[[0,1]]
item_mapped_titles.columns=['item_id','title']

df=df.merge(item_mapped_titles,on='item_id')
df.groupby('title').mean()['rating'].sort_values(ascending=False).head()
df.groupby('title').count()['user_id'].sort_values(ascending=False)

sns.set_style('white')

avg_ratings_mapped_to_titles=pd.DataFrame(df.groupby('title').mean()['rating'])
freq_of_reviews_mapped_to_titles=pd.DataFrame(df.groupby('title').count()['user_id'])

movie_info=avg_ratings_mapped_to_titles.merge(freq_of_reviews_mapped_to_titles,left_index=True,right_index=True)
movie_info=avg_ratings_mapped_to_titles
movie_info['count_of_reviews']=freq_of_reviews_mapped_to_titles

movie_info.sort_values(by='rating',ascending=False)
movie_mat=df.pivot_table(index='user_id',columns='title',values='rating')
starwars_user_ratings=movie_mat['Star Wars (1977)']

corrTable=pd.DataFrame(movie_mat.corrwith(starwars_user_ratings),columns=['Corrilation'])
corrTable.dropna(inplace=True)
corrTable.sort_values(by='Corrilation',ascending=False)

rat_rev_corr_table=movie_info.join(corrTable)
rat_rev_corr_table[rat_rev_corr_table['count_of_reviews'] >= 100].sort_values(by=['Corrilation','rating'],ascending=False)

def recommend_movies(movie_name):
    target_movie=movie_mat[movie_name]
    corr_table=pd.DataFrame(movie_mat.corrwith(target_movie),columns=['Corrilation'])
    corr_table.dropna(inplace=True)
    rat_rev_corr_table=corr_table.join(movie_info)
    rat_rev_corr_table=rat_rev_corr_table[rat_rev_corr_table['count_of_reviews']>=100]
    recommended_movies=rat_rev_corr_table.sort_values(by=['Corrilation','rating','count_of_reviews'],ascending=False).drop(columns=['count_of_reviews'])
    return recommended_movies.head(n=10)

movie_name=input()
print(recommend_movies(movie_name))