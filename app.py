import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

trails=pickle.load(open('trails.pkl', 'rb'))
similarity=pickle.load(open('all_features.pkl', 'rb'))
trail_list=trails['name_area'].values
area_list=trails['area'].unique()

trails['elevation_gain']=round(trails['elevation_gain']*3.28084/1000,1)
trails['length']=trails['length']*0.621371

st.image('yosemite.jpg')
st.header("USA Trail Recommendations")
st.text("Find your next favorite trail! Enter a trail you love, and we'll suggest similar options. Filter by location to explore nearby adventures.")

selected_trail = st.selectbox("Select a trail:", trail_list)
selected_area = st.selectbox("Select an area (optional):", area_list, index=None)

import streamlit.components.v1 as components

# This function to calculate the similarities is adapted from the "Trailforks Recommender" project by Wen Yang: 
# https://towardsdatascience.com/build-trail-recommender-for-trailforks-8ea64b1a2fe4
def get_similar(trail, n=5):


    # compute distance
    target_feature = similarity[trails['name_area']==trail].reshape(1, -1)
    couple_dist = pairwise_distances(similarity,
                                     target_feature, metric='cosine')
    # get similar dataframe
    indices = list(
        map(lambda x: x.item(), np.argsort(couple_dist.ravel())))
    # similar_score
    cosine_similarity = 1 - couple_dist[indices].ravel()

    df_sim_all = pd.DataFrame(
        {"tfidf_index": indices, "similar_score": cosine_similarity})
    #df_sim = df_sim_all[1:n+1]
    output = trails[['name','state_name', 'area_name', 'length', 'elevation_gain', 'difficulty_rating']].iloc[df_sim_all['tfidf_index']]
    if selected_area:
        output = output[output['area_name']==selected_area]
    return output[1:n+1]


selected_df = trails[trails['name_area']==selected_trail]
selected_df = selected_df[['name','state_name', 'area_name', 'length', 'elevation_gain', 'difficulty_rating']]

if st.button('Show Hikes'):
    st.dataframe(selected_df, hide_index=True)
    st.dataframe(get_similar(selected_trail,5), hide_index=True)
   
