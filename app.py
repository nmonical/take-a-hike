import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

trails=pickle.load(open('trails.pkl', 'rb'))
similarity=pickle.load(open('all_features.pkl', 'rb'))
trail_list=trails['name'].values

st.header("USA Trail Recommendations")

selected_trail = st.selectbox("Select a trail:", trail_list)

import streamlit.components.v1 as components

def get_similar(trail, n=5):

#idx: target item's index

    # 1. compute distance
    target_feature = similarity[trails['name']==trail].reshape(1, -1)
    couple_dist = pairwise_distances(similarity,
                                     target_feature, metric='cosine')
    # 2. get similar dataframe: no need to filter out the first
    # because the first won't be the unseen url
    indices = list(
        map(lambda x: x.item(), np.argsort(couple_dist.ravel())))
    # similar_score
    cosine_similarity = 1 - couple_dist[indices].ravel()

    df_sim_all = pd.DataFrame(
        {"tfidf_index": indices, "similar_score": cosine_similarity})
    
    df_sim = df_sim_all[1:n+1]
    return trails[['name','state_name', 'area_name', 'length', 'elevation_gain', 'difficulty_rating']].iloc[df_sim['tfidf_index']]

#if st.button('Show Hikes'):
 #   hike_name, hike_state, area_name = get_similar(selected_trail, n=5)['name'].tolist(), get_similar(selected_trail, n=5)['state_name'].tolist(), get_similar(selected_trail, n=5)['area_name'].tolist()
  #  col0, col1, col2, col3, col4, col5 = st.columns(6)
#    with col0:
 #       st.text('Name')
  #      st.text('State')
   # with col1:
        #st.write(hike_name[0]  % "https://alltrails.com/trail/"+slug[0])
        #st.markdown(hike_name[0]  % "https://alltrails.com/trail/"+slug[0])
    #    st.text(hike_name[0])
     #   st.text(hike_state[0])
      #  st.text(area_name[0])
    #with col2:
     #   st.text(hike_name[1])
      #  st.text(hike_state[1])
       # st.text(area_name[1])
    #with col3:
     #   st.text(hike_name[2])
      #  st.text(hike_state[2])
       # st.text(area_name[2])
    #with col4:
     #   st.text(hike_name[3])
      #  st.text(hike_state[3])
       # st.text(area_name[3])
    #with col5:
     #   st.text(hike_name[4])
      #  st.text(hike_state[4])
       # st.text(area_name[4])
selected_df = trails[trails['name']==selected_trail]
selected_df = selected_df[['name','state_name', 'area_name', 'length', 'elevation_gain', 'difficulty_rating']]

if st.button('Show Hikes'):
    st.dataframe(selected_df, hide_index=True)
    st.dataframe(get_similar(selected_trail,5), hide_index=True)
   
