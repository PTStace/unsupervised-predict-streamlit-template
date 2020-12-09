"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Welcome","About The App","EDA","Recommender System","Solution Overview","Contact Us","About Us"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.markdown("**Content-based filtering**: uses item features to recommend other items similar to what the user likes, based on their previous actions or explicit feedback")
        st.markdown("**Collaborative filtering**: builds a model from your past behavior (i.e. movies watched or selected by the you) as well as similar decisions made by other users")
        st.write("Describe your winning approach on this page")


    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
    if page_selection == "Welcome":
        st.subheader("==========================================================")
        st.markdown("Welcome To Our Movie Review App")
        st.subheader("==========================================================")
        st.image('resources/imgs/giphy.gif', width=900)
        st.image('resources/imgs/Movie-Show-Gif-960.gif')

    if page_selection == "About The App":
        st.title("About the App")
        st.markdown("Are you a movie lover? Are you tired of wasting your time watching tons of trailers and ending up not watching their movies? Are you tired of finishing your popcorns before you find the right movie? That has come to an end!!")
        st.image(["resources/imgs/tired1.jpg", "resources/imgs/tired22.jpg"],width=300)
        st.markdown("Then we have got the right App for you.")
        st.subheader("How The App Works")
        st.markdown("The Movie Recommender App filters or predicts your preferences based on your favourite or watched movie selections. With just a few clicks, you will select three of your most favourite movies from thousands of movies on the app and you will get top 10 movies you are most likely to enjoy. You have an option to view some data visualizations including word clouds that show the most popular words that appear in movie titles and plots on the most popular genres. The app also contains a contact page, where users of the app can rate our app and give feedback and suggestions. Links to movie sites are also included, so the user has quick and easy to access the recommended movies.")
        st.subheader("Data Description")
        st.markdown("The dataset used for the movie recommender app consists of several million 5-star ratings obtained from users of the online MovieLens movie recommendation service. The data for the MovieLens dataset is maintained by the GroupLens research group in the Department of Computer Science and Engineering at the University of Minnesota. Additional movie content data was legally scraped from IMDB.")

    if page_selection == "EDA":
        st.title("Exploratory Data Analysis")
        st.image(('resources/imgs/counts_graph.png'), use_column_width=True)
        st.image(('resources/imgs/ratings_graph.png'), use_column_width=True)
        st.image(('resources/imgs/counts_ratings.png'), use_column_width=True)

    if page_selection == "Contact Us":
        st.title("Get in touch with us")
        st.markdown('''<span style="color:blue"> **Help us improve this app by rating it. Tell us how to give you a better user experience.** </span>''', unsafe_allow_html=True)
        @st.cache(allow_output_mutation=True)
        def get_data():
            return []
        name = st.text_input("User name")
        inputs = st.text_input("Let us improve your user experience!!!")
        rate = st.slider("Rate us", 0, 5)
        if st.button("Submit"):
            get_data().append({"User name": name, "Suggestion": inputs,"rating":rate})
        st.markdown('''<span style="color:blue"> **What other users said:** </span>''', unsafe_allow_html=True)
        st.write(pd.DataFrame(get_data()))
        st.markdown('''<span style="color:blue"> **For any questions contact us here:** </span>''', unsafe_allow_html=True)
       
    if page_selection == "About Us":
        st.markdown("<h1 style='text-align: center; color: blue;'>About Us</h1>", unsafe_allow_html=True)
        st.markdown("")
        st.info("1. Palesa Hlungwani")
        st.image(('resources/imgs/Palesa.jpg'), use_column_width=True)
        #st.image('resources/imgs/')
        st.markdown("* Github account:PTStace")
        st.markdown("* Kaggle account:palesa_hlungwani")
        st.markdown("* email:ptshlungwani@gmail.com")
        st.markdown("")
        st.info("2. Orline Sorelle Ketcha")
        st.image(('resources/imgs/Orline.jpg'), use_column_width=True)
        #st.image('resources/imgs/')
        st.markdown("* Github account:OrlineSorel")
        st.markdown("* Kaggle account:Sorelle94")
        st.markdown("* email:sorelleketcha@gmail.com")
        st.markdown("")
        st.info("3. Thiyasizwe Kubeka")
        st.image(('resources/imgs/Thiya.jpg'), use_column_width=True)
        #st.image('resources/imgs/')
        st.markdown("* Github account:thiyasizwe_kubeka")
        st.markdown("* Kaggle account:Thiyasizwa_kubeka")
        st.markdown("* email:thiyasizwekubeka@gmail.com")
        st.markdown("")
        st.info("4. Katleho Mokhele")
        #st.image('resources/imgs/')
        st.image(('resources/imgs/Katleho.jpg'), use_column_width=True)
        st.markdown("* Github account:Katness AI")
        st.markdown("* Kaggle account:Katleho Mokhele")
        st.markdown("* email:katleho@southatlantic.net")
        st.markdown("")
        st.info("5. Mfumo Baloyi")
        #st.image(Image.open('resources/imgs/pro wordcloud.jpeg'), caption=None, use_column_width=True)
        st.image(('resources/imgs/Mfumo.jpg'), use_column_width=True)
        #st.image('resources/imgs/')
        st.markdown("* Github account:Mfumoe")
        st.markdown("* Kaggle account:Mfumoe")
        st.markdown("* email:www.baloyimfumoe@gmail.com")






if __name__ == '__main__':
    main()
