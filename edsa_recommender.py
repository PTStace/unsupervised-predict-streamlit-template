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
rating_m = pd.read_csv('resources/data/ratings.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Welcome","About The App","EDA","Recommender System","Search for a movie","Solution Overview","Contact Us","About Us"]

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
        st.markdown("<h1 style='text-align: left; color: black;'>Content-based filtering: uses item features to recommend other items similar to what the user likes, based on their previous actions or explicit feedback</h1>", unsafe_allow_html=True)
        st.image(('resources/imgs/content-based.png'), use_column_width=True)
        st.markdown("<h1 style='text-align: left; color: black;'>Collaborative filtering: builds a model from your past behavior (i.e. movies watched or selected by the you) as well as similar decisions made by other users</h1>", unsafe_allow_html=True)
        st.image(('resources/imgs/collaborative filtering.png'), use_column_width=True)
        st.write("Describe your winning approach on this page")

    if page_selection == "Search for a movie":
        st.title("Search for Movies")
        st.image(('resources/imgs/franchises.jpg'), use_column_width=True)
        st.markdown('Please Refer to the About Machine Learning Page to learn more about the techniques used to recommend movies. If you decide not to use the recommender systems you can use this page to filter movies based on the rating of the movie , the year in which the movie was released and the genre of the movies. After you change the filter you will be left with movies that are specific to that filter used. Then when you scroll down you will see the movie name and the link to a youtube trailer of that movie. When you click the link ,you will see a page on youtube for that specific movie and you can watch the trailer and see if you like it. This is an alternative method to you if you are not satisfied with the recommender engine . Enjoy! ', unsafe_allow_html=True)
        # Movies
        df = pd.read_csv('resources/data/movies.csv')

        
        def explode(df, lst_cols, fill_value='', preserve_index=False):
            import numpy as np
             # make sure `lst_cols` is list-alike
            if (lst_cols is not None
                    and len(lst_cols) > 0
                    and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
                lst_cols = [lst_cols]
            # all columns except `lst_cols`
            idx_cols = df.columns.difference(lst_cols)
            # calculate lengths of lists
            lens = df[lst_cols[0]].str.len()
            # preserve original index values    
            idx = np.repeat(df.index.values, lens)
            # create "exploded" DF
            res = (pd.DataFrame({
                        col:np.repeat(df[col].values, lens)
                        for col in idx_cols},
                        index=idx)
                    .assign(**{col:np.concatenate(df.loc[lens>0, col].values)
                            for col in lst_cols}))
            # append those rows that have empty lists
            if (lens == 0).any():
                # at least one list in cells is empty
                res = (res.append(df.loc[lens==0, idx_cols], sort=False)
                            .fillna(fill_value))
            # revert the original index order
            res = res.sort_index()   
            # reset index if requested
            if not preserve_index:        
                res = res.reset_index(drop=True)
            return res 
        movie_data = pd.merge(rating_m, df, on='movieId')
        movie_data['year'] = movie_data.title.str.extract('(\(\d\d\d\d\))',expand=False)
        #Removing the parentheses
        movie_data['year'] = movie_data.year.str.extract('(\d\d\d\d)',expand=False)

        movie_data.genres = movie_data.genres.str.split('|')
        movie_rating = st.sidebar.number_input("Pick a rating ",0.5,5.0, step=0.5)

        movie_data = explode(movie_data, ['genres'])
        movie_title = movie_data['genres'].unique()
        title = st.selectbox('Genre', movie_title)
        movie_data['year'].dropna(inplace = True)
        movie_data = movie_data.drop(['movieId','timestamp','userId'], axis = 1)
        year_of_movie_release = movie_data['year'].sort_values(ascending=False).unique()
        release_year = st.selectbox('Year', year_of_movie_release)

        movie = movie_data[(movie_data.rating == movie_rating)&(movie_data.genres == title)&(movie_data.year == release_year)]
        df = movie.drop_duplicates(subset = ["title"])
        if len(df) !=0:
            st.write(df)
        if len(df) ==0:
            st.write('We have no movies for that rating!')        
        def youtube_link(title):
    
            """This function takes in the title of a movie and returns a Search query link to youtube
    
            INPUT: ('Avengers age of ultron')
            -----------
    
            OUTPUT: https://www.youtube.com/results?search_query=The+little+Mermaid&page=1
            ----------
            """
            title = title.replace(' ','+')
            base = "https://www.youtube.com/results?search_query="
            q = title
            page = "&page=1"
            URL = base + q + page
            return URL            
        if len(df) !=0:           
            for _, row in df.iterrows():
                st.write(row['title'])
                st.write(youtube_link(title = row['title']))



    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
    if page_selection == "Welcome":
        st.subheader("==========================================================")
        st.markdown('''<span style="color:black"> **Welcome To Our Movie Review App.** </span>''', unsafe_allow_html=True)
        st.subheader("==========================================================")
        st.image('resources/imgs/giphy.gif', use_column_width=True)
        st.image('resources/imgs/Movie-Show-GIF-960.gif', use_column_width=True)

    if page_selection == "About The App":
        st.title("About the App")
        st.markdown("Are you a movie lover? Are you tired of wasting your time watching tons of trailers and ending up not watching their movies? Are you tired of finishing your popcorns before you find the right movie? That has come to an end!!")
        st.image(["resources/imgs/Tired 1.1.gif", "resources/imgs/Tired 1.2.jpg"],use_column_width=True)
        st.markdown("Then we have got the right App for you.")
        st.subheader("How The App Works")
        st.markdown("The Movie Recommender App filters or predicts your preferences based on your favourite or watched movie selections. With just a few clicks, you will select three of your most favourite movies from thousands of movies on the app and you will get top 10 movies you are most likely to enjoy. You have an option to view some data visualizations including word clouds that show the most popular words that appear in movie titles and plots on the most popular genres. The app also contains a contact page, where users of the app can rate our app and give feedback and suggestions. Links to movie sites are also included, so the user has quick and easy to access the recommended movies.")
        st.subheader("Data Description")
        st.markdown("The dataset used for the movie recommender app consists of several million 5-star ratings obtained from users of the online MovieLens movie recommendation service. The data for the MovieLens dataset is maintained by the GroupLens research group in the Department of Computer Science and Engineering at the University of Minnesota. Additional movie content data was legally scraped from IMDB.")

    if page_selection == "EDA":
        st.title("Exploratory Data Analysis")
        st.subheader("All Time Popular Movies By Ratings Insights")
        st.markdown("The graph shows all the movies that have been rated the most for all movies in the dataset. The most popular can be seen as a 1994 movie Shawshank Redemption with a rating count of more than 30 thousand.,")
        st.image(('resources/imgs/all time popular movies by ratings.png'), use_column_width=True)
        st.subheader("Released Movies Per Year Insights")
        st.markdown("The graph shows the number of movies that have been released each year from 1971 to 2017. It is safe to note that there has been an exponential increase in the number of movies released each year. we can go as far as to say that movies released in the 2010s are about 4 times the number of those that were released in the 1070s.")
        st.image(('resources/imgs/total movies released per year.png'), use_column_width=True)
        st.subheader("Popular Genres By Rating Insights")
        st.markdown("The treemap depicts the genres that are most popular to the least popular in terms of ratings. From the treemap it can be seen that the most rated genre happens to be Drama, followed by Comedy with IMAX the least rated.")
        st.image(('resources/imgs/popular genres.png'), use_column_width=True)
        st.subheader("Percentage Of Users Per Ratings Insights")
        st.markdown("The graphs shows the total number of user percentage based on their ratings. Most users rated movies with a rating of 4.0(26.53%)")
        st.image(('resources/imgs/percentage of users per rating.png'), use_column_width=True)
        st.subheader("Popular Actors/Actresses Insights")
        st.markdown("The graph shows who the most popular actors/actresses appearing in the movies are and the number of movies they appear in. The actor Samuel L Jackson is the most popular, appearing in more than 80 movies. Actress Julianne Moore is the most popular amoung the actresses.")
        st.image(('resources/imgs/popular actors.png'), use_column_width=True)
        st.subheader("Rating Distribution")
        st.markdown("This graph shows how ratings are distributed. Just like in, Percentage of Users Per Ratings Insights, most movies have a rating of 4.0")
        st.image(('resources/imgs/rating distribution.png'), use_column_width=True)
        st.subheader("Movie Runtime Ditribution")
        st.markdown("Below we can a see a distribution of movie runtime. Majority of the movies have a 100 minutes runtime.")
        st.image(('resources/imgs/runtime distribution.png'), use_column_width=True)
        st.subheader("Popular Movies Wordcloud")
        st.markdown("The wordcloud below shows the letters appearing the most in the movie titles. Love, Girl, Man and Night are words that appear the biggest. This means that more movies have such words in their title. It makes sense that these words appear the most as we have more movies in the drama, comedy and romance genre.")
        st.image(('resources/imgs/most popular movies wordcloud.png'), use_column_width=True)
        st.subheader("Popular Movie Directors")
        st.markdown("The graph below shows the most popular movie directors. It makes sense for Woody Allen to be the most popular director as the first movie he directed was in 1965 and since then he has directed about 50 movies with the latest released in 2020. Woody has also been acting since 1965 to date, he has directed 3 short films and directed about 12 tv shows.")
        st.image(('resources/imgs/popular movie directors.png'), use_column_width=True)
        st.subheader("Average Budget Per Genre")
        st.markdown("The graph below shows the average budget used for movies in each genre. The War genre seems to have the highest average budget, with the Documentary genre having the least budget.")
        st.image(('resources/imgs/average budget per genre.png'), use_column_width=True)
        st.subheader("Average Runtime Per Genre")
        st.markdown("Western genre has the highest average movie runtime at about 120 minutes. Animation genre has the least average runtime at about 76-77 minutes.")
        st.image(('resources/imgs/average runtime per genre.png'), use_column_width=True)
        st.subheader("Top 20 Popular Movies From 2010")
        st.markdown("The 2014 movie Intersteller had the highest ratings. The movie only had a budget of $165 million but went to make $696.3 million from the box office. The movie also bagged 6 awards including an Academy award for best visual effects.")
        st.image(('resources/imgs/top 20 popular movies by ratings from 2010.png'), use_column_width=True)
        

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
        st.markdown("* Github account:PTStace")
        st.markdown("* Kaggle account:palesa_hlungwani")
        st.markdown("* email:ptshlungwani@gmail.com")
        st.markdown("")

        st.info("2. Orline Sorelle Ketcha")
        st.image(('resources/imgs/Orline.jpg'), use_column_width=True)
        st.markdown("* Github account:OrlineSorel")
        st.markdown("* Kaggle account:Sorelle94")
        st.markdown("* email:sorelleketcha@gmail.com")
        st.markdown("")

        st.info("3. Thiyasizwe Kubeka")
        st.image(('resources/imgs/Thiya.jpg'), use_column_width=True)
        st.markdown("* Github account:thiyasizwe_kubeka")
        st.markdown("* Kaggle account:Thiyasizwa_kubeka")
        st.markdown("* email:thiyasizwekubeka@gmail.com")
        st.markdown("")

        st.info("4. Katleho Mokhele")
        st.image(('resources/imgs/Katleho.jpg'), use_column_width=True)
        st.markdown("* Github account:Katness AI")
        st.markdown("* Kaggle account:Katleho Mokhele")
        st.markdown("* email:katleho@southatlantic.net")
        st.markdown("")

        st.info("5. Mfumo Baloyi")
        st.image(('resources/imgs/Mfumo.jpg'), use_column_width=True)
        st.markdown("* Github account:MfumoB")
        st.markdown("* Kaggle account:Mfumoe")
        st.markdown("* email:www.baloyimfumoe@gmail.com")






if __name__ == '__main__':
    main()
