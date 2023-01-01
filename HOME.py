import os
import random
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title='DiscoverLab',
    page_icon='recommend.jpg',
    layout='wide',
    initial_sidebar_state="auto"
)


def picture_path(path):
    image_list = [i for i in os.listdir(path)]
    image_names = [Image.open(os.path.join(path, i)) for i in image_list]
    return image_names


def DiscoverLab():
    st.markdown('<style>' + open('styles.css').read() + '</style>', unsafe_allow_html=True)

    repo_url = "https://www.github.com/LOVE-DOCTOR"
    link_text = "Click here to visit my GitHub repository"
    caption = f"GitHub Profile: [{link_text}]({repo_url})"

    st.markdown(
        """
        <div class="main container">
            DISCOVER YOUR NEXT FAVOURITE WITH
        </div>
         <div class="main container2 typed-out">
            DiscoverLab
        </div>
        
        
        """,
        unsafe_allow_html=True
    )

    p1_text = 'DiscoverLab offers three pages of personalized recommendations: an anime recommender system, ' \
              'a movie recommender system, and a music recommender system. All three of these systems use a ' \
              'content-based approach to recommendation, to provide recommendations based on your past viewing or ' \
              'listening history.'
    st.markdown(
        f"""
                <div class="text"><p>{p1_text}</p></div>
                """,
        unsafe_allow_html=True)

    p2_text = "But what exactly is content-based recommendation? In a content-based system, recommendations are " \
              "generated based on the characteristics of the items being recommended, rather than the preferences or " \
              "behavior of other users. For example, if you enter the title of an anime show you've previously " \
              "enjoyed into our anime recommendation system, the system will look for other anime shows with similar " \
              "characteristics (e.g., genre, title, synopsis) and recommend those to you. "
    st.markdown(
        f"""
        <div class="text"><p>{p2_text}</p></div>
        """,
        unsafe_allow_html=True)

    p3_text = "To use the recommendation systems, simply enter the title of an anime, movie, or song that you've " \
              "previously enjoyed, and our system will generate a list of six recommendations for similar content. " \
              "It's that easy! "
    st.markdown(
        f"""
        <div class="text"><p>{p3_text}</p></div>
        """,
        unsafe_allow_html=True)

    sys1, sys2, sys3 = st.columns(3)
    with sys1:
        anime_image_names = picture_path('./anime')
        st.markdown(
            """
        <div class="column-header">AnimeOdyssey</div>
        """, unsafe_allow_html=True)

        st.image(random.choice(anime_image_names))
        anime_odd = "AnimeOdyssey is designed to help you discover new shows and expand your anime horizons. By " \
                    "cross " \
                    "referencing an anime show you've previously enjoyed with the ones stored on our database, " \
                    "the system is able to provide customized recommendations for anime shows you're likely to enjoy. " \
                    "Whether you're a fan of action, romance, or comedy, we've got you covered. "
        st.markdown(
            f"""
            <div class="column-grid">{anime_odd}</div>
            """,
            unsafe_allow_html=True

        )

    with sys2:
        movie_image_names = picture_path('./movie')
        st.markdown(
            """
        <div class="column-header">MovieMatchmaker</div>
        """, unsafe_allow_html=True)
        st.image(random.choice(movie_image_names))
        movie_match = "Are you tired of endlessly scrolling through streaming platforms, trying to find the perfect " \
                      "movie? Look no further! MovieMatchmaker is here to revolutionize your film-finding " \
                      "experience. Our personalized recommendation system uses your preferences to curate a list of " \
                      "movies that are guaranteed to make your movie night a hit. From classic comedies to " \
                      "suspenseful thrillers, we've got a wide range of genres and styles to choose from. So sit " \
                      "back, relax, and let MovieMatchmaker do the work for you! "

        st.markdown(
            f"""
            <div class="column-grid">{movie_match}</div>
            """,
            unsafe_allow_html=True

        )

    with sys3:
        music_image_names = picture_path('./music')
        st.markdown(
            """
        <div class="column-header">SoundScape</div>
        """, unsafe_allow_html=True)
        st.image(random.choice(music_image_names))
        sound_scape = "Are you looking to discover new music that matches your preferences? SoundScape is here to " \
                      "help! Our personalized recommendation system uses your tastes to provide customized " \
                      "recommendations for songs and artists that you'll enjoy. No matter what type of music you like " \
                      "- pop, rock, hip hop, or something else - we have a diverse range of genres and styles to " \
                      "choose from. So let SoundScape help you find your next favorite artist and song! "

        st.markdown(
            f"""
            <div class="column-grid">{sound_scape}</div>
            """,
            unsafe_allow_html=True

        )
    st.caption('Created: Samson Shittu')
    st.caption(caption)
    st.caption('GitHub Repo: ')


DiscoverLab()


