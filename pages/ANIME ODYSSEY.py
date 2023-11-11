import json
from itertools import islice

import pandas as pd
import brotli
import streamlit as st
import hydralit_components as hc
import torch
from sentence_transformers import SentenceTransformer, util

st.set_page_config(
    page_title='AnimeOdyssey',
    page_icon='recommend.jpg',
    layout='wide',
    initial_sidebar_state="auto"
)

st.markdown('<style>' + open('styles.css').read() + '</style>', unsafe_allow_html=True)
st.markdown('<style>' + open('anime-styles.css').read() + '</style>', unsafe_allow_html=True)


class AnimeOdyssey:
    def __init__(_self,
                 parquet: str = './data/anime/animes.parquet',
                 model_path: str = 'all-mpnet-base-v2',
                 embedding_with_title: str = './data/anime/an_combined_embeddings_b.pt',
                 ):
        
        _self.parquet = parquet
        _self.model_path = model_path
        _self.embedding_with_title = embedding_with_title

    @st.cache_data
    def read_parquet_data(_self):
        data = pd.read_parquet(_self.parquet)
        return data

    @st.cache_data
    def read_model(_self):
        model = SentenceTransformer(_self.model_path)
        return model

    @st.cache_data
    def generate_embeddings(_self):
        combined_embeddings = torch.load(_self.embedding_with_title)
        return combined_embeddings


def AnimeOdysseyPage():
    repo_url = "https://www.github.com/LOVE-DOCTOR"
    link_text = "Click here to visit my GitHub repository"
    caption = f"GitHub Profile: [{link_text}]({repo_url})"

    st.title('AnimeOdyssey')
    st.markdown('Input the name of your favourite anime show and get a recommendation of similar animes')
    receive_and_process_input()

    st.caption('Created: Samson Shittu')
    st.caption(caption)
    st.caption('GitHub Repo: ')


def get_index(movie_name):
    df = AnimeOdyssey().read_parquet_data()
    queries = [movie_name]
    top_k = min(16, len(df['genre_b'].tolist()))
    model = AnimeOdyssey().read_model()
    embeddings = AnimeOdyssey().generate_embeddings()

    similarity_index = list()
    score = list()

    for query in queries:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        query_embedding = model.encode(query, convert_to_tensor=True, device=device)
        
        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.cos_sim(query_embedding, embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        for scores, idx in zip(top_results[0], top_results[1]):
            similarity_index.append(int(idx))
            score.append(scores)

    return similarity_index, score


def receive_and_process_input():
    anim = AnimeOdyssey()
    data = anim.read_parquet_data()
    data_anime_name = data['Anime Title'].tolist()
    anime_show = st.selectbox('Name of your anime show: ', data_anime_name)
    st.write(
        """
        <script>
            document.querySelector('select').addEventListener('keydown', function(event) {

                // Filter the options based on the characters that have been typed so far
                var options = Array.from(this.options).filter(function(option) {
                    return option.text.toLowerCase().includes(event.target.value.toLowerCase());
                });

                // Update the options in the dropdown menu
                while (this.options.length > 0) {
                    this.remove(0);
                }
                options.forEach(function(option) {
                    this.add(option);
                }.bind(this));
            });
        </script>
        """,
        unsafe_allow_html=True,
    )
    if anime_show:
        with hc.HyLoader('...', hc.Loaders.pretty_loaders):
            st.write('Warning: Search results may contain sexual content - Hentai')
            index, scores = get_index(anime_show)
            index, scores = index[1:], scores[1:]

            top_6_ind = [int(i) for i in index]

            data_names = data_anime_name.index(anime_show)

            my_search = data[data_names: data_names + 1]
            first_recommendation = data[top_6_ind[0]: top_6_ind[0] + 1]
            second_recommendation = data[top_6_ind[1]: top_6_ind[1] + 1]
            third_recommendation = data[top_6_ind[2]: top_6_ind[2] + 1]
            fourth_recommendation = data[top_6_ind[3]: top_6_ind[3] + 1]
            fifth_recommendation = data[top_6_ind[4]: top_6_ind[4] + 1]
            sixth_recommendation = data[top_6_ind[5]: top_6_ind[5] + 1]
            seventh_recommendation = data[top_6_ind[6]: top_6_ind[6] + 1]
            eighth_recommendation = data[top_6_ind[7]: top_6_ind[7] + 1]
            ninth_recommendation = data[top_6_ind[8]: top_6_ind[8] + 1]
            tenth_recommendation = data[top_6_ind[9]: top_6_ind[9] + 1]
            eleventh_recommendation = data[top_6_ind[10]: top_6_ind[10] + 1]
            twelfth_recommendation = data[top_6_ind[11]: top_6_ind[11] + 1]
            thirteenth_recommendation = data[top_6_ind[12]: top_6_ind[12] + 1]
            fourteenth_recommendation = data[top_6_ind[13]: top_6_ind[13] + 1]
            fifteenth_recommendation = data[top_6_ind[14]: top_6_ind[14] + 1]


            rec1, rec2, rec3 = st.columns(3)
            rec4, rec5, rec6 = st.columns(3)
            rec7, rec8, rec9 = st.columns(3)
            rec10, rec11, rec12 = st.columns(3)
            rec13, rec14, rec15 = st.columns(3)
            

            with rec1:
                st.markdown(
                    f"""
                    <h3>{list(first_recommendation['Anime Title'])[0]}</h3>
                    """, unsafe_allow_html=True
                )
                st.image(list(first_recommendation['MAL Url'])[0])
                st.write(list(first_recommendation['Genres'])[0])
                st.write(list(first_recommendation['Score'])[0])
                with st.expander('Synopsis'):
                    st.write(list(first_recommendation['Summary'])[0])

            with rec2:
                st.markdown(
                    f"""
                    <h3>{list(second_recommendation['Anime Title'])[0]}</h3>
                    """, unsafe_allow_html=True
                )
                st.image(list(second_recommendation['MAL Url'])[0])
                st.write(list(second_recommendation['Genres'])[0])
                st.write(list(second_recommendation['Score'])[0])
                with st.expander('Synopsis'):
                    st.write(list(second_recommendation['Summary'])[0])

            with rec3:
                st.markdown(
                    f"""
                    <h3>{list(third_recommendation['Anime Title'])[0]}</h3>
                    """, unsafe_allow_html=True
                )
                st.image(list(third_recommendation['MAL Url'])[0])
                st.write(list(third_recommendation['Genres'])[0])
                st.write(list(third_recommendation['Score'])[0])
                with st.expander('Synopsis'):
                    st.write(list(third_recommendation['Summary'])[0])

            with rec4:
                st.markdown(
                    f"""
                    <h3>{list(fourth_recommendation['Anime Title'])[0]}</h3>
                    """, unsafe_allow_html=True
                )
                st.image(list(fourth_recommendation['MAL Url'])[0])
                st.write(list(fourth_recommendation['Genres'])[0])
                st.write(list(fourth_recommendation['Score'])[0])
                with st.expander('Synopsis'):
                    st.write(list(fourth_recommendation['Summary'])[0])

            with rec5:
                st.markdown(
                    f"""
                    <h3>{list(fifth_recommendation['Anime Title'])[0]}</h3>
                    """, unsafe_allow_html=True
                )
                st.image(list(fifth_recommendation['MAL Url'])[0])
                st.write(list(fifth_recommendation['Genres'])[0])
                st.write(list(fifth_recommendation['Score'])[0])
                with st.expander('Synopsis'):
                    st.write(list(fifth_recommendation['Summary'])[0])

            with rec6:
                st.markdown(
                    f"""
                    <h3>{list(sixth_recommendation['Anime Title'])[0]}</h3>
                    """, unsafe_allow_html=True
                )
                st.image(list(sixth_recommendation['MAL Url'])[0])
                st.write(list(sixth_recommendation['Genres'])[0])
                st.write(list(sixth_recommendation['Score'])[0])
                with st.expander('Synopsis'):
                    st.write(list(sixth_recommendation['Summary'])[0])
                    
            with rec7:
                st.markdown(
                    f"""
                    <h3>{list(seventh_recommendation['Anime Title'])[0]}</h3>
                    """, unsafe_allow_html=True
                )
                st.image(list(seventh_recommendation['MAL Url'])[0])
                st.write(list(seventh_recommendation['Genres'])[0])
                st.write(list(seventh_recommendation['Score'])[0])
                with st.expander('Synopsis'):
                    st.write(list(seventh_recommendation['Summary'])[0])

            with rec8:
                st.markdown(
                    f"""
                    <h3>{list(eighth_recommendation['Anime Title'])[0]}</h3>
                    """, unsafe_allow_html=True
                )
                st.image(list(eighth_recommendation['MAL Url'])[0])
                st.write(list(eighth_recommendation['Genres'])[0])
                st.write(list(eighth_recommendation['Score'])[0])
                with st.expander('Synopsis'):
                    st.write(list(eighth_recommendation['Summary'])[0])

            with rec9:
                st.markdown(
                    f"""
                    <h3>{list(ninth_recommendation['Anime Title'])[0]}</h3>
                    """, unsafe_allow_html=True
                )
                st.image(list(ninth_recommendation['MAL Url'])[0])
                st.write(list(ninth_recommendation['Genres'])[0])
                st.write(list(ninth_recommendation['Score'])[0])
                with st.expander('Synopsis'):
                    st.write(list(ninth_recommendation['Summary'])[0])

            with rec10:
                st.markdown(
                    f"""
                    <h3>{list(tenth_recommendation['Anime Title'])[0]}</h3>
                    """, unsafe_allow_html=True
                )
                st.image(list(tenth_recommendation['MAL Url'])[0])
                st.write(list(tenth_recommendation['Genres'])[0])
                st.write(list(tenth_recommendation['Score'])[0])
                with st.expander('Synopsis'):
                    st.write(list(tenth_recommendation['Summary'])[0])

            with rec11:
                st.markdown(
                    f"""
                    <h3>{list(eleventh_recommendation['Anime Title'])[0]}</h3>
                    """, unsafe_allow_html=True
                )
                st.image(list(eleventh_recommendation['MAL Url'])[0])
                st.write(list(eleventh_recommendation['Genres'])[0])
                st.write(list(eleventh_recommendation['Score'])[0])
                with st.expander('Synopsis'):
                    st.write(list(eleventh_recommendation['Summary'])[0])

            with rec12:
                st.markdown(
                    f"""
                    <h3>{list(twelfth_recommendation['Anime Title'])[0]}</h3>
                    """, unsafe_allow_html=True
                )
                st.image(list(twelfth_recommendation['MAL Url'])[0])
                st.write(list(twelfth_recommendation['Genres'])[0])
                st.write(list(twelfth_recommendation['Score'])[0])
                with st.expander('Synopsis'):
                    st.write(list(twelfth_recommendation['Summary'])[0])

            with rec13:
                st.markdown(
                    f"""
                    <h3>{list(thirteenth_recommendation['Anime Title'])[0]}</h3>
                    """, unsafe_allow_html=True
                )
                st.image(list(thirteenth_recommendation['MAL Url'])[0])
                st.write(list(thirteenth_recommendation['Genres'])[0])
                st.write(list(thirteenth_recommendation['Score'])[0])
                with st.expander('Synopsis'):
                    st.write(list(thirteenth_recommendation['Summary'])[0])
                    
            with rec14:
                st.markdown(
                    f"""
                    <h3>{list(fourteenth_recommendation['Anime Title'])[0]}</h3>
                    """, unsafe_allow_html=True
                )
                st.image(list(fourteenth_recommendation['MAL Url'])[0])
                st.write(list(fourteenth_recommendation['Genres'])[0])
                st.write(list(fourteenth_recommendation['Score'])[0])
                with st.expander('Synopsis'):
                    st.write(list(fourteenth_recommendation['Summary'])[0])

            with rec15:
                st.markdown(
                    f"""
                    <h3>{list(fifteenth_recommendation['Anime Title'])[0]}</h3>
                    """, unsafe_allow_html=True
                )
                st.image(list(fifteenth_recommendation['MAL Url'])[0])
                st.write(list(fifteenth_recommendation['Genres'])[0])
                st.write(list(fifteenth_recommendation['Score'])[0])
                with st.expander('Synopsis'):
                    st.write(list(fifteenth_recommendation['Summary'])[0])




if __name__ == '__main__':
    AnimeOdysseyPage()
