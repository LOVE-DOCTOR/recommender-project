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


class AnimeOdyssey:
    def __init__(_self,
                 parquet: str = './data/anime/search_recom.parquet',
                 model_path: str = 'all-mpnet-base-v2',
                 embedding_with_title: str = './data/anime/an_combined_embeddings.pt',

                 ):
        _self.parquet = parquet
        _self.model_path = model_path
        _self.embedding_with_title = embedding_with_title

    @st.experimental_singleton
    def read_parquet_data(_self):
        data = pd.read_parquet(_self.parquet)
        return data

    @st.experimental_singleton
    def read_model(_self):
        model = SentenceTransformer(_self.model_path)
        return model

    @st.experimental_singleton
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
    top_k = min(7, len(df['genre'].tolist()))
    model = AnimeOdyssey().read_model()
    embeddings = AnimeOdyssey().generate_embeddings()

    similarity_index = list()
    score = list()

    for query in queries:
        query_embedding = model.encode(query, convert_to_tensor=True)

        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.cos_sim(query_embedding, embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        for scores, idx in zip(top_results[0], top_results[1]):
            similarity_index.append(int(idx))
            score.append(scores)

    return similarity_index, score


def get_input_similarity(index, words):
    index_sim = {}
    for key, val in words.items():
        sim = words[index].similarity(val)
        index_sim.update({key: sim})

    return dict(sorted(index_sim.items(), key=lambda x: x[1], reverse=True))


def receive_and_process_input():
    anim = AnimeOdyssey()
    data = anim.read_parquet_data()
    data_anime_name = data['Name'].tolist()
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

            rec1, rec2, rec3 = st.columns(3)
            rec4, rec5, rec6 = st.columns(3)

            with rec1:
                st.header(list(first_recommendation['Name'])[0])
                st.image(list(first_recommendation['link'])[0])
                st.write(list(first_recommendation['Genres'])[0])
                st.write(list(first_recommendation['Score'])[0])
                with st.expander('Synopsis'):
                    st.write(list(first_recommendation['sypnopsis'])[0])

            with rec2:
                st.header(list(second_recommendation['Name'])[0])
                st.write(list(second_recommendation['Genres'])[0])
                st.write(list(second_recommendation['Score'])[0])
                with st.expander('Synopsis'):
                    st.write(list(second_recommendation['sypnopsis'])[0])

            with rec3:
                st.header(list(third_recommendation['Name'])[0])
                st.write(list(third_recommendation['Genres'])[0])
                st.write(list(third_recommendation['Score'])[0])
                with st.expander('Synopsis'):
                    st.write(list(third_recommendation['sypnopsis'])[0])

            with rec4:
                st.header(list(fourth_recommendation['Name'])[0])
                st.write(list(fourth_recommendation['Genres'])[0])
                st.write(list(fourth_recommendation['Score'])[0])
                with st.expander('Synopsis'):
                    st.write(list(fourth_recommendation['sypnopsis'])[0])

            with rec5:
                st.header(list(fifth_recommendation['Name'])[0])
                st.write(list(fifth_recommendation['Genres'])[0])
                st.write(list(fifth_recommendation['Score'])[0])
                with st.expander('Synopsis'):
                    st.write(list(fifth_recommendation['sypnopsis'])[0])

            with rec6:
                st.header(list(sixth_recommendation['Name'])[0])
                st.write(list(sixth_recommendation['Genres'])[0])
                st.write(list(sixth_recommendation['Score'])[0])
                with st.expander('Synopsis'):
                    st.write(list(sixth_recommendation['sypnopsis'])[0])


if __name__ == '__main__':
    AnimeOdysseyPage()
