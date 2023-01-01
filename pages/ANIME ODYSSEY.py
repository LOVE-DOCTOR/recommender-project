import json
from itertools import islice

import brotli
import pandas as pd
import spacy
from spacy.tokens import DocBin
import streamlit as st
import hydralit_components as hc

st.set_page_config(
    page_title='AnimeOdyssey',
    page_icon='recommend.jpg',
    layout='wide',
    initial_sidebar_state="auto"
)

st.markdown('<style>' + open('styles.css').read() + '</style>', unsafe_allow_html=True)


class AnimeOdyssey:
    def __init__(_self,
                 csv: str = './data/anime/search_recom.csv',
                 spacy: str = 'en_core_web_lg',
                 dictionary: str = './data/anime/saved_dict.brotli',
                 combined_features: str = './data/anime/combined_features.spacy'
                 ):
        _self.csv = csv
        _self.spacy = spacy
        _self.dictionary = dictionary
        _self.combined_features = combined_features

    @st.experimental_singleton
    def read_csv_data(_self):
        data = pd.read_csv(_self.csv)
        data['lower_names'] = [i.lower() for i in data['Name']]
        return data

    @st.experimental_singleton
    def read_spacy_model(_self):
        nlp = spacy.load(_self.spacy)
        return nlp

    @st.experimental_singleton
    def read_genre_doc_tokens(_self):
        nlp = _self.read_spacy_model()
        docbin = DocBin().from_disk(_self.combined_features)
        return list(docbin.get_docs(nlp.vocab))

    @st.experimental_singleton
    def read_dictionary_data(_self):
        with open(_self.dictionary, "rb") as f:
            compressed_dict = f.read()
            byte_string = brotli.decompress(compressed_dict)

        json_string = byte_string.decode()
        my_dict = json.loads(json_string)
        new_dict_values = _self.read_genre_doc_tokens()
        updated_dict = {int(k): v for k, v in zip(my_dict.keys(), new_dict_values)}

        return updated_dict


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


def get_index_genres(movie_name):
    data = AnimeOdyssey().read_csv_data()

    try:
        return int(data[data['Name'] == movie_name]['index'])
    except TypeError:
        st.write('Alternate')
        return int(data[data['lower_names'] == movie_name]['index'])


def get_input_similarity(index, words):
    index_sim = {}
    for key, val in words.items():
        sim = words[index].similarity(val)
        index_sim.update({key: sim})

    return dict(sorted(index_sim.items(), key=lambda x: x[1], reverse=True))


def receive_and_process_input():
    anim = AnimeOdyssey()
    data = anim.read_csv_data()
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

            data_dict = anim.read_dictionary_data()
            index = get_index_genres(anime_show)
            similarities_dict = get_input_similarity(index, data_dict)
            top_100_recommendations_sort = islice(similarities_dict.keys(), 100)

            check_df = pd.DataFrame()
            check_df['movie_index'] = [i for i in top_100_recommendations_sort]
            newer = []
            for i in check_df['movie_index']:
                if i in list(data['index']):
                    score = data[data['index'] == i]
                    score = list(score['Score'])
                    newer.append(score[0])

            top_recommendations = list(check_df['movie_index'])
            top_recommendations = top_recommendations[1:]
            newer = newer[1:]

            recommend_dict = {top_recommendations[i]: newer[i] for i in range(len(newer))}
            recommend_dict_sorted = dict(sorted(recommend_dict.items(), key=lambda x: x[1], reverse=True))
            recommend_dict_list = [i for i in recommend_dict_sorted.keys()]

            top_6_ind = [recommend_dict_list[i] for i in range(6)]

            first_recommendation = data[data['index'] == top_6_ind[0]]
            second_recommendation = data[data['index'] == top_6_ind[1]]
            third_recommendation = data[data['index'] == top_6_ind[2]]
            fourth_recommendation = data[data['index'] == top_6_ind[3]]
            fifth_recommendation = data[data['index'] == top_6_ind[4]]
            sixth_recommendation = data[data['index'] == top_6_ind[5]]

            rec1, rec2, rec3 = st.columns(3)
            rec4, rec5, rec6 = st.columns(3)

            with rec1:
                st.header(list(first_recommendation['Name'])[0])
                st.write(list(first_recommendation['Genres'])[0])
                st.write(f'Rating: {recommend_dict[top_6_ind[0]]}')
                with st.expander('Synopsis'):
                    st.write(list(first_recommendation['sypnopsis'])[0])

            with rec2:
                st.header(list(second_recommendation['Name'])[0])
                st.write(list(second_recommendation['Genres'])[0])
                st.write(f'Rating: {recommend_dict[top_6_ind[1]]}')
                with st.expander('Synopsis'):
                    st.write(list(second_recommendation['sypnopsis'])[0])

            with rec3:
                st.header(list(third_recommendation['Name'])[0])
                st.write(list(third_recommendation['Genres'])[0])
                st.write(f'Rating: {recommend_dict[top_6_ind[2]]}')
                with st.expander('Synopsis'):
                    st.write(list(third_recommendation['sypnopsis'])[0])

            with rec4:
                st.header(list(fourth_recommendation['Name'])[0])
                st.write(list(fourth_recommendation['Genres'])[0])
                st.write(f'Rating: {recommend_dict[top_6_ind[3]]}')
                with st.expander('Synopsis'):
                    st.write(list(fourth_recommendation['sypnopsis'])[0])

            with rec5:
                st.header(list(fifth_recommendation['Name'])[0])
                st.write(list(fifth_recommendation['Genres'])[0])
                st.write(f'Rating: {recommend_dict[top_6_ind[4]]}')
                with st.expander('Synopsis'):
                    st.write(list(fifth_recommendation['sypnopsis'])[0])

            with rec6:
                st.header(list(sixth_recommendation['Name'])[0])
                st.write(list(sixth_recommendation['Genres'])[0])
                st.write(f'Rating: {recommend_dict[top_6_ind[5]]}')
                with st.expander('Synopsis'):
                    st.write(list(sixth_recommendation['sypnopsis'])[0])


if __name__ == '__main__':
    AnimeOdysseyPage()
