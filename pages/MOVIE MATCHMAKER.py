import streamlit as st
import pandas as pd
import hydralit_components as hc
import torch
from sentence_transformers import SentenceTransformer, util
from streamlit_searchbox import st_searchbox

st.set_page_config(
    page_title='MovieMatchmaker',
    layout='wide',
    initial_sidebar_state="auto"
)

st.markdown('<style>' + open('styles.css').read() + '</style>', unsafe_allow_html=True)


class MovieMatchmaker:

    def __init__(_self,
                 parquet: str = './data/movies/movies.parquet',
                 model_path: str = 'all-mpnet-base-v2',
                 embedding_with_title: str = './data/movies/combined_embeddings.pt',
                 embedding_without_title: str = './data/movies/combined2_embeddings.pt'

                 ):
        _self.parquet = parquet
        _self.model_path = model_path
        _self.embedding_with_title = embedding_with_title
        _self.embedding_without_title = embedding_without_title

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
        combined2_embeddings = torch.load(_self.embedding_without_title)
        return combined_embeddings, combined2_embeddings


def MovieMatchmakerPage():
    repo_url = "https://www.github.com/LOVE-DOCTOR"
    link_text = "Click here to visit my GitHub repository"
    caption = f"GitHub Profile: [{link_text}]({repo_url})"

    st.title('MovieMatchmaker')
    st.markdown('Input the name of your favourite movie and get a recommendation of similar movies')
    receive_and_process_input()

    st.caption('Created: Samson Shittu Babatunde')
    st.caption(caption)
    st.caption('GitHub Repo: ')


def get_index(movie_name):
    df = MovieMatchmaker().read_parquet_data()
    queries = [movie_name]
    top_k = min(7, len(df['combined'].tolist()))
    model = MovieMatchmaker().read_model()
    embeddings = MovieMatchmaker().generate_embeddings()[0]

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


def receive_and_process_input():
    data = MovieMatchmaker().read_parquet_data()
    data_movie_names = data['original_title'].tolist()

    data = data.reset_index()

    movie_name = st.selectbox('Movie name: ', data_movie_names)
    # movie_name = st_searchbox(data_movie_names, key='movie_name_searchbox')
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
    if movie_name:
        with hc.HyLoader('...', hc.Loaders.pretty_loaders):
            img_first_url = 'https://image.tmdb.org/t/p/w400'
            index, scores = get_index(movie_name)
            index, scores = index[1:], scores[1:]

            top_6_ind = [int(i) for i in index]

            data_names = data_movie_names.index(movie_name)

            my_search = data[data_names: data_names + 1]
            first_recommendation = data[top_6_ind[0]: top_6_ind[0] + 1]
            second_recommendation = data[top_6_ind[1]: top_6_ind[1] + 1]
            third_recommendation = data[top_6_ind[2]: top_6_ind[2] + 1]
            fourth_recommendation = data[top_6_ind[3]: top_6_ind[3] + 1]
            fifth_recommendation = data[top_6_ind[4]: top_6_ind[4] + 1]
            sixth_recommendation = data[top_6_ind[5]: top_6_ind[5] + 1]

            sea1, sea2, sea3 = st.columns(3)
            with sea1:
                img_search_url = f'{img_first_url}{list(my_search["poster_path"])[0]}'
                st.write(f'<div class="search-head">Hello</div>',unsafe_allow_html=True)
                st.markdown(f"""
                                <h3>{list(my_search['original_title'])[0]}</h3>
                                """, unsafe_allow_html=True)
                st.image(img_search_url)
                with st.expander('Overview'):
                    st.write(list(my_search['overview'])[0])

            st.write('<div class="search-head"><h3>RESULTS</h3></div>', unsafe_allow_html=True)
            rec1, rec2, rec3 = st.columns(3)
            rec4, rec5, rec6 = st.columns(3)

            with rec1:
                image_url0 = f'{img_first_url}{list(first_recommendation["poster_path"])[0]}'
                st.markdown(f"""
                <h3>{list(first_recommendation['original_title'])[0]}</h3>
                """, unsafe_allow_html=True)
                st.image(image_url0)
                st.write(list(first_recommendation['genre'])[0])
                st.write('Original Language: ', list(first_recommendation['original_language'])[0])
                st.write('Media: ', list(first_recommendation['media_type'])[0].capitalize())
                st.write('Adult: ', list(first_recommendation['adult'])[0])
                st.write(f'Rating: {list(first_recommendation["vote_average"])[0]}')
                with st.expander('Overview'):
                    st.write(list(first_recommendation['overview'])[0])

            with rec2:
                image_url1 = f'{img_first_url}{list(second_recommendation["poster_path"])[0]}'
                st.markdown(f"""
                                <h3>{list(second_recommendation['original_title'])[0]}</h3>
                                """, unsafe_allow_html=True)
                st.image(image_url1)
                st.write(list(second_recommendation['genre'])[0])
                st.write('Original Language: ', list(second_recommendation['original_language'])[0])
                st.write('Media: ', list(second_recommendation['media_type'])[0].capitalize())
                st.write('Adult: ', list(second_recommendation['adult'])[0])
                st.write(f'Rating: {list(second_recommendation["vote_average"])[0]}')
                with st.expander('Overview'):
                    st.write(list(second_recommendation['overview'])[0])

            with rec3:
                image_url2 = f'{img_first_url}{list(third_recommendation["poster_path"])[0]}'
                st.markdown(f"""
                                <h3>{list(third_recommendation['original_title'])[0]}</h3>
                                """, unsafe_allow_html=True)
                st.image(image_url2)
                st.write(list(third_recommendation['genre'])[0])
                st.write('Original Language: ', list(third_recommendation['original_language'])[0])
                st.write('Media: ', list(third_recommendation['media_type'])[0].capitalize())
                st.write('Adult: ', list(third_recommendation['adult'])[0])
                st.write(f'Rating: {list(third_recommendation["vote_average"])[0]}')
                with st.expander('Overview'):
                    st.write(list(third_recommendation['overview'])[0])

            with rec4:
                image_url3 = f'{img_first_url}{list(fourth_recommendation["poster_path"])[0]}'
                st.markdown(f"""
                                <h3>{list(fourth_recommendation['original_title'])[0]}</h3>
                                """, unsafe_allow_html=True)
                st.image(image_url3)
                st.write(list(fourth_recommendation['genre'])[0])
                st.write('Original Language: ', list(fourth_recommendation['original_language'])[0])
                st.write('Media: ', list(fourth_recommendation['media_type'])[0].capitalize())
                st.write('Adult: ', list(fourth_recommendation['adult'])[0])
                st.write(f'Rating: {list(fourth_recommendation["vote_average"])[0]}')
                with st.expander('Overview'):
                    st.write(list(fourth_recommendation['overview'])[0])

            with rec5:
                image_url4 = f'{img_first_url}{list(fifth_recommendation["poster_path"])[0]}'
                st.markdown(f"""
                                <h3>{list(fifth_recommendation['original_title'])[0]}</h3>
                                """, unsafe_allow_html=True)
                st.image(image_url4)
                st.write(list(fifth_recommendation['genre'])[0])
                st.write('Original Language: ', list(fifth_recommendation['original_language'])[0])
                st.write('Media: ', list(fifth_recommendation['media_type'])[0].capitalize())
                st.write('Adult: ', list(fifth_recommendation['adult'])[0])
                st.write(f'Rating: {list(fifth_recommendation["vote_average"])[0]}')
                with st.expander('Overview'):
                    st.write(list(fifth_recommendation['overview'])[0])

            with rec6:
                image_url5 = f'{img_first_url}{list(sixth_recommendation["poster_path"])[0]}'
                st.markdown(f"""
                                <h3>{list(sixth_recommendation['original_title'])[0]}</h3>
                                """, unsafe_allow_html=True)
                st.image(image_url5)
                st.write(list(sixth_recommendation['genre'])[0])
                st.write('Original Language: ', list(sixth_recommendation['original_language'])[0])
                st.write('Media: ', list(sixth_recommendation['media_type'])[0].capitalize())
                st.write('Adult: ', list(sixth_recommendation['adult'])[0])
                st.write(f'Rating: {list(sixth_recommendation["vote_average"])[0]}')
                with st.expander('Overview'):
                    st.write(list(sixth_recommendation['overview'])[0])


if __name__ == '__main__':
    MovieMatchmakerPage()
