# Importation of required libraries

import json
import pandas as pd
import brotli
import time
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
import torch

def replace_bottom(word):
        word = word.replace('[Written by MAL Rewrite]', '')
        return word
    
class AnimeRecom:
    
    def __init__(self):
        self.anime = './train_data/anime.csv'
        self.scaler = MinMaxScaler()
        
    def read_files(self):
        df = pd.read_csv(self.anime)
        return df
    
    

    def process_files(self):
        print('Processing files...')
        start = time.perf_counter()
        # retrieve the anime dataset from the read_files function
        df = self.read_files()
        
        # Reshape the values to be consistent between 0 and 1
        df = df.assign(
            Score=self.scaler.fit_transform(
                df.Score.values.reshape(-1, 1)
            )
        )
        
        # Select a subset of the attributes
        to_search = df[['Rating', 'Summary', 'Genres', 'Anime Title', 'English', 'MAL Url', 'Score']]
        
        
        # Resetting the index and dropping rows with missing values
        to_search = to_search.reset_index()
        to_search.drop_duplicates(inplace=True)
        to_search.reset_index(inplace=True)
        to_search.drop('index', axis=1, inplace=True)
        print(f'Finished processing in {time.perf_counter() - start}')
        print(' ')
        return to_search
    
    def column_combination(self):
        start = time.perf_counter()
        to_search = self.process_files()
        print('Preprocessing dataset')
        # Joining the three desired columns together
        to_search['genre'] = to_search['Rating']  + "\n" + to_search['Genres'] + '\n' + to_search['Summary']
        to_search['genre_b'] = to_search['Summary'] + '\n' + to_search['Genres'] +'\n' + to_search['Rating']
        to_search['genre_no_rate'] = to_search['Genres'] + '\n' + to_search['Summary']
        # Create a new column for retrieval later on
        to_search['feature'] = to_search['Rating']  + "\n" + to_search['Anime Title'] + "\n" + to_search['Genres'] + '\n' + to_search['Summary']
        
        # Convert all to string
        to_search['genre'] = to_search['genre'].astype(str)
        to_search['genre_b'] = to_search['genre_b'].astype(str)
        
        # Apply the replace bottom to the genre function
        to_search['genre'] = to_search['genre'].apply(replace_bottom)
        to_search.drop('level_0', inplace=True, axis=1)
        to_search.to_parquet('./data/anime/animes.parquet', compression='brotli')
        print(f'Finished preprocessing in {time.perf_counter() - start}')
        print(' ')
        return to_search
    
    def save_index(self, search):
        print('Saving index')
        start = time.perf_counter()
        search.reset_index(inplace=True)
        movies_index = search['index'].tolist()
        dict_bytes = str.encode(str(movies_index))
        compressed = brotli.compress(dict_bytes)

        with open('./data/anime/id.brotli', 'wb') as f:
            f.write(compressed)
        end = time.perf_counter()
        
        print(f'Index saved in {end - start}')
        print(' ')
        
    def save_embeddings(self, search):
        start = time.perf_counter()
        print('Saving embeddings..')
        print('... Loading all-mpnet-base-v2')
        model = SentenceTransformer('all-mpnet-base-v2', device='cuda')
        print('... encoding values')
        embeddings = model.encode(search['genre_b'].tolist(), 
                                  convert_to_tensor=True, 
                                  show_progress_bar=True,
                                  batch_size=32,
                                  device='cuda')
        
        
        torch.save(embeddings, './data/anime/an_combined_embeddings_b.pt')
        
        end = time.perf_counter()
        print(f'Embeddings saved in {end-start}')
        
        
if __name__ == '__main__':
    anim = AnimeRecom()
    df = anim.column_combination()
    anim.save_index(df)
    anim.save_embeddings(df)