import streamlit as st
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import gdown
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
final_data = pd.read_csv('dataset.csv')

final_data = final_data[final_data.state != 'No Info']
final_data['state'] = final_data['state'].str.title()
final_data['city'] = final_data['city'].str.title()

# Define the Google Drive file IDs
glove_vectors_id = '1TF69O5cRt7uJVI3KLGP3yNt7j5XbElkw'
corpus_embeddings_id = '1LMWsY9iExrAW1l3OZQAdNAQCEOAUZX95'

# Download GloVe vectors from Google Drive
gdown.download('https://drive.google.com/uc?id=' + glove_vectors_id, 'glove_vectors.model', quiet=False)

# Download corpus embeddings from Google Drive
gdown.download('https://drive.google.com/uc?id=' + corpus_embeddings_id, 'corpus_embeddings.pkl', quiet=False)

# Load GloVe vectors
with open('glove_vectors.model', 'rb') as f:
    glove_vectors = pickle.load(f)

# Load corpus embeddings
with open('corpus_embeddings.pkl', 'rb') as f:
    corpus_embeddings = pickle.load(f)

# Preprocessing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
num_features = 300

def get_average_glove(tokens, model, num_features):
    valid_words = [word for word in tokens if word in model]
    if not valid_words:
        return np.zeros(num_features)
    return np.mean([model[word] for word in valid_words], axis=0)

# Streamlit app
st.title('Job Recommendation System')

# Dropdown for category selection
state_option = final_data['state'].unique()
state_option = np.sort(state_option)
type_options = final_data['jenis_pekerjaan'].unique()
sponsor_options = final_data['disponsori'].unique()
apply_options = final_data['tipe_pendaftaran'].unique()

state_option = np.concatenate([np.array(['Any']), state_option])
type_options = np.concatenate([np.array(['Any']), type_options])
sponsor_options = np.concatenate([np.array(['Any']), sponsor_options])
apply_options = np.concatenate([np.array(['Any']), apply_options])

selected_state = st.selectbox('Select State', state_option)
if selected_state == 'Any':
    city_option = ['Any']
else:
    city_option = final_data[final_data['state'] == selected_state]['city'].unique()
    city_option = np.concatenate([np.array(['Any']), city_option])

city_option = np.sort(city_option)

selected_city = st.selectbox('Select City', city_option)
selected_type = st.selectbox('Select Job Type', type_options)
selected_sponsor = st.selectbox('Select Sponsor Type', sponsor_options)
selected_apply = st.selectbox('Select Application Type', apply_options)

city_filter = (final_data['city'] == selected_city) if selected_state != 'Any' else (final_data['city'] == final_data['city'])
type_filter = (final_data['jenis_pekerjaan'] == selected_type) if selected_type != 'Any' else (final_data['jenis_pekerjaan'] == final_data['jenis_pekerjaan'])
sponsor_filter = (final_data['disponsori'] == selected_sponsor) if selected_sponsor != 'Any' else (final_data['disponsori'] == final_data['disponsori'])
apply_filter = (final_data['tipe_pendaftaran'] == selected_apply) if selected_apply != 'Any' else (final_data['tipe_pendaftaran'] == final_data['tipe_pendaftaran'])

# Text box to input new skill description
new_text = st.text_area('Enter New Skill Description')

if st.button('Search Jobs'):
    if new_text:
        # Filter data based on selected categories
        filtered_data = final_data[(city_filter) & (type_filter) & (sponsor_filter) & (apply_filter)]
        
        if filtered_data.shape[0] < 3:
            st.write('Too Many Filters, Reduce Filters!')
        else:
            filtered_indices = filtered_data.index
            filtered_embeddings = corpus_embeddings[filtered_indices]

            # Preprocess and embed the new input text
            input_tokens = word_tokenize(new_text.lower())
            input_tokens = [lemmatizer.lemmatize(word) for word in input_tokens if word not in stop_words]
            input_embedding = get_average_glove(input_tokens, glove_vectors, num_features)

            # Compute cosine similarities
            cosine_sim_new = cosine_similarity([input_embedding], filtered_embeddings).flatten()

            # Get top 10 similar jobs
            top_3_indices = cosine_sim_new.argsort()[-3:][::-1]
            top_3_titles = filtered_data.iloc[top_3_indices]['url_posting_pekerjaan']

            isBreak = False
            for i, index in enumerate(top_3_indices, 1):
                if cosine_sim_new[index] < 0.6:
                    st.write('Your Skill Description is Too Short, Add More Details!')
                    isBreak = True
                    break
            
            if not isBreak:
                st.write("Top 3 Jobs for You:")
                for i, title in enumerate(top_3_titles, 1):
                    st.write(f"{i}. {title}.")
            
    else:
        st.write("Enter a skill description to search for matching jobs.")
