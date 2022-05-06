import streamlit as st
import matplotlib.pyplot as plt
# from tensorflow.keras import layers, Sequential, models
# import pickle as pkl
from PIL import Image
from lib_func import embedding,TransformerBlock # embed_sentence_with_TF,
# from tensorflow.keras.backend import expand_dims
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import gensim.downloader as api
# from gensim.models import KeyedVectors

from tensorflow.keras.layers import Dropout  #MultiHeadAttention,LayerNormalization,Layer
from tensorflow.keras.layers import Input, GlobalAveragePooling1D, Dense #Embedding
from tensorflow.keras.models import Model # Sequential,
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


image = Image.open('Demo_Project_NLP_sentiment_analysis_benbhk/top.png')

# st.set_page_config(layout="wide")

st.image(image)


def img_normalizer(X):
    return X/255-0.5

st.title(" Text sentiment analysis ")
st.subheader(" Benjamin Barre - Demo Project ")
st.write('In this project, I uses the Tesorflow.Keras library with an attention based Transformer to analyze the sentiment of a text. This model uses a transfer learning from the gensim-Word2Vec library (glove-wiki-gigaword).')
st.write('It may takes severals minutes to starts (cheap deployment as it is just a demonstration model)')
st.write('You can check the code on github.')
# st.write('In this project, I confront three models to analyze the sentiment of a text. Each model uses a transfer learning from the gensim-Word2Vec library (glove-wiki-gigaword).')
# st.write('- The first model uses the XGBoost library')
# st.write('- The second model uses the classical Tesorflow.Keras library (Adam optimization)')
# st.write('- The third model also uses the Tesorflow.Keras library with an attention based Transformer (Adam optimization as well)')
st.write("Link to the [github repo](https://github.com/Benjaminbhk/Demo_Project_NLP_sentiment_analysis_benbhk.git)")

st.markdown('''---''')

if 'vectors_reloaded' not in st.session_state:
    st.markdown(
        '''Loading of the Word2Vec model (glove-wiki-gigaword-200). Loading may take several minutes (cheap deployment as it is just a demonstration model).'''
    )
    # st.session_state['vectors_reloaded'] = KeyedVectors.load_word2vec_format(
    #     'Demo_Project_NLP_sentiment_analysis_benbhk/models/glove-wiki-gigaword-200.txt',
    #     binary=False)
    st.session_state['vectors_reloaded'] = api.load('glove-wiki-gigaword-200')
    # vectors_reloaded = st.session_state['vectors_reloaded']
    st.markdown('''Word2Vec model loaded''')

sentence = st.text_area('Text to analyze (Press ctrl + enter to run)', value="",placeholder="Your text here")

if sentence != '':

    embed_dim = 200  # Embedding size for each token
    num_heads = 1  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    maxlen = 100

    inputs = Input(shape=(maxlen,embed_dim))
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(inputs)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(20, activation="relu")(x)
    x = Dropout(0.1)(x)
    outputs = Dense(5, activation="softmax")(x)

    model_TR = Model(inputs=inputs, outputs=outputs)

    model_TR.load_weights('Demo_Project_NLP_sentiment_analysis_benbhk/models/transformer_weights_W2V_200_1attention_head_accuracy_0667.h5')

    # -------------------------------------------------------------------------------------------------#
    # Text imput and processing

    st.markdown(
        '''If you are looking for some inspiration, here are some links to twitter accounts..
                - [Barack Obama](https://twitter.com/BarackObama?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor)
                - [Elon Musk](https://twitter.com/elonmusk?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor)
                - [CNN](https://twitter.com/CNN?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor)
                - [Cristiano Ronaldo](https://twitter.com/Cristiano?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor)
                - [The New York Times](https://twitter.com/nytimes?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor)
                ''')

    sentence = str.encode(sentence)
    sentence = text_to_word_sequence(sentence.decode("utf-8"))
    sentence= [sentence]
    sentence = embedding(st.session_state['vectors_reloaded'], sentence)
    sentence = pad_sequences(sentence, dtype='float32', padding='post', maxlen=100)
    prediction = model_TR.predict(sentence).tolist()[0]


    # st.write(f'prediction : {prediction}')

    fig, ax = plt.subplots(figsize=(15, 4))
    plt.bar(
        ['Very negative', 'Negative', 'Neutral', 'Positive', 'Very positive'],
        prediction,
        color=['darkred', 'indianred', 'lightskyblue', 'mediumseagreen', 'darkgreen'])
    plt.xlabel('Sentiment Analysis')
    plt.xticks(rotation=45, ha="center")
    plt.ylabel('Prediction in %')


    font = {'family' : 'normal',
            'size'   : 15}
    plt.rc('font', **font)

    st.pyplot(fig)
