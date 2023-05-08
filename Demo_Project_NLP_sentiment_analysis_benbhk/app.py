import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from lib_func import embedding, TransformerBlock
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import gensim.downloader as api
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

from tensorflow.keras.layers import Dropout, Input, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model

# Load image and display it
image = Image.open('Demo_Project_NLP_sentiment_analysis_benbhk/top.png')
st.image(image)

# Define text normalization function
def img_normalizer(X):
    return X/255-0.5

# Display app title and description
st.title("Text sentiment analysis")
st.subheader("Benjamin Barre - Demo Project")
st.write('In this project, I use the Tensorflow.Keras library with an attention-based Transformer to analyze the sentiment of a text. This model uses transfer learning from the gensim-Word2Vec library (glove-wiki-gigaword).')
st.write('It may take several minutes to start (cheap deployment as it is just a demonstration model).')
st.write('You can check the code on GitHub.')
st.write("Link to the [GitHub repo](https://github.com/Benjaminbhk/Demo_Project_NLP_sentiment_analysis_benbhk.git)")

st.markdown('''---''')

# Load Word2Vec model if not already loaded
if 'vectors_reloaded' not in st.session_state:
    st.markdown('''Loading of the Word2Vec model (glove-wiki-gigaword-200). Loading may take several minutes (cheap deployment as it is just a demonstration model).''')
    st.write("You can visit my [page/portfolio](https://inky-distance-393.notion.site/Hi-I-am-Benjamin-Barre-Data-Scientist-f8a4416f9dc64d9ab8c677a5a32ab03d) while it loads.")
    st.session_state['vectors_reloaded'] = api.load('glove-wiki-gigaword-200')
    st.markdown('''Word2Vec model loaded''')

# Display some links to Twitter accounts for inspiration
st.markdown('''
            If you are looking for some inspiration, here are some links to Twitter accounts:
            - [Barack Obama](https://twitter.com/BarackObama?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor)
            - [Elon Musk](https://twitter.com/elonmusk?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor)
            - [CNN](https://twitter.com/CNN?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor)
            - [Cristiano Ronaldo](https://twitter.com/Cristiano?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor)
            - [The New York Times](https://twitter.com/nytimes?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor)
            ''')

# Get user input for text to analyze
sentence = st.text_area('Text to analyze (Press ctrl + enter to run)', value="", placeholder="Your text here")

if sentence != '':
    embed_dim = 200  # Embedding size for each token
    num_heads = 1  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    maxlen = 100

    # Define Transformer model architecture
    inputs = Input(shape=(maxlen, embed_dim))
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(inputs)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(20, activation="relu")(x)
    x = Dropout(0.1)(x)
    outputs = Dense(5, activation="softmax")(x)

    model_TR = Model(inputs=inputs, outputs=outputs)

    # Load pre-trained model weights
    model_TR.load_weights('Demo_Project_NLP_sentiment_analysis_benbhk/models/transformer_weights_W2V_200_1attention_head_accuracy_0667.h5')

    # Process input text
    sentence = str.encode(sentence)
    sentence = text_to_word_sequence(sentence.decode("utf-8"))
    sentence = [sentence]
    sentence = embedding(st.session_state['vectors_reloaded'], sentence)
    sentence = pad_sequences(sentence, dtype='float32', padding='post', maxlen=100)

    # Make a prediction with the model
    prediction = model_TR.predict(sentence).tolist()[0]

    # Visualize the prediction results
    fig, ax = plt.subplots(figsize=(15, 4))
    bar_labels = ['Very negative', 'Negative', 'Neutral', 'Positive', 'Very positive']
    bar_colors = ['darkred', 'indianred', 'lightskyblue', 'mediumseagreen', 'darkgreen']
    bars = plt.bar(bar_labels, prediction, color=bar_colors)
    plt.xlabel('Sentiment Analysis')
    plt.xticks(rotation=45, ha="center")
    plt.ylabel('Prediction in %')

    font = {'family': 'normal', 'size': 15}
    plt.rc('font', **font)

    st.pyplot(fig)

    # Calculate the score between 0 and 10
    weights = [0, 0.25, 0.5, 0.75, 1]
    score = sum([prediction[i] * weights[i] for i in range(len(prediction))])
    normalized_score = score * 10

    # Display the score below the bar chart
    st.markdown(f'<p style="font-size: 28px; text-align: center;font-weight: bold;">Score: {normalized_score:.2f}/10"</p>', unsafe_allow_html=True)

    # Add an explanation next to the score
    st.markdown(f'<p style="font-size: 14px; text-align: center;">(0 is for a very negative tweet and 10 for a very positive tweet</p>', unsafe_allow_html=True)

    # Display an explanation of the message sentiment based on the score range
    st.markdown(f'<p style="font-size: 28px; text-align: center;font-weight: bold;">The message is {"very negative" if normalized_score <= 2 else "negative" if normalized_score <= 4 else "neutral" if normalized_score <= 6 else "positive" if normalized_score <= 8 else "very positive"}.</p>', unsafe_allow_html=True)

    st.markdown('''---''')
    st.write("My [page/portfolio](https://inky-distance-393.notion.site/Hi-I-am-Benjamin-Barre-Data-Scientist-f8a4416f9dc64d9ab8c677a5a32ab03d).")


# import streamlit as st
# import matplotlib.pyplot as plt
# # from tensorflow.keras import layers, Sequential, models
# # import pickle as pkl
# from PIL import Image
# from lib_func import embedding,TransformerBlock # embed_sentence_with_TF,
# # from tensorflow.keras.backend import expand_dims
# from tensorflow.keras.preprocessing.text import text_to_word_sequence
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import numpy as np
# import gensim.downloader as api
# # from gensim.models import KeyedVectors

# from tensorflow.keras.layers import Dropout  #MultiHeadAttention,LayerNormalization,Layer
# from tensorflow.keras.layers import Input, GlobalAveragePooling1D, Dense #Embedding
# from tensorflow.keras.models import Model # Sequential,
# import numpy as np
# import warnings
# warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


# image = Image.open('Demo_Project_NLP_sentiment_analysis_benbhk/top.png')

# # st.set_page_config(layout="wide")

# st.image(image)


# def img_normalizer(X):
#     return X/255-0.5

# st.title(" Text sentiment analysis ")
# st.subheader(" Benjamin Barre - Demo Project ")
# st.write('In this project, I uses the Tesorflow.Keras library with an attention based Transformer to analyze the sentiment of a text. This model uses a transfer learning from the gensim-Word2Vec library (glove-wiki-gigaword).')
# st.write('It may takes severals minutes to starts (cheap deployment as it is just a demonstration model)')
# st.write('You can check the code on github.')
# # st.write('In this project, I confront three models to analyze the sentiment of a text. Each model uses a transfer learning from the gensim-Word2Vec library (glove-wiki-gigaword).')
# # st.write('- The first model uses the XGBoost library')
# # st.write('- The second model uses the classical Tesorflow.Keras library (Adam optimization)')
# # st.write('- The third model also uses the Tesorflow.Keras library with an attention based Transformer (Adam optimization as well)')
# st.write("Link to the [github repo](https://github.com/Benjaminbhk/Demo_Project_NLP_sentiment_analysis_benbhk.git)")

# st.markdown('''---''')

# if 'vectors_reloaded' not in st.session_state:
#     st.markdown(
#         '''Loading of the Word2Vec model (glove-wiki-gigaword-200). Loading may take several minutes (cheap deployment as it is just a demonstration model).'''
#     )
#     st.write(
#         "You can visit my [page/portfolio](https://inky-distance-393.notion.site/Hi-I-am-Benjamin-Barre-Data-Scientist-f8a4416f9dc64d9ab8c677a5a32ab03d) while it loads."
#     )
#     # st.session_state['vectors_reloaded'] = KeyedVectors.load_word2vec_format(
#     #     'Demo_Project_NLP_sentiment_analysis_benbhk/models/glove-wiki-gigaword-200.txt',
#     #     binary=False)
#     st.session_state['vectors_reloaded'] = api.load('glove-wiki-gigaword-200')
#     # vectors_reloaded = st.session_state['vectors_reloaded']
#     st.markdown('''Word2Vec model loaded''')

# st.markdown('''
#             If you are looking for some inspiration, here are some links to twitter accounts :
#             - [Barack Obama](https://twitter.com/BarackObama?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor)
#             - [Elon Musk](https://twitter.com/elonmusk?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor)
#             - [CNN](https://twitter.com/CNN?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor)
#             - [Cristiano Ronaldo](https://twitter.com/Cristiano?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor)
#             - [The New York Times](https://twitter.com/nytimes?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor)
#             ''')

# sentence = st.text_area('Text to analyze (Press ctrl + enter to run)', value="",placeholder="Your text here")

# if sentence != '':

#     embed_dim = 200  # Embedding size for each token
#     num_heads = 1  # Number of attention heads
#     ff_dim = 32  # Hidden layer size in feed forward network inside transformer

#     maxlen = 100

#     inputs = Input(shape=(maxlen,embed_dim))
#     transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
#     x = transformer_block(inputs)
#     x = GlobalAveragePooling1D()(x)
#     x = Dropout(0.1)(x)
#     x = Dense(20, activation="relu")(x)
#     x = Dropout(0.1)(x)
#     outputs = Dense(5, activation="softmax")(x)

#     model_TR = Model(inputs=inputs, outputs=outputs)

#     model_TR.load_weights('Demo_Project_NLP_sentiment_analysis_benbhk/models/transformer_weights_W2V_200_1attention_head_accuracy_0667.h5')

#     # -------------------------------------------------------------------------------------------------#
#     # Text imput and processing



#     sentence = str.encode(sentence)
#     sentence = text_to_word_sequence(sentence.decode("utf-8"))
#     sentence= [sentence]
#     sentence = embedding(st.session_state['vectors_reloaded'], sentence)
#     sentence = pad_sequences(sentence, dtype='float32', padding='post', maxlen=100)
#     prediction = model_TR.predict(sentence).tolist()[0]


#     # st.write(f'prediction : {prediction}')

#     fig, ax = plt.subplots(figsize=(15, 4))
#     plt.bar(
#         ['Very negative', 'Negative', 'Neutral', 'Positive', 'Very positive'],
#         prediction,
#         color=['darkred', 'indianred', 'lightskyblue', 'mediumseagreen', 'darkgreen'])
#     plt.xlabel('Sentiment Analysis')
#     plt.xticks(rotation=45, ha="center")
#     plt.ylabel('Prediction in %')


#     font = {'family' : 'normal',
#             'size'   : 15}
#     plt.rc('font', **font)

#     st.pyplot(fig)

# st.markdown('''---''')
# st.write(
#         "My [page/portfolio](https://inky-distance-393.notion.site/Hi-I-am-Benjamin-Barre-Data-Scientist-f8a4416f9dc64d9ab8c677a5a32ab03d)."
#     )
