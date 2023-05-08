# Sentiment Analysis with Streamlit
This project is a web application that performs sentiment analysis on a text input using an attention-based transformer model with transfer learning from the gensim Word2Vec library (glove-wiki-gigaword). The user interface is built using the Streamlit framework, which provides an interactive and user-friendly experience.

## Features
Interactive user interface for inputting text and analyzing sentiment
Real-time prediction using a trained transformer model
Displays the sentiment analysis results in a bar chart
Displays a score between 0 and 10 based on the sentiment analysis
Displays a text interpretation of the score range

## Technologies
Python
TensorFlow for training and using the transformer model
Streamlit for creating the user interface

## Getting Started
To run the project locally, follow these steps:

1. Clone the repository:

git clone https://github.com/your-username/sentiment-analysis-app.git

2. Create a virtual environment and install the required dependencies:

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

3. Run the Streamlit app:

streamlit run app.py

4. Open a web browser and navigate to the local URL displayed in the terminal.

## Demo
The app is deployed on Streamlit sharing platform, you can access it through this link :

## Contributing
Feel free to contribute to this project by submitting issues, pull requests, or providing feedback on the application. Your contributions are welcome and appreciated!

## License
This project is licensed under the MIT License. See the LICENSE file for details.
