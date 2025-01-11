# import pickle
# import streamlit as st
# import pandas as pd
# import re
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
#
#
# @st.cache_resource
# def load_model_and_vectorizer():
#     with open('fake_news_model.pkl', 'rb') as model_file:
#         model = pickle.load(model_file)
#     with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
#         vectorizer = pickle.load(vec_file)
#     return model, vectorizer
#
# model, vectorizer = load_model_and_vectorizer()
#
#
# # news_df=pd.read_csv('train.csv')
# # news_df=news_df.fillna(' ')
# # news_df['content']=news_df['author']+" "+news_df['title']
# # X=news_df.drop('label',axis=1)
# # Y=news_df['label']
#
# stop_words=set(stopwords.words('english'))
#
#
# def stemming(content):
#     ps = PorterStemmer()
#     stemmed_content=re.sub('[^a-zA-Z]'," ",content)
#     stemmed_content=stemmed_content.lower()
#     stemmed_content=stemmed_content.split()
#     stemmed_content=[ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
#     stemmed_content=" ".join(stemmed_content)
#     return stemmed_content
#
# # news_df['content'] = news_df['content'].apply(stemming)
#
#
# X = news_df['content'].values
# Y = news_df['label'].values
#
#
# vector=TfidfVectorizer()
# vector.fit(X)
# X=vector.transform(X)
#
# X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=1)
#
#
# model=LogisticRegression()
# model.fit(X_train,Y_train)
#
#
#
# #Website
#
# st.title('Fake News Detector')
# input_text=st.text_input("Enter News Article:","")
#
# if st.button("Check for fake News"):
#     if input_text:
#         input_vector=vectorizer.transform([input_text])
#
#         prediction = model.predict(input_vector)
#
#     if prediction[0]==1:
#          st.success('News is fake')
#     else:
#          st.error('News is real')
#
# else:
#     st.warning('No fake news detected')

import pickle
import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from bs4 import BeautifulSoup

nltk.download('stopwords')

@st.cache_resource
def load_model_and_vectorizer():
    with open('C:/Users/Naveen Sen/PycharmProjects/FakeNewsDetector/fake_news_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('C:/Users/Naveen Sen/PycharmProjects/FakeNewsDetector/tfidf_vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

def stemming(content):
    ps = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]', " ", content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = " ".join(stemmed_content).strip()
    return stemmed_content

def fetch_news_from_website(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            headlines = soup.find_all('h3')  
            return [headline.text.strip() for headline in headlines if headline.text.strip()]
        else:
            st.error(f"Error fetching news: HTTP {response.status_code}")
            return []
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return []

st.title('Fake News Detector')


option = st.radio("Choose Input Source:", ("Enter Manually", "Fetch Online News"))

if option == "Enter Manually":
    input_text = st.text_input("Enter News Article:", "")

    if st.button("Check for Fake News"):
        if input_text:
            processed_text = stemming(input_text)
            input_vector = vectorizer.transform([processed_text])

        
            prediction = model.predict(input_vector)

            if prediction[0] == 1:
                st.error('News is Fake')
            else:
                st.success('News is Real')

elif option == "Fetch Online News":
    news_url=st.text_input("Enter News URL:", "")

    if st.button("Fetch and Check News"):
        if news_url:
            news_headlines = fetch_news_from_website(news_url)

            if news_headlines:
                st.subheader("Fetched News Headlines")
                for i, news in enumerate(news_headlines, 1):
                    processed_text = stemming(news)
                    input_vector = vectorizer.transform([processed_text])
                    prediction = model.predict(input_vector)

                    st.write(f"**{i}. {news}**")
                    if prediction[0] == 1:
                        st.success("Prediction: Real")
                    else:
                        st.error("Prediction: Fake")

