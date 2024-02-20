import streamlit as st
import pandas as pd
from nltk.tokenize import word_tokenize

data = pd.read_csv("HindiSentiWordnet.txt", delimiter=' ')

fields = ['POS_TAG', 'ID', 'POS', 'NEG', 'LIST_OF_WORDS']

# Creating a dictionary which contain a tuple for every word. Tuple contains a list of synonyms,
# positive score and negative score for that word.
words_dict = {}
for i in data.index:
    words = data[fields[4]][i].split(',')
    for word in words:
        words_dict[word] = (data[fields[0]][i],
                            data[fields[2]][i], data[fields[3]][i])

def sentiment(text):
    words = word_tokenize(text)
    votes = []
    pos_polarity = 0
    neg_polarity = 0
    # adverbs, nouns, adjective, verb are only used
    allowed_words = ['a', 'v', 'r', 'n']
    for word in words:
        if word in words_dict:
            # if word in dictionary, it picks up the positive and negative score of the word
            pos_tag, pos, neg = words_dict[word]
            # print(word, pos_tag, pos, neg)
            if pos_tag in allowed_words:
                if pos > neg:
                    pos_polarity += pos
                    votes.append(1)
                elif neg > pos:
                    neg_polarity += neg
                    votes.append(0)
    # calculating the no. of positive and negative words in total in a review to give class labels
    pos_votes = votes.count(1)
    neg_votes = votes.count(0)
    if pos_votes > neg_votes:
        return 1
    elif neg_votes > pos_votes:
        return 0
    else:
        if pos_polarity < neg_polarity:
            return 0
        else:
            return 1


st.title('Hindi Sentence Sentiment Analysis')

with st.form(key="form1"):
    user_input = st.text_input('Enter The Sentence')
    submit = st.form_submit_button(label="Get Sentiment")

st.subheader("Predicted Sentiment: ")
if submit == True:
    x = sentiment(user_input)
    if x == 1:
        st.subheader("Positive Sentence")
    else:
        st.subheader("Negative Sentence")
