# hindi-sentence-sentiment-analysis

### The Hindi Sentiment Analysis System is a computerized technique used to determine the sentiment or emotional meaning of words or sentences in the Hindi language. The goal of this system is to identify the emotional state or thoughts of individuals by analyzing various textual expressions.

This is a NLP project which takes any Hindi sentence as input and predicts whether it is a Positive or a Negative Sentence.

There are total 7 files
1. hindi_stopwords.txt-> this file contains all the stopwords of hindi language
2. hindisentiwordnet.txt -> this file contains the main dataset which is used to predict the output. It has POS_Tags (parts of speech tag), ID, POS (positive polarity), NEG(negative polarity), List of Word
3. NEG_hindi.txt -> this file contains many Negative Hindi Sentences
4. POS_hindi.txt -> this file contains many Positive Hindi Sentences
5. main.py -> this is the main python file which shows the accuracy of the dataset.
6. nlp.py -> this is the frontend of the project made using streamlit

How to Run?
option1: python main.py
option2: streamlit run nlp.py
