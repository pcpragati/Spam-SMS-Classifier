import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
import altair as alt
import pandas as pd
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

stop_words = set(stopwords.words('english'))


def transform_text(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Tokenize using regex (safer for deployment)
    tokens = re.findall(r'\b\w+\b', text)

    # 3. Remove stopwords and punctuation, then stem
    cleaned_tokens = []
    for word in tokens:
        if word not in stop_words and word not in string.punctuation:
            cleaned_tokens.append(ps.stem(word))

    # 4. Return as string
    return " ".join(cleaned_tokens)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.set_page_config(page_title="Spam SMS Detector", page_icon="📩", layout="wide")

st.title("📩 Spam SMS Classifier")
st.markdown("This app uses a Machine Learning model to classify SMS messages as **Spam** or **Not Spam**.")

# Tabs for better structure
tab1, tab2 = st.tabs(["🔮 Predict", "ℹ️ About"])

with tab1:
    st.subheader("Enter your SMS below:")
    input_sms = st.text_area("✍️ Type your message here...")
    if st.button("🚀 Predict"):
        if input_sms.strip() == "":
            st.warning("⚠️ Please enter a message to classify.")
        else:
            # 1. Preprocess
            transformed_sms = transform_text(input_sms)

            # 2. Vectorize
            vector_input = tfidf.transform([transformed_sms])

            # 3. Predict
            result = model.predict(vector_input)[0]
            proba = model.predict_proba(vector_input)[0]  # [Not Spam, Spam]

            # 4. Display Result
            if result == 1:
                st.error(f"🚨 This message is classified as **Spam**")
            else:
                st.success(f"✅ This message is classified as **Not Spam**")
            # Confidence chart
            st.markdown("### 🔎 Model Confidence")
            data = pd.DataFrame({
                "Label": ["Not Spam", "Spam"],
                "Probability": [proba[0], proba[1]]
            })
            chart = alt.Chart(data).mark_bar().encode(
                x="Label",
                y="Probability",
                color="Label"
            )
            st.altair_chart(chart, use_container_width=True)
    with tab2:
        st.subheader("📖 About this App")
        st.write("""
        This **Spam SMS Detector** is built using **classical Machine Learning techniques**.

        ### ⚙️ How it works:
        1. **Text Preprocessing**  
           - Lowercasing  
           - Tokenization  
           - Removing stopwords & punctuation  
           - Stemming (Porter Stemmer)  

        2. **Feature Extraction**  
           - Using **TF-IDF (Term Frequency - Inverse Document Frequency)** vectorization to convert text into numerical features.  

        3. **Model Used**  
           - After experimenting with multiple models, **Multinomial Naive Bayes** gave the best accuracy for this dataset.  

        ### ✅ Labels:
        - **Spam (1):** Promotional or unwanted messages.  
        - **Not Spam (0):** Safe / genuine messages.  

        ---
        ⚡ This is a demo app made to showcase how Machine Learning can be applied for text classification tasks like Spam Detection.
        """)

    # st.title("Spam SMS Classifier")
    # input_sms = st.text_area("Enter your message")
    #
    # if st.button('Predict'):
    #
    #     # 1. preprocess
    #     transformed_sms = transform_text(input_sms)
    #
    #     # 2. vectorize
    #     vector_input = tfidf.transform([transformed_sms])
    #
    #     # 3. predict
    #     result = model.predict(vector_input)[0]
    #
    #     # 4. display
    #     if result == 1:
    #         st.header("Spam")
    #     else:
    #         st.header("Not Spam")