import keras
import streamlit as st
from utilitiesFunctions import *
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import numpy as np


def clear_text():
    """Reset the session state for the CLear Text button"""
    st.session_state['city'] = ''
    st.session_state['heading'] = ''


def preprocessInput(text):
    """Preprocess the input for LSTM"""
    # Clean the text data
    textInput = basic_cleaning(text)
    tokens = tokenize_text(textInput)
    lemmaInput = lemmatize_token(tokens)
    stopwordFreeInput = remove_stopwords(lemmaInput)

    # Join the list of tokens back into a single string
    rejoined_text = ' '.join(stopwordFreeInput)

    # Preprocessed the text for LSTM
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([rejoined_text])
    X_seq = tokenizer.texts_to_sequences([rejoined_text])
    X_pad = pad_sequences(X_seq, maxlen=17)
    return X_pad


# Load the saved model from the HDF5 file
interfaceModel = keras.models.load_model('Model/LSTM_tt.h5')

label_mapping = {
    0: 'activities', 1: 'appliances', 2: 'artists', 3: 'automotive',
    4: 'cell-phones', 5: 'childcare', 6: 'general', 7: 'household-services',
    8: 'housing', 9: 'photography', 10: 'real-estate', 11: 'shared',
    12: 'temporary', 13: 'therapeutic', 14: 'video-games', 15: 'wanted-housing'
}

wordcloud_dict ={
        'activities': 'Images/activities wordcloud.png',
        'appliances': 'Images/appliances wordcloud.png',
        'artists': 'Images/artists wordcloud.png',
        'automotive': 'Images/automotive wordcloud.png',
        'cell-phones': 'Images/cell-phones wordcloud.png',
        'childcare': 'Images/childcare wordcloud.png',
        'general': 'Images/general wordcloud.png',
        'household-services': 'Images/household-services wordcloud.png',
        'housing': 'Images/housing wordcloud.png',
        'photography': 'Images/photography wordcloud.png',
        'real-estate': 'Images/real-state wordcloud.png',
        'shared': 'Images/shared wordcloud.png',
        'temporary': 'Images/temporary wordcloud.png',
        'therapeutic': 'Images/therapeutic wordcloud.png',
        'video-games': 'Images/video-games wordcloud.png',
        'wanted-housing': 'Images/wanted-housing wordcloud.png',
    }


def display_image(selected_option):
    caption = f"{selected_option} wordcloud"
    st.image(wordcloud_dict[selected_option], caption=caption, use_column_width=True)


def form_page():
    """Set up the form page"""
    section = ['', 'for-sale', 'housing', 'community', 'services']
    st.title('Form Prediction')
    with st.form('postDetailForm'):
        city = st.text_input('Enter city:', key='city')
        section = st.selectbox('Select post section:', section, index=0)
        heading = st.text_input('Enter post heading:', key='heading')

        col1, col2 = st.columns([5, 1])
        with col1:
            predictForm = st.form_submit_button(label='Predict Form')
        with col2:
            clearForm = st.form_submit_button(label='Clear Form', on_click=clear_text)

        if predictForm:
            # Validate required fields
            if not city.strip() or not heading.strip():
                st.error('Please enter text for city and heading!')
            else:
                form_data = {
                    'city': city,
                    'section': section,
                    'heading': heading
                }
                form_data_str = ' '.join(str(value) for value in form_data.values())
                text = preprocessInput(form_data_str)

                # Make a prediction on the preprocessed text
                post_pred = interfaceModel.predict(text)
                predicted_labels = np.argmax(post_pred, axis=1)
                predicted_categories = [label_mapping[label] for label in predicted_labels]

                # Output the result
                st.write(f'Predicted label: {predicted_labels}')
                st.write(f'Predicted categories: {predicted_categories}')


def visuals():
    """Visualizations of the data and model performance"""
    st.title('Data Visualization')
    st.markdown('---')

    with st.container():
        img_col, txt_col = st.columns([2,1])
        with img_col:
            st.image('Images/Distribution of categories of the training data.png')
        with txt_col:
            st.subheader('Distribution of categories in the training data')
            st.write('Shared is the most represented categories in the training dataset with 11.2%, whereas housing '
                     'is the least represented category with just 2.0%')

    with st.container():
        img_col, txt_col = st.columns([2,1])
        with img_col:
            st.image('Images/Confusion Matrix for LSTM.png')
        with txt_col:
            st.subheader('Confusion Matrix for LSTM')
            st.write('The chosen model is most adept at classifying therapeutic followed by cell-phones. However, '
                     'the model is also very weak at classifying housing, artists and activities')

    with st.container():
        img_col, txt_col = st.columns([2, 1])
        with img_col:
            st.image('Images/Model Performance Comparison.png')
        with txt_col:
            st.subheader('Model Performance Comparison')
            st.write('The best model is the baseline LSTM with 78% accuracy.')

    # Select category to choose the wordcloud visualization
    selected_category = st.selectbox('Select a category to visualize the wordcloud:', list(wordcloud_dict.keys()))

    # Display image based on selected option
    display_image(selected_category)



def app():
    """Setting up the Streamlit app with all the pages"""
    st.set_page_config(page_title='Post Category Prediction')
    pages = {
        'Form Prediction': form_page,
        'Data Visualization': visuals
    }
    page = st.sidebar.radio('Go to', tuple(pages.keys()))
    pages[page]()


app()

