import helper
import streamlit as st
from pydub import AudioSegment
from PIL import Image

project_slide = "https://docs.google.com/presentation/d/1UGMgex6G5fAtTqPs33SiFulAznYufKwQ/edit?usp=sharing&ouid=118405775020092724633&rtpof=true&sd=true"

st.write(f""" # Sentiment Classifier Through Voice!
    * This app classifies emotions from given voice data.
    * Author: ***Yunus KARATEPE***.
    * Thanks to **Ilker TINKIR** who is worked with me at training models phase of this project. 
    * You can check project slide from [here]({project_slide}).
""")

gpus = helper.gpu_config(8000)
st.subheader('Available Gpus')
gpus

emotions_dict = {
    0: 'Neutral',
    1: 'Calm',
    2: 'Happy',
    3: 'Sad',
    4: 'Angry',
    5: 'Fearful',
    6: 'Disgusted',
    7: 'Surprised'
}

st.subheader('Emotions')
emotions_dict

acc_values = Image.open('accs.png')
st.write('* Here are some accuracy values for different machine learning \
    algorithms to let you know which algorithm performed better in production phase.')
st.image(acc_values, use_column_width=True)



ml_algorithms = [
    "MLPClassifier",
    "LogisticRegression",
    "RandomForestClassifier",
    "LinearDiscriminantAnalysis",
    "KNeighborsClassifier",
    "SVC_Polynomial_Kernel",
    "GaussianNB",
    "GradientBoostingClassifier",
    "AdaBoostClassifier",
    "LinearSVC",
    "SVC_RBF_Kernel"
]

suggested_ml_algorithms = [
    "LogisticRegression",
    "LinearDiscriminantAnalysis",
    "SVC_Polynomial_Kernel",
    "LinearSVC",
]


st.sidebar.header('User Inputs')

   
selected_ml_algorithms = st.sidebar.multiselect('Select ML-Algorithms that you want to use to predict', ml_algorithms, suggested_ml_algorithms)

epoch_selection = st.sidebar.selectbox('Use 50 or 100 epoch trained models?', ['50 epochs', '100 epochs'], 1)

spectrogram_duration = st.sidebar.slider('Time-Axis Duration of each spectrogram', 100, 500, 140)

uploaded = st.sidebar.file_uploader('Upload a mp3 or wav file to predict it\'s emotion', ['wav', 'mp3'])

def convert_mp3_to_wav(voice_file):
    sound = AudioSegment.from_mp3(voice_file)
    sound.export('./voice.wav', format='wav')

if uploaded:
    st.subheader('Uploaded Voice')
    st.audio(uploaded)
    convert_mp3_to_wav(uploaded)
    st.subheader('Click the button to predict emotion of selected file!')
    if st.button('Predict'):
        
        results, _ = helper.predict_with_ml(ml_algorithms=selected_ml_algorithms, \
            use_less_trained_model=(epoch_selection=='50 epochs'), spectrogram_duration=spectrogram_duration)
        
        st.subheader('Predicted Emotions by Voting (top 3)')
        st.dataframe(results)
        


