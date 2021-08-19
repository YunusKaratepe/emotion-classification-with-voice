import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

import pickle

FRAME_SIZE = 2048
HOP_LENGTH = 512

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

def gpu_config(memory_limit):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) >= 1:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
        return gpus    
    
    return None

def create_spectrogram(spectrogram_duration=150, order=0, spec_type='log'):
    audio_file='./voice.wav'
    voice, sr = librosa.load(audio_file)
    after_stft = librosa.stft(voice, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)
    
    i = 0
    while True:
        after_stft_i = after_stft[:,i*spectrogram_duration:(i+1)*spectrogram_duration]
        if len(after_stft_i[1]) <= 50:
            break
        abs_stft_result = np.abs(after_stft_i) ** 2
        log_stft_result = librosa.power_to_db(abs_stft_result)
        if order != 0:
            log_stft_delta = librosa.feature.delta(log_stft_result, order)
            plot_spectrogram(i, log_stft_delta, sr, HOP_LENGTH, y_axis=spec_type)
        plot_spectrogram(i, log_stft_result, sr, HOP_LENGTH, y_axis=spec_type)
        
        i += 1
    return i

def plot_spectrogram(index, Y, sr, hop_length, y_axis="log"):
    plt.figure(figsize=(25, 10))
    plt.axis('off')
    librosa.display.specshow(Y,
                             sr=sr,
                             hop_length=hop_length,
                             y_axis=y_axis)

    plt.savefig(f"./spectrogram{index}.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    
def predict_emotion(index, cnn_model_path, spec_type='log', test_path="none"):

    if not os.path.exists(f"./spectrogram{index}.png"):
        print("Please first run Create Spectrogram section.")
        return
    if not os.path.exists(cnn_model_path):
        print("Model path does not exists.")
        return


    cnn = load_model(cnn_model_path)

    #spec = cv2.imread("./spectrogram.png")
    # spec = cv2.imread(test_path)

    # spec = cv2.resize(spec, (128, 128))

    if test_path != "none":
        spec = tf.keras.preprocessing.image.load_img(test_path, target_size=(128, 128))

    else:
        spec = tf.keras.preprocessing.image.load_img(f"./spectrogram{index}.png", target_size=(128, 128))




    spec = np.reshape(spec, [1, 128, 128, 3])


    classes = cnn.predict(spec)
    
    return emotions_dict[list(classes[0]).index(max(list(classes[0])))]


def predict_emotion_ml(index, cnn_model_path, spec_type='log', seed=10, ml_algorithm='SVC_Polynomial_Kernel', test_path="none", use_less_trained_model=False):
    if use_less_trained_model:
        ml_model_path = "./models/ml_models_epoch50/seed" + str(seed) + "/"  + ml_algorithm + ".sav"
    else:
        ml_model_path = "./models/ml_models/seed" + str(seed) + "/"  + ml_algorithm + ".sav"
    if not os.path.exists(ml_model_path):
        print("ML-Model type is not valid.")
        return
    if not os.path.exists(f"./spectrogram{index}.png"):
        print("Please first run Create Spectrogram section.")
        return
    if not os.path.exists(cnn_model_path):
        print("Model path does not exists.")
        return

    cnn = load_model(cnn_model_path)
    
    
    if test_path != "none":
        spec = tf.keras.preprocessing.image.load_img(test_path, target_size=(128, 128))

    else:
        spec = tf.keras.preprocessing.image.load_img(f"./spectrogram{index}.png", target_size=(128, 128))


    spec = np.reshape(spec, [1, 128, 128, 3])
    
    outputLayer = cnn.layers[-5]
    intermediate_layer_model = Model(inputs=cnn.input,
                                    outputs=outputLayer.output)

    intermediate_output = intermediate_layer_model.predict(spec)


    ml_model = pickle.load(open(ml_model_path, 'rb'))

    # ---------------------------------------------- problematik

    
    # ml_output = ml_model.predict_proba(lst)
    # ml_list = list(ml_output)
    
    
    for i in range(9):
        result = ml_model.score(intermediate_output, [i])
        # print(str(int(result)), end="\t")
        if result == 1:
            return emotions_dict[int(i)]
    
    # emotions_dict[ml_list.index(max(ml_list))] 
    
def clear_spectrograms(path_to_spectrograms):
    files = os.listdir(path_to_spectrograms)
    for f in files:
        if f.startswith('spectrogram') and f.endswith('.png'):
            os.remove(f)
            
def predict_with_ml(ml_algorithms=None, spec_type='log', order=0, spectrogram_duration=150, use_less_trained_model=False):
    import warnings
    warnings.filterwarnings('ignore')

    max_i = create_spectrogram(spectrogram_duration=spectrogram_duration, order=order, spec_type=spec_type) # ignore warning
    all_predicts = []
    
    for i in range(max_i):
        for s in [10, 50, 100]:
            if use_less_trained_model:
                cnn_model_path = "./models/cnn_models_epoch50/seed" + str(s) + "/logSpecAugment.h5"
            else:
                cnn_model_path = "./models/cnn_models/seed" + str(s) + "/logSpecAugment.h5"
            
            all_predicts.append(predict_emotion(i, cnn_model_path=cnn_model_path, spec_type=spec_type))
            for ml_algo in ml_algorithms:
                all_predicts.append(predict_emotion_ml(i, cnn_model_path=cnn_model_path, spec_type=spec_type, seed=s, ml_algorithm=ml_algo, use_less_trained_model=use_less_trained_model))

                
    from collections import Counter
    most_commons = Counter(all_predicts).most_common(3)
    clear_spectrograms(path_to_spectrograms='./')
    return most_commons, all_predicts
    

# print(gpu_config(8000))
# print(predict_with_ml(audio_file='./audios/happy.mp3'))