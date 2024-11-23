import librosa
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import sounddevice as sd
import soundfile as sf
import os
import tensorflow.keras as keras


def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_best')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=80)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    print(f"Extracted MFCC features: {mfccs_scaled_features.shape}")
    return mfccs_scaled_features


with open(r'C:\\Users\\surya\\OneDrive\\Desktop\\Extracted_Features306.pkl', 'rb') as f:
    Data = pickle.load(f)


df = pd.DataFrame(Data, columns=['feature', 'class'])
Y = np.array(df['class'].tolist())


labelencoder = LabelEncoder()
y = to_categorical(labelencoder.fit_transform(Y))


loaded_model = keras.models.load_model("C:\\Users\\surya\\Downloads\\ML\\saved_model301.h5")


def record_audio(duration, sample_rate=22050):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  
    audio = audio.flatten()
    print("Recording complete.")
    return audio, sample_rate


def save_audio_to_wav(audio, sample_rate, file_path):
    sf.write(file_path, audio, sample_rate)
    print(f"Audio saved to {file_path}")


def amplify_and_denoise(audio, sample_rate):
    
    audio = librosa.effects.preemphasis(audio)
    print("Audio amplified using pre-emphasis filter.")
    
    audio, _ = librosa.effects.trim(audio, top_db=20)
    print("Silent parts trimmed from audio.")
    return audio


duration = 10  
audio_data, sr = record_audio(duration)


wav_file_path = 'recorded_audio.wav'
save_audio_to_wav(audio_data, sr, wav_file_path)


audio_data = amplify_and_denoise(audio_data, sr)


save_audio_to_wav(audio_data, sr, wav_file_path)


input_features = features_extractor(wav_file_path)
print(f"Input features shape: {input_features.shape}")
input_features = input_features.reshape(1, -1)  


predictions = loaded_model.predict(input_features)
predicted_class_index = np.argmax(predictions)
predicted_class = labelencoder.inverse_transform([predicted_class_index])

print("Predicted class:", predicted_class)


