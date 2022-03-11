import tensorflow.keras as keras
import numpy as np
import librosa

MODEL_PATH = "model.h5"
SAMPLES_TO_CONSIDER = 22050  # 1 sec


class Keyword_Spotting_Service_Class:
    """
    Singleton class for keyword spotting inference with trained models.

    :param model: Trained model
    """

    model = None
    _mappings = [
        "go",
        "off",
        "right",
        "down",
        "no",
        "left",
        "on",
        "stop",
        "up",
        "yes"
    ]
    _instance = None

    def predict(self, file_path):

        # extract MFCCs
        MFCCs = self.preprocess(file_path)

        # convert 2d MFCCs array into 4d array -> (# samples, # segments, # coefficients, # channels)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # make prediciton
        predicitions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predicitions)
        predicted_keyword = self._mappings[predicted_index]

        return predicted_keyword

    def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
        """Extract MFCCs from audio file.
        :param file_path (str): Path of audio file
        :param num_mfcc (int): # of coefficients to extract
        :param n_fft (int): Interval we consider to apply STFT. Measured in # of samples
        :param hop_length (int): Sliding window for STFT. Measured in # of samples
        :return MFCCs (ndarray): 2-dim array with MFCC data of shape (# time steps, # coefficients)
        """

        # load audio file
        signal, sample_rate = librosa.load(file_path)

        # ensure consistency in the audio file length
        if len(signal) >= SAMPLES_TO_CONSIDER:
            # ensure consistency of the length of the signal
            signal = signal[:SAMPLES_TO_CONSIDER]

            # extract MFCCs
            MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                         hop_length=hop_length)
        return MFCCs.T


def Keyword_Spotting_Service():
    """Factory function for Keyword_Spotting_Service class.
    :return _Keyword_Spotting_Service._instance (_Keyword_Spotting_Service):
    """

    # ensure that we only have 1 instance of KSS
    if Keyword_Spotting_Service_Class._instance is None:
        Keyword_Spotting_Service_Class._instance = Keyword_Spotting_Service_Class()
        Keyword_Spotting_Service_Class.model = keras.models.load_model(
            MODEL_PATH)
    return Keyword_Spotting_Service_Class._instance


if __name__ == "__main__":

    # create 2 instances of the keyword spotting service
    kss = Keyword_Spotting_Service()
    kss1 = Keyword_Spotting_Service()

    # check that different instances of the keyword spotting service point back to the same object (singleton)
    assert kss is kss1

    # make a prediction
    keyword = kss.predict("down.wav")
    print(keyword)
