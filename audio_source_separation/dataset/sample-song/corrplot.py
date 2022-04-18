import sklearn
import numpy as np
import pandas as pd
from pydub import AudioSegment
import soundfile as sf
from scipy import signal
from scipy.io import wavfile
from matplotlib import pyplot as plt
from numpy import sin, linspace, pi
from pylab import plot, show, title, xlabel, ylabel, subplot
from numpy import arange
import seaborn as sns
import librosa


def plot_correlation(src_name, tgt_name):

    #SOURCE_FILE, sampling_rate = sf.read(src_name)
    #TARGET_FILE, sampling_rate2 = sf.read("../../egs/bss-example/mnmf/Outputs/" + tgt_name)
    SOURCE_FILE, sampling_rate = librosa.load(src_name, sr=8000)  # Downsample 44.1kHz to 8kHz
    TARGET_FILE, sampling_rate2 = librosa.load("../../egs/bss-example/mnmf/Outputs/" + tgt_name, sr=8000)  # Downsample 44.1kHz to 8kHz
    y = pd.Series(SOURCE_FILE)
    x = pd.Series(TARGET_FILE)
    correlation = y.corr(x)
    print(correlation)

    # plt.suptitle('String 4 Source Correlation')
    # plt.scatter(x, y)
    # plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='red')
    # plt.xlabel('x axis')
    # plt.ylabel('y axis')
    # plt.title("correlation: " + str(correlation))
    # # plt.xlim([4, 6])
    # # plt.ylim([-4e8, 8e8])
    # plt.show()
    X = mix_sources([SOURCE_FILE, TARGET_FILE])
    fig = plt.figure()
    plt.subplot(4, 1, 1)
    plt.suptitle('Correlation for ' + src_name)
    plt.scatter(x, y)
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='red')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title("correlation: " + str(correlation))
    plt.subplot(4, 1, 2)
    for s in X:
        plt.plot(s, alpha=0.8)
    plt.title("Time Domain Envelope")
    plt.subplot(4, 1, 3)
    fft_spectrum = np.fft.rfft(SOURCE_FILE)
    freq = np.fft.rfftfreq(SOURCE_FILE.size, d=1 / sampling_rate)
    fft_spectrum_abs = np.abs(fft_spectrum)
    plt.plot(freq, fft_spectrum_abs, color='orangered')
    plt.xlim([0, 5000])
    plt.xlabel("frequency, Hz")
    plt.ylabel("Amplitude, units")
    plt.subplot(4, 1, 4)
    fft_spectrum = np.fft.rfft(TARGET_FILE)
    freq = np.fft.rfftfreq(TARGET_FILE.size, d=1. / sampling_rate2)
    fft_spectrum_abs = np.abs(fft_spectrum)
    plt.plot(freq, fft_spectrum_abs, color='orangered', )
    plt.xlim([0, 5000])
    plt.xlabel("frequency, Hz")
    plt.ylabel("Amplitude, units")
    fig.tight_layout()
    plt.show()

def mix_sources(mixtures, apply_noise=False):
    for i in range(len(mixtures)):

        max_val = np.max(mixtures[i])

        if max_val > 1 or np.min(mixtures[i]) < 1:
            mixtures[i] = mixtures[i] / (max_val / 2) - 0.5

    X = np.c_[[mix for mix in mixtures]]

    if apply_noise:
        X += 0.02 * np.random.normal(size=X.shape)

    return X

if __name__ == '__main__':
    src = "Bss_numb_piano.wav"
    tgt = "output_Bss_numb_mix1.wav_0.wav"
    plot_correlation(src, tgt)
