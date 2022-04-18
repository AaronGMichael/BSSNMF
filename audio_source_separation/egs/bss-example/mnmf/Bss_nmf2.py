import sys

from scipy.io import wavfile
sys.path.append("../../../src")
import numpy as np
import scipy.signal as ss
import soundfile as sf
import IPython.display as ipd
import matplotlib.pyplot as plt
from bss.mnmf import FastMultichannelISNMF as FastMNMF


def Bss_nmf():
    plt.rcParams['figure.dpi'] = 200
    filenamesrc1 = "Bss_numb_Vocal.wav"
    source1, sr = sf.read("../../../dataset/sample-song/" + filenamesrc1)
    filenamesrc2 = "Bss_numb_piano.wav"
    source2, sr = sf.read("../../../dataset/sample-song/" + filenamesrc2)
    y = np.vstack([source1, source2])
    mixturesrc = "Bss_numb_mix1.wav"
    mixture, sr = sf.read("../../../dataset/sample-song/" + mixturesrc)
    x = mixture.T
    n_channels, T = x.shape
    n_sources = n_channels
    for idx in range(n_channels):
        ipd.display(ipd.Audio(x[idx], rate=sr))
    fft_size, hop_size = 4096, 2048
    _, _, X = ss.stft(x, nperseg=fft_size, noverlap=fft_size-hop_size)
    np.random.seed(111)
    mnmf = FastMNMF(n_basis=4)
    print(mnmf)
    Y = mnmf(X, iteration=50)
    _, y = ss.istft(Y, nperseg=fft_size, noverlap=fft_size-hop_size)
    y = y[:, :T]
    for idx in range(n_sources):
        ipd.display(ipd.Audio(y[idx], rate=sr))
        wavfile.write('Outputs/output_' + mixturesrc + '_' + str(idx) + '.wav', sr, y[idx])
    plt.figure()
    plt.plot(mnmf.loss, color='black')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

if __name__ == '__main__':
    Bss_nmf()
