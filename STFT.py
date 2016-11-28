import numpy as np
from scipy.signal import get_window
from scipy import fftpack as ff
from math import *
import matplotlib.pyplot as pl
import wave
from cmath import *
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import pickle
import scipy, pylab




def dft(x, k, s):
    N = len(x)
    res = 0
    for n in range(N):
        res += rect(x[n], 2*s*pi*k*n/N)
    return res

def FT(x):
    return [dft(x, k, -1) for k in range(len(x))]

def IFT(X):
    return [((1/len(X))*dft(X, n, 1)).real for n in range(len(X))]

def spec(x, Fe, Fmax=100000):
    X = ff.fft(x)
    M = min(len(X)//2, Fmax)
    abX = [sqrt(abs(X[k])) for k in range(M)]
    tab = np.linspace(0, M , M)
    pl.grid(True)
    pl.plot(tab, abX)
    pl.show()

def spec_3d(x, Fe):
    fig = pl.figure()
    ax = fig.gca(projection = '3d')
    X = ff.fft(x)
    Xre = [X[k].real for k in range(len(X))]
    Xim = [X[k].imag for k in range(len(X))]
    for j in range(len(X)):
        if Xre[j] != 0:
            Xre[j] /= sqrt(abs(Xre[j]))
        if Xim[j] != 0:
            Xim[j] /= sqrt(abs(Xim[j]))
    Z = [0 for i in range(len(X))]
    tab = np.linspace(0, 2*Fe, len(X))
    ax.plot(tab, Xre, Z)
    ax.plot(tab, Z, Xim)
    ax.legend()
    pl.show()

def hanning_w(M, n):
    if n < -M/2.0 or n > M/2.0:
        return 0
    else:
        return (1/2)*(1 + cos(2*pi*n/M))




def passe_bas(x, f_c, Fe):
    X = ff.fft(x)
    for p in range(int(len(X)/2)):
        X[p] = X[p]/(1 + complex(0,p*Fe/f_c))
        X[len(X)-p-1] = X[p]
    y = ff.ifft(X)
    return y

def passe_haut(x, f_c, Fe):
    X = ff.fft(x)
    for p in range(int(len(X)/2)):
        X[p] = X[p]*complex(0, p*Fe/fc)/(1+complex(0, p*Fe/f_c))
        X[len(X)-p-1] = X[p]
    y = ff.ifft(X)
    return y


def keep_reco(x, Spectre, alpha):
    X = ff.fft(x)
    ecart = 0
    for k in range(len(X)):
        ecart = abs(X[k])-Spectre[k]
        X[k] = X[k] *(1 - alpha*ecart[k])
    y = ff.ifft(X)
    return y



def cut(x, f1, f2):
    assert f1 <= f2
    X = ff.fft(x)
    N = len(X)
    for i in range(1,N//2+1):
        if i < f1*(N//2):
            X[i] = 0
            X[N-i] = 0
        elif i > f2*(N//2):
            X[i] = 0
            X[N-i] = 0
    y = ff.ifft(X)
    for i in range(len(y)):
        y[i] = (y[i].real)
    return y


def get_wav(file_name):
    file_path = "Enr/" + file_name
    fichier = wave.open(file_path, 'r')
    Fe = fichier.getframerate()
    size_w = fichier.getnframes()
    data_b = fichier.readframes(size_w)
    data = []
    for i in range(len(data_b)):
        if type(data_b[i]) is not int:
            break
        else:
            data += [data_b[i]]
    fichier.close()
    return data, Fe

def draw(signal, start, stop):
    size_s = stop - start
    tab = np.linspace(start, stop, size_s)
    signal_cut = signal[start:stop]
    pl.plot(tab, signal_cut)
    pl.show()



def set_wav(file_name, data, Fe):
    file_path1 = "Enr/" + file_name
    fichier = wave.open(file_path1, 'w')
    fichier.setparams((2, 2, Fe, len(data), 'NONE', 'not compressed'))
    for i in range(len(data)):
        data_b =wave.struct.pack('B', min(max(0,int(data[i])), 255))
        fichier.writeframes(data_b)
    fichier.close()


def stft(x, fs, frame, hop):
    framesamp = int(frame*fs)
    hopsamp = int(hop*fs)
    w = scipy.hanning(framesamp)
    X = scipy.array([ff.fft(w*x[i:i+framesamp]) for i in range(len(x)-framesamp, hopsamp)])
    return X

def istft(X, fs, T, hop):
    x = scipy.zeros(T*fs)
    framesamp = X.shape[1]
    hopsamp = int(hop*fs)
    for n,i in enumerate(range(len(x)-framesamp, hopsamp)):
        x[i:i+framsamp] += scipy.real(scipy.ifft[X[n]])
    return x





I_X, F_E = get_wav("Enr_3.wav")
NB = len(I_X)
Xplot = np.linspace(0, NB, NB)
Yplot = [hanning_w(500000, i - (NB//2)) for i in range(NB)]
pl.plot(Xplot, Yplot)
pl.show()
print(NB)
#spec_3d(I_X[10000:11000], F_E)
#I_Y = cut(I_X, 0, 0.5)
#spec(I_X, F_E)
#set_wav("Res_3.wav", I_Y, F_E)
