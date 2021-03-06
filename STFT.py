import numpy as np
from scipy.signal import get_window
from scipy import fftpack as ff
from math import *
from scipy.io.wavfile import read
from scipy.io.wavfile import write
import matplotlib.pyplot as pl
import wave
from cmath import *
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import pickle
import scipy, pylab
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import sys, os



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

def spec(x, Fe, Fmax=10000):
    X = ff.fft(x)
    M = min(len(X)//2, Fmax)
    abX = [sqrt(abs(X[k])) for k in range(1,M)]
    tab = np.linspace(1, M , M-1)
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
    if n < 0 or n >= M:
        return 0
    else:
        return (1/2)*(1 + cos(2*pi*(n-M//2)/M))



def passe_bas(x, f_c, alpha, Fe):
    X = ff.fft(x)
    for p in range(int(len(X)/2)):
        X[p] = alpha*X[p]/(1 + complex(0,p*Fe/(len(X)/2*f_c)))
        X[len(X)-p-1] = X[p]
    y = ff.ifft(X)
    return y

def passe_haut(x, f_c, Fe):
    X = ff.fft(x)
    for p in range(int(len(X)/2)):
        X[p] = alpha*X[p]*(complex(0, p*Fe/(len(X)*f_c))/(1+complex(0, p*Fe/(len(X)/2*f_c))))
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



def put_signal(x_import):
    x_s = []
    for i in range(len(x_import)):
        if x_import[i]>127.5:
            x_s += [x_import[i]-256]
        else:
            x_s += [x_import[i]]
    return x_s

def put_swav(signal):
    x_w = []
    for i in range(len(signal)):
        if signal[i] < 0:
            x_w += [min(256+signal[i],255)]
        else:
            x_w += [signal[i]]
    return x_w


def mediane(tab):
    tableau = tab
    tableau.sort()
    if len(tableau)%2 == 0:
        return (tableau[(len(tableau)//2)-1]+tableau[len(tableau)//2])/2.0
    else:
        return tableau[len(tableau)//2]

def filtre_median(x, width):
    y = []
    for i in range(width):
        y += [x[i]]
    for i in range(width,len(x)-width):
        y += [mediane(x[i-width:i+width+1])]
    for i in range(len(x)-width, len(x)):
        y += [x[i]]
    return y

def puts(x):
    s = [x[i][0] for i in range(len(x))]
    return s

def reputs(s):
    x = [[s[i], s[i]] for i in range(len(s))]
    y = np.asarray(x, dtype=np.int16)
    return y


def moyenne(tab):
    moy = 0
    for i in range(len(tab)):
        moy += tab[i]
    moy = moy/len(tab)
    return moy

def filtre_moyen(x, width):
    y = []
    for i in range(width):
        y += [x[i]]
    for i in range(width, len(x)-width):
        y += [moyenne(x[i-width:i+width+1])]
    for i in range(len(x)-width, len(x)):
        y += [x[i]]
    return y




def stft(x, fs, frame, hop):
    framesamp = int(frame*fs)
    hopsamp = int(hop*fs)
    w = scipy.hanning(framesamp)
    X = scipy.array([ff.fft(w*x[i:i+framesamp]) for i in range(0,len(x)-framesamp, hopsamp)])
    return X

def istft(X, fs, npts, hop):
    x = scipy.zeros(npts)
    framesamp = X.shape[1]
    hopsamp = int(hop*fs)
    for n,i in enumerate(range(0,len(x)-framesamp, hopsamp)):
        x[i:i+framesamp] += scipy.real(ff.ifft(X[n]))
    return x


def passe_haut_stft(x, f_c, fs, frame, hop, Fe):
    X = stft(x, fs, frame, hop)
    for t in range(len(X)):
        for p in range(len(X[t])):
            X[t][p] = X[t][p]*(complex(0, p*Fe/(len(X)*f_c))/(1+complex(0, p*Fe/(len(X)*f_c))))
    y = istft(X, fs, len(x), hop)
    return y

def passe_bas_stft(x, f_c, fs, frame, hop, Fe):
    X = stft(x, fs, frame, hop)
    for t in range(len(X)):
        for p in range(len(X[t])):
            X[t][p] = X[t][p]/(1 + complex(0,p*Fe/(len(X)*f_c)))
    y = istft(X, fs, len(x), hop)
    return y


def spectrogramme(Y_stft):
        div = 50
        deb = 0
        start  = 200
        divtps = 1
        x = np.linspace(start, len(Y_stft)//divtps, len(Y_stft)//divtps-start)
        y = np.linspace(deb, len(Y_stft[0])//div, len(Y_stft[0])//div-deb)
        z = []
        m = pow(10,5)
        M = 0
        for j in range(deb, len(Y_stft[0])//div):
            l=[]
            for i in range(start, len(Y_stft)//divtps):
                a = (abs(Y_stft[i][j]))
                if a < m:
                    m = a
                if a > M:
                    M = a
                l += [a]
            z += [l]
        levels1 = MaxNLocator(nbins = 15).tick_values(m, M)
        cmap1 = pl.get_cmap('PiYG')
        norm1 = BoundaryNorm(levels1, ncolors = cmap1.N, clip = True)
        fig, ax1 = pl.subplots(nrows = 1)
        """im = ax0.pcolormesh(x, y, z, cmap = cmap1, norm = norm1)
        fig.colorbar(im, ax = ax0)
        ax0.set_title('STFT1')"""
        cf = ax1.contourf(x, y, z, levels = levels1, cmap = cmap1)
        fig.colorbar(cf, ax = ax1)
        ax1.set_title('STFT2')
        fig.tight_layout()
        pl.show()

def Var(x):
    V = 0
    E = moyenne(x)
    for i in range(len(x)):
        V += pow(x[i]-E, 2)
    V = V/len(x)
    return V

def cut_ecart(x, fs, frame, hop, alpha):
    sigma = sqrt(Var(x))
    npts = len(x)
    print(sigma)
    print(moyenne(x))
    X = stft(x, fs, frame, hop)

    for i in range(len(X)):
        for j in range(len(X[i])):
            if abs(X[i][j]) > alpha*sigma:
                X[i][j] = 0
            #X[i][j] = min(abs(X[i][j]), alpha*sigma)*rect(1,phase(X[i][j]))
    y = istft(X, fs, npts, hop)
    print(len(y))
    return y

def filtre_ecart(x, fs, frame, hop, alpha):
    sigma = sqrt(Var(x))
    npts = len(x)
    print(sigma)
    print(moyenne(x))
    X = stft(x, fs, frame, hop)

    for i in range(len(X)):
        for j in range(len(X[i])):
            #if abs(X[i][j]) > alpha*sigma:
                #X[i][j] = 0
            X[i][j] = min(abs(X[i][j]), alpha*sigma)*rect(1,phase(X[i][j]))
    y = istft(X, fs, npts, hop)
    print(len(y))
    return y


def plot_signal(x):
    t = [i for i in range(len(x))]
    pl.plot(t, x)
