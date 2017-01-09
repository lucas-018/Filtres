from STFT import *
from math import *
import numpy as np
import matplotlib.pyplot as pl
from scipy import fftpack as ff
from scipy.io.wavfile import read
from scipy.io.wavfile import write
from cmath import *
from pylab import *
import sys, os
import scipy


test_1 = False
test_2 = False
test_3 = False
test_4 = True
test_5 = False


FRAME = 1000
HOP = FRAME//2
FS = 1
FILE = "Enr/Enr_16.wav"


if test_1:
    F, X_I = read(FILE)
    N = len(X)
    t = [i for i in range(N)]
    pl.plot(t, X);
    pl.show()
    write("resTest.wav", F, X)

if test_2:
    F, X = read(FILE)
    S = puts(X)
    Y = stft(S, FS, FRAME, HOP)
    spectrogramme(Y)

if test_3:
    F, X = read(FILE)
    N = len(X)
    S = puts(X)
    plot_signal(S)
    Y = cut_ecart(S, FS, FRAME, HOP, 100)
    plot_signal(Y)
    pl.show()
    Z = reputs(Y)
    write("resTest.wav", F, Z)

if test_4:
    F, X = read(FILE)
    N = len(X)
    S = puts(X)
    plot_signal(S)
    Y = passe_bas(S, 300, 2, F)
    plot_signal(Y)
    pl.show()
    Z = reputs(Y)
    write("resPasseBas.wav", F, Z)

if test_5:
    F, X = read(FILE)
    N = len(X)
    S = puts(X)
    plot_signal(S)
    Y = passe_bas_stft(S, 300, FS, FRAME, HOP, F)
    plot_signal(Y)
    pl.show()
    Z = reputs(Y)
    write("resPasseBasSTFT.wav", F, Z)
