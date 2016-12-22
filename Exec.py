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
test_3 = True
test_4 = False



FRAME = 1000
HOP = FRAME//2
FS = 1



if test_1:
    F, X_I = read("Enr/Enr_9.wav")
    N = len(X)
    t = [i for i in range(N)]
    pl.plot(t, X);
    pl.show()
    write("resTest.wav", F, X)

if test_2:
    F, X = read("Enr/Enr_9.wav")
    S = puts(X)
    Y = stft(S, FS, FRAME, HOP)
    spectrogramme(Y)

if test_3:
    F, X = read("Enr/Enr_9.wav")
    N = len(X)
    S = puts(X)
    plot_signal(S)
    Y = filtre_ecart(S, FS, FRAME, HOP, 100)
    plot_signal(Y)
    pl.show()
    Z = reputs(Y)
    write("resTest.wav", F, Z)
