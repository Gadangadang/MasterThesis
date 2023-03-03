import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from os.path import isfile, join
from sklearn.metrics import roc_curve, RocCurveDisplay, auc

from Utilities.config import *
from Utilities.pathfile import *

plt.style.use("bmh")
sns.color_palette("hls", 1)

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def f(x):
    return x

c = 0
for mu, sigma, mu2, sigma2 in [(0.3, 0.1, 0.5, 0.1),(0.41, 0.3, 0.49, 0.3)]:
    print( mu, sigma, mu2, sigma)
    if c == 0:
        hist_name = "Sep"
    else:
        hist_name = "bad"
    c += 1

    
    s = np.random.normal(mu, sigma, 1000)

    s_lab = np.zeros(1000)

    
    s2 = np.random.normal(mu2, sigma2, 1000)

    s2_lab = np.ones(1000)

    count, bins, ignored = plt.hist([s, s2], 30, density=False, label=[fr"$\mu$: {mu}, $\sigma$: {sigma}", fr"$\mu$: {mu2}, $\sigma$: {sigma2}"])

    plt.legend()
    plt.title("Example using two normal distributions", fontsize=25)
    plt.ylabel("#Number of events", fontsize=25)
    plt.xlabel("Output of the distributions", fontsize=25)
    plt.savefig(STORE_IMG_PATH / Path(f"histo_example_{hist_name}.pdf"))
    plt.close()

    label = np.concatenate((s_lab, s2_lab))

    scores = np.concatenate((s,s2))

    fpr, tpr, thresholds = roc_curve(label, scores, pos_label=1)
    sorted_index = np.argsort(fpr)
    fpr =  np.array(fpr)[sorted_index]
    tpr = np.array(tpr)[sorted_index]

    roc_auc = auc(fpr,tpr)
    
    x = np.linspace(0,1, 1001)

    #RocCurveDisplay.from_predictions(label, scores, sample_weight=weights)
    plt.plot(fpr, tpr, label=f"AUC score: {roc_auc:.2f}")
    plt.plot(x, f(x), "k--")
    plt.xlabel("False positive rate", fontsize=25)
    plt.ylabel("True positive rate", fontsize=25)
    plt.legend()
    plt.title(f"ROC curve of two normal distributions", fontsize=25)
    plt.savefig(STORE_IMG_PATH / Path(f"roc_curve_example_{hist_name}.pdf"))
    plt.close()
