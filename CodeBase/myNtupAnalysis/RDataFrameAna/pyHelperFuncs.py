import re
import sys
import time
import array
import ROOT as R
import matplotlib
import numpy as np
import pandas as pd
from os import listdir
from typing import Tuple
import plottingTool as pt
import plotly.express as px
import matplotlib.pyplot as plt
from samples import configure_samples
from os.path import isfile, join, isdir


fldic = {
    "eee": 0,
    "eem": 1,
    "emm": 2,
    "mmm": 3,
    "mme": 4,
    "mee": 5,
}


bkgdic = {
    "Wjets": {"color": R.kMagenta},
    "Zjets2": {"color": R.kBlue - 7},
    "diboson2L": {"color": R.kRed - 7},
    "diboson3L": {"color": R.kBlue - 7},
    "diboson4L": {"color": R.kRed - 5},
    "Diboson": {"color": R.kOrange + 10},
    "higgs": {"color": R.kBlue + 2},
    "singletop": {"color": R.kGreen - 1},
    "topOther": {"color": R.kRed + 3},
    "triboson": {"color": R.kOrange - 5},
    "ttbar": {"color": R.kMagenta + 5},
    "data18": {"color": R.kBlack},
}

trgdic = {
    "2015": {
        "1L": [
            "HLT_e24_lhmedium_L1EM20VH",
            "HLT_e60_lhmedium",
            "HLT_e120_lhloose",
            "HLT_mu20_iloose_L1MU15",
            "HLT_mu50",
        ],
        "2L": [
            "HLT_2e12_lhloose_L12EM10VH",
            "HLT_2mu10",
            "HLT_mu18_mu8noL1",
            "HLT_e17_lhloose_mu14",
            "HLT_e7_lhmedium_mu24",
        ],
        "3L": [
            "HLT_e17_lhloose_2e9_lhloose",
            "HLT_mu18_2mu4noL1",
            "HLT_2e12_lhloose_mu10",
            "HLT_e12_lhloose_2mu10",
        ],
    },
    "2016": {
        "1L": [
            "HLT_e24_lhmedium_nod0_L1EM20VH",
            "HLT_e24_lhtight_nod0_ivarloose",
            "HLT_e26_lhtight_nod0_ivarloose",
            "HLT_e60_lhmedium_nod0",
            "HLT_e140_lhloose_nod0",
            "HLT_mu26_ivarmedium",
            "HLT_mu50",
        ],
        "2L": [
            "HLT_2e15_lhvloose_nod0_L12EM13VH",
            "HLT_2e17_lhvloose_nod0",
            "HLT_2mu10",
            "HLT_2mu14",
            "HLT_mu20_mu8noL1",
            "HLT_mu22_mu8noL1",
            "HLT_e17_lhloose_nod0_mu14",
            "HLT_e24_lhmedium_nod0_L1EM20VHI_mu8noL1",
            "HLT_e7_lhmedium_nod0_mu24",
        ],
        "3L": [
            "HLT_e24_lhvloose_nod0_2e12_lhvloose_nod0_L1EM20VH_3EM10VH",
            "HLT_e12_lhloose_nod0_2mu10",
            "HLT_2e12_lhloose_nod0_mu10",
            "HLT_mu20_2mu4noL1",
            "HLT_3mu6",
            "HLT_3mu6_msonly",
            "HLT_e17_lhloose_nod0_2e10_lhloose_nod0_L1EM15VH_3EM8VH",
        ],
    },
    "2017": {
        "1L": [
            "HLT_e26_lhtight_nod0_ivarloose",
            "HLT_e60_lhmedium_nod0",
            "HLT_e140_lhloose_nod0",
            "HLT_e300_etcut",
            "HLT_mu26_ivarmedium",
            "HLT_mu50",
        ],
        "2L": [
            "HLT_2e17_lhvloose_nod0_L12EM15VHI",
            "HLT_2e24_lhvloose_nod0",
            "HLT_2mu14",
            "HLT_mu22_mu8noL1",
            "HLT_e17_lhloose_nod0_mu14",
            "HLT_e26_lhmedium_nod0_mu8noL1",
            "HLT_e7_lhmedium_nod0_mu24",
        ],
        "3L": [
            "HLT_e24_lhvloose_nod0_2e12_lhvloose_nod0_L1EM20VH_3EM10VH",
            "HLT_e12_lhloose_nod0_2mu10",
            "HLT_2e12_lhloose_nod0_mu10",
            "HLT_mu20_2mu4noL1",
            "HLT_3mu6",
            "HLT_3mu6_msonly",
        ],
    },
    "2018": {
        "1L": [
            "HLT_e26_lhtight_nod0_ivarloose",
            "HLT_e60_lhmedium_nod0",
            "HLT_e140_lhloose_nod0",
            "HLT_e300_etcut",
            "HLT_mu26_ivarmedium",
            "HLT_mu50",
        ],
        "2L": [
            "HLT_2e17_lhvloose_nod0_L12EM15VHI",
            "HLT_2e24_lhvloose_nod0",
            "HLT_2mu14",
            "HLT_mu22_mu8noL1",
            "HLT_e17_lhloose_nod0_mu14",
            "HLT_e26_lhmedium_nod0_mu8noL1",
            "HLT_e7_lhmedium_nod0_mu24",
        ],
        "3L": [
            "HLT_e24_lhvloose_nod0_2e12_lhvloose_nod0_L1EM20VH_3EM10VH",
            "HLT_e12_lhloose_nod0_2mu10",
            "HLT_2e12_lhloose_nod0_mu10",
            "HLT_mu20_2mu4noL1",
            "HLT_3mu6",
        ],
    },
}


def getTriggerThreshold(tname):
    thr = []
    # print(tname)
    reg = re.findall(r"_\d*([e]*[mu]*\d{1,})_{0,}", tname)
    for r in reg:
        # print(int(re.sub('\D', '', r)))
        thr.append(int(re.sub("\D", "", r)))
    return max(thr)


trigstr = {}
evtrigstr = {}
for yr in trgdic.keys():
    for x in trgdic[yr].keys():
        if not len(trgdic[yr][x]):
            continue
        if not x in trigstr.keys():
            trigstr[x] = {}
            evtrigstr[x] = {}
        if not yr in trigstr[x].keys():
            trigstr[x][yr] = "("
            evtrigstr[x][yr] = "("
        for trigger in trgdic[yr][x]:
            if trigger == "1":
                trigstr[x][yr] += "(1) || "
                evtrigstr[x][yr] += "1 || "
            else:
                trigstr[x][yr] += "(lep%s && lepPt > %i) || " % (
                    trigger,
                    getTriggerThreshold(trigger),
                )
                evtrigstr[x][yr] += "trigMatch_%s || " % (trigger)
        trigstr[x][yr] = trigstr[x][yr][:-4] + ")"
        evtrigstr[x][yr] = evtrigstr[x][yr][:-4] + ")"


def convertRDFCutflowToTex(cutflow1, cutflow2):
    i = 0
    tabstr = ""
    for c in cutflow1:
        cname = c.GetName()
        c2 = cutflow2.At(cname)
        if i == 0:
            nevc1 = c.GetAll()
            nevc2 = c2.GetAll()
        cname = cname.replace(">", "$>$")
        cname = cname.replace("<", "$<$")
        tabstr += (
            "%-30s & $%.0f$ & $%.0f$ & $%.2f$ & $%.2f$ & $%.0f$ & $%.0f$ & $%.2f$ & $%.2f$ \\\ \n"
            % (
                cname,
                c.GetPass(),
                c.GetAll(),
                c.GetEff(),
                (c.GetPass() / nevc1) * 100.0,
                c2.GetPass(),
                c2.GetAll(),
                c2.GetEff(),
                (c2.GetPass() / nevc2) * 100.0,
            )
        )
        i += 1
    print(tabstr)


def writeHistsToFile(histo, d_samp, writetofile=True):
    global fldic
    for k in histo.keys():
        col = -1
        sp = k.split("_")
        typ = ""
        for i in range(len(sp)):
            s = "_".join(sp[i:])
            if s in d_samp.keys():
                typ = s
        if not typ:
            print("Did to find match for key %s" % k)
            continue
        # for plk in d_samp.keys():
        #    if plk == typ:
        # print(typ)
        evtyp = list(fldic.keys())
        if "flcomp" in k:
            for i in range(1, histo[k].GetNbinsX() + 1):
                histo[k].GetXaxis().SetBinLabel(i, evtyp[i - 1])
        if d_samp[typ]["type"] == "bkg":
            histo[k].SetFillColor(d_samp[typ]["f_color"])
            histo[k].SetLineColor(d_samp[typ]["f_color"])
            histo[k].SetMarkerStyle(0)
            histo[k].SetMarkerSize(0)
        elif d_samp[typ]["type"] == "data":
            histo[k].SetFillColor(d_samp[typ]["f_color"])
            histo[k].SetLineColor(d_samp[typ]["l_color"])
            histo[k].SetMarkerStyle(20)
        elif d_samp[typ]["type"] == "sig":
            histo[k].SetFillColor(0)
            histo[k].SetLineColor(d_samp[typ]["l_color"])
            histo[k].SetMarkerStyle(0)
            histo[k].SetMarkerSize(0)
            histo[k].SetLineStyle(9)
            histo[k].SetLineWidth(2)
        if writetofile:
            histo[k].Write()


def getHistograms(fname):
    histo = {}
    f1 = R.TFile(fname)
    dirlist = f1.GetListOfKeys()
    it = dirlist.MakeIterator()
    key = it.Next()
    while key:
        cl = R.gROOT.GetClass(key.GetClassName())
        if cl.InheritsFrom("TH1D") or cl.InheritsFrom("TH2D"):
            obj = key.ReadObj()
            histo[obj.GetName().replace("h_", "")] = obj.Clone()
            histo[obj.GetName().replace("h_", "")].SetDirectory(0)
            key = it.Next()
        else:
            key = it.Next()
            continue
    f1.Close()
    return histo


def getTreeName(fname):
    f1 = R.TFile(fname)
    dirlist = f1.GetListOfKeys()
    it = dirlist.MakeIterator()
    key = it.Next()
    while key:
        cl = R.gROOT.GetClass(key.GetClassName())
        if cl.InheritsFrom("TTree"):
            obj = key.ReadObj()
            if obj.GetName() in ["CutBookkeepers", "MetaTree"]:
                key = it.Next()
                continue
            return obj.GetName()
        else:
            key = it.Next()
            continue
    f1.Close()
    return "noname"


def getDataFrames(mypath, nev=0):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    df = {}
    files = {}
    for of in onlyfiles:
        if not "merged" in of or not of.endswith(".root"):
            continue
        sp = of.split("_")
        typ = ""
        for s in sp:
            if "merged" in s or s.isnumeric():
                break
            typ += s
        if not typ in files.keys():
            files[typ] = {"files": [], "treename": ""}
        treename = getTreeName(mypath + "/" + of)
        if treename == "noname":
            print("ERROR \t Could not find any TTree in %s" % (mypath + "/" + of))
            continue
        files[typ]["treename"] = treename
        files[typ]["files"].append(mypath + "/" + of)

        # print(typ)
        # if not typ == "singleTop": continue
        # df[typ] = R.Experimental.MakeNTupleDataFrame("mini",mypath+"/"+of)#("%s_NoSys"%typ,mypath+"/"+of)
    for typ in files.keys():
        print("Adding %i files for %s" % (len(files[typ]["files"]), typ))
        df[typ] = R.RDataFrame(files[typ]["treename"], files[typ]["files"])
        if nev:
            df[typ] = df[typ].Range(nev)
    return df


def getRatio1D(hT, hL, vb=0):
    asym = R.TGraphAsymmErrors()
    hR = hT.Clone(hT.GetName().replace("hT", "hE"))
    hR.Divide(hT, hL, 1.0, 1.0, "b")
    if vb:
        print(":::->Dividing T = %.2f on L = %.2f" % (hT.Integral(), hL.Integral()))
    asym.Divide(hT, hL, "cl=0.683 b(1,1) mode")
    for i in range(0, hR.GetNbinsX() + 1):
        hR.SetBinError(i + 1, asym.GetErrorY(i))
    return hR


def plot_rmm_matrix(
    df: pd.DataFrame, process: str, rmm_structure: dict, N_row: int
) -> None:

    col = len(df.columns)
    row = len(df)

    print("")
    print(f"Size: {row}")
    print("")
    df2 = df.mean()

    tot = len(df2)
    row = int(np.sqrt(tot))
    print(row)

    rmm_mat = np.zeros((row, row))

    df2 = df2.to_numpy()

    p = 0

    for i in range(row):
        for j in range(row):
            rmm_mat[i, j] = df2[p]
            p += 1

    names = [" "]

    for i in range(1, N_row):
        name = rmm_structure[i][0]
        names.append(name)

    """fig, ax = plt.subplots()

    im, cbar = heatmap(rmm_mat, names, names, ax=ax, cbarlabel="Intensity")
    texts = annotate_heatmap(im, valfmt="{x:.3f}")

    im = ax.imshow(rmm_mat)

    fig.tight_layout()

    plt.savefig(f"../../../Figures/histo_var_check/rmm_avg_{process}.pdf")
    plt.show()"""
    
    rmm_mat[rmm_mat == 0] = np.nan
    

    fig = px.imshow(rmm_mat,
                    labels=dict(x="Particles", y="Particles", color="Intensity"),
                    x=names,
                    y=names,
                    aspect="auto",
                    color_continuous_scale='Viridis',
                    text_auto=".3f"
            )
    fig.update_xaxes(side="top")
    
    fig.write_image(f"../../../Figures/histo_var_check/rmm_avg_{process}.pdf")


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    **textkw,
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def get_column_names(df, histo):
    names = histo.keys()

    new_feats = []
    for name in names:
        if name[-5:] == "higgs":
            new_feats.append(name[:-6])

    all_cols = []
    for c in df["higgs"].GetColumnNames():
        if c in new_feats:
            all_cols.append(str(c))

    extra = [
        "wgt_SG",
        "flcomp",
        "ele_0_charge",
        "ele_1_charge",
        "ele_2_charge",
        "muo_0_charge",
        "muo_1_charge",
        "muo_2_charge",
    ]
    for col in extra:
        all_cols.append(col)

    return all_cols


def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res