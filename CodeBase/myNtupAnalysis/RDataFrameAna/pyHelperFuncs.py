import re
import sys
import time
import array
import ROOT as R
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
import plottingTool as pt
import plotly.express as px
import matplotlib.pyplot as plt
from os import listdir, remove, walk
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

R.gROOT.SetBatch(True)

myLighterBlue=R.kAzure-2
myLightBlue  =R.TColor.GetColor('#9ecae1')
myMediumBlue =R.TColor.GetColor('#0868ac')
myDarkBlue   =R.TColor.GetColor('#08306b')

# Greens
myLightGreen   =R.TColor.GetColor('#c7e9c0')
myMediumGreen  =R.TColor.GetColor('#41ab5d')
myDarkGreen    =R.TColor.GetColor('#006d2c')

# Oranges
myLighterOrange=R.TColor.GetColor('#ffeda0')
myLightOrange  =R.TColor.GetColor('#fec49f')
myMediumOrange =R.TColor.GetColor('#fe9929')

# Greys
myLightestGrey=R.TColor.GetColor('#f0f0f0')
myLighterGrey=R.TColor.GetColor('#e3e3e3')
myLightGrey  =R.TColor.GetColor('#969696')

# Pinks
myLightPink = R.TColor.GetColor('#fde0dd')
myMediumPink = R.TColor.GetColor('#fcc5c0')
myDarkPink = R.TColor.GetColor('#dd3497')



bkgdic = {"Zjets":{"color":myMediumGreen},
          "Wjets":{"color":myLightGreen},
          "ttbar":{"color":myMediumBlue},
          "singletop":{"color":myLightBlue},
          "Diboson":{"color":myMediumOrange},
          "data22":{"color":R.kBlack}
}

bkgdic = {'Wjets':{"color":myLightGreen},
          'Zeejets':{"color":myMediumGreen},
          'diboson2L':{"color":myMediumOrange},
          'diboson3L':{"color":myLighterOrange},
          'diboson4L':{"color":myLightOrange},
          'higgs':{"color":myLightPink},
          'singletop':{"color":myLightBlue},
          'topOther':{"color":myDarkBlue},
          'triboson':{"color":myDarkPink},
          'ttbar':{"color":myMediumBlue},
          'Zmmjets':{"color":myLightGreen},
          'Zttjets':{"color":myDarkGreen},
          "data15":{"color":R.kBlack},
          "data16":{"color":R.kBlack},
          "data17":{"color":R.kBlack},
          "data18":{"color":R.kBlack}
}


trgdic = {"2022":{"1L":["HLT_mu24_ivarmedium_L1MU14FCH",
                        "HLT_mu60_L1MU14FCH",
                        "HLT_mu50_L1MU14FCH",
                        "HLT_e26_lhtight_ivarloose_L1EM22VHI",
                        "HLT_e60_lhmedium_L1EM22VHI",
                        "HLT_e300_etcut_L1EM22VHI"],
                  "2L":["HLT_2mu10_l2mt_L1MU10BOM",
                        "HLT_2mu14_L12MU8F",
                        "HLT_mu20_ivarmedium_mu8noL1_L1MU14FCH",
                        "HLT_mu22_mu8noL1_L1MU14FCH",
                        "HLT_2e17_lhvloose_L12EM15VHI",
                        "HLT_e26_lhtight_e14_etcut_50invmAB130_L1EM22VHI"
                  ],
                  "3L":["1"]},
          "2015":{"1L":["HLT_e24_lhmedium_L1EM20VH",
                        "HLT_e60_lhmedium",
                        "HLT_e120_lhloose",
                        "HLT_mu20_iloose_L1MU15",
                        "HLT_mu50"],
                  "2L":["HLT_2e12_lhloose_L12EM10VH",
                        "HLT_2mu10",
                        "HLT_mu18_mu8noL1",
                        "HLT_e17_lhloose_mu14",
                        "HLT_e7_lhmedium_mu24"
                  ],
                  "3L":["HLT_e17_lhloose_2e9_lhloose",
                        "HLT_mu18_2mu4noL1",
                        "HLT_2e12_lhloose_mu10",
                        "HLT_e12_lhloose_2mu10"
                  ]},   
          "2016":{"1L":["HLT_e24_lhmedium_nod0_L1EM20VH",
                        "HLT_e24_lhtight_nod0_ivarloose",
                        "HLT_e26_lhtight_nod0_ivarloose",
                        "HLT_e60_lhmedium_nod0",
                        "HLT_e140_lhloose_nod0",
                        "HLT_mu26_ivarmedium",
                        "HLT_mu50"],
                  "2L":["HLT_2e15_lhvloose_nod0_L12EM13VH",
                        "HLT_2e17_lhvloose_nod0",
                        "HLT_2mu10",
                        "HLT_2mu14",
                        "HLT_mu20_mu8noL1",
                        "HLT_mu22_mu8noL1",
                        "HLT_e17_lhloose_nod0_mu14",
                        "HLT_e24_lhmedium_nod0_L1EM20VHI_mu8noL1",
                        "HLT_e7_lhmedium_nod0_mu24"
                  ],
                  "3L":["HLT_e24_lhvloose_nod0_2e12_lhvloose_nod0_L1EM20VH_3EM10VH",
                        "HLT_e12_lhloose_nod0_2mu10",
                        "HLT_2e12_lhloose_nod0_mu10",
                        "HLT_mu20_2mu4noL1",
                        "HLT_3mu6",
                        "HLT_3mu6_msonly",
                        "HLT_e17_lhloose_nod0_2e10_lhloose_nod0_L1EM15VH_3EM8VH"
                  ]},
          "2017":{"1L":["HLT_e26_lhtight_nod0_ivarloose",
                        "HLT_e60_lhmedium_nod0",  
                        "HLT_e140_lhloose_nod0",    
                        "HLT_e300_etcut",                                
                        "HLT_mu26_ivarmedium",	     
                        "HLT_mu50"]
                  ,"2L":["HLT_2e17_lhvloose_nod0_L12EM15VHI",
                         "HLT_2e24_lhvloose_nod0",
                         "HLT_2mu14",
                         "HLT_mu22_mu8noL1",
                         "HLT_e17_lhloose_nod0_mu14",
                         "HLT_e26_lhmedium_nod0_mu8noL1",
                         "HLT_e7_lhmedium_nod0_mu24"
                  ],
                  "3L":["HLT_e24_lhvloose_nod0_2e12_lhvloose_nod0_L1EM20VH_3EM10VH",
                        "HLT_e12_lhloose_nod0_2mu10",
                        "HLT_2e12_lhloose_nod0_mu10",
                        "HLT_mu20_2mu4noL1",
                        "HLT_3mu6",
                        "HLT_3mu6_msonly"
                  ]},
          "2018":{"1L":["HLT_e26_lhtight_nod0_ivarloose",
                        "HLT_e60_lhmedium_nod0",  
                        "HLT_e140_lhloose_nod0",    
                        "HLT_e300_etcut",                                
                        "HLT_mu26_ivarmedium",	     
                        "HLT_mu50"],
                  "2L":["HLT_2e17_lhvloose_nod0_L12EM15VHI",
                        "HLT_2e24_lhvloose_nod0",
                        "HLT_2mu14",
                        "HLT_mu22_mu8noL1",
                        "HLT_e17_lhloose_nod0_mu14",
                        "HLT_e26_lhmedium_nod0_mu8noL1",
                        "HLT_e7_lhmedium_nod0_mu24"],
                  "3L":["HLT_e24_lhvloose_nod0_2e12_lhvloose_nod0_L1EM20VH_3EM10VH",
                        "HLT_e12_lhloose_nod0_2mu10",
                        "HLT_2e12_lhloose_nod0_mu10",
                        "HLT_mu20_2mu4noL1",
                        "HLT_3mu6"
                  ]}
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
        if not len(trgdic[yr][x]): continue
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
                trigstr[x][yr] += "(lep%s && lepPt > %i) || "%(trigger,getTriggerThreshold(trigger))
                evtrigstr[x][yr] += "trigMatch_%s || "%(trigger)
        trigstr[x][yr] = trigstr[x][yr][:-4]+")"
        evtrigstr[x][yr] = evtrigstr[x][yr][:-4]+")"



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


def writeHistsToFile(histo, d_samp, writetofile = True):
    for k in histo.keys():
        col = -1
        sp = k.split("_")
        typ = ""
        for i in range(len(sp)):
            s = "_".join(sp[i:])
            if s in d_samp.keys():
                typ = s
        if not typ:
            print("Did to find match for key %s"%k)
            continue
        #for plk in d_samp.keys():
        #    if plk == typ:
        #print(typ)
        evtyp = list(fldic.keys())
        if "flcomp" in k:
            for i in range(1,histo[k].GetNbinsX()+1):
                histo[k].GetXaxis().SetBinLabel(i,evtyp[i-1])
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
        cl = R.gROOT.GetClass(key.GetClassName());
        if cl.InheritsFrom("TH1D") or cl.InheritsFrom("TH2D"):
            obj = key.ReadObj()
            histo[obj.GetName().replace("h_","")] = obj.Clone()
            histo[obj.GetName().replace("h_","")].SetDirectory(0)
            #print(obj.GetName())
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

def getDataFrames1(mypath, nev = 0):
    
    onlyfiles = []
    for path,dirs,files in walk(mypath):
        #print(path,dirs,files)
        for f in files:
            if isfile(join(path, f)) and f.endswith("_merged_processed.root"):
                #print(join(path,f))
                onlyfiles.append(join(path, f))
                
    df = {}
    files = {}
    for of in onlyfiles:
        if not "merged" in of or not of.endswith(".root"): continue
        sp = of.split("/")[-1].split("_")
        typ = ""
        for s in sp:
            if "merged" in s or s.isnumeric(): break
            typ += s
        if not typ in files.keys():
            files[typ] = {"files":[], "treename":""}
        treename = getTreeName(of)
        if treename == "noname":
            print("ERROR \t Could not find any TTree in %s"%(of))
            continue
        files[typ]["treename"] = treename
        files[typ]["files"].append(of)
        
        #print(typ)
        #if not typ == "singleTop": continue
        #df[typ] = R.Experimental.MakeNTupleDataFrame("mini",of)#("%s_NoSys"%typ,of)
    for typ in files.keys():
        print("Adding %i files for %s"%(len(files[typ]["files"]),typ))
        df[typ] = R.RDataFrame(files[typ]["treename"],files[typ]["files"])
        if nev:
            df[typ] = df[typ].Range(nev)
    return df


def getRatio1D(hT,hL,vb=0):
    asym = R.TGraphAsymmErrors();
    hR = hT.Clone(hT.GetName().replace("hT","hE"))
    hR.Divide(hT,hL,1.,1.,'b')
    if vb: print(":::->Dividing T = %.2f on L = %.2f" %(hT.Integral(),hL.Integral()))
    asym.Divide(hT,hL,"cl=0.683 b(1,1) mode")
    for i in range(0,hR.GetNbinsX()+1):
        hR.SetBinError(i+1,asym.GetErrorY(i))
    return hR



def plot_rmm_matrix(
    df: pd.DataFrame, process: str, rmm_structure: dict, N_row: int
) -> None:

    col = len(df.columns)
    row = len(df)

    
    df2 = df.mean()

    tot = len(df2)
    row = int(np.sqrt(tot))
  

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

