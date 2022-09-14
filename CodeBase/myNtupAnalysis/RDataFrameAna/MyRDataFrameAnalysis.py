import ROOT as R
from os import listdir
from os.path import isfile, join, isdir
import array
import time
import sys
from samples import configure_samples
import numpy as np
import pandas as pd
import plottingTool as pt

d_samp,d_type,d_reg = configure_samples()#False,False,True,False,False)

R.EnableImplicitMT(200)

R.gROOT.ProcessLine(".L helperFunctions.cxx+");
R.gSystem.AddDynamicPath("-I/home/sgfrette/myNtupAnalysis/RDataFrameAna")
R.gInterpreter.Declare('#include "helperFunctions.h"') # Header with the definition of the myFilter function
R.gSystem.Load("helperFunctions_cxx.so") # Library with the myFilter function

from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

hname = "MET_2L_mm"


fldic = {"eee":0,
         "eem":1,
         "emm":2,
         "mmm":3,
         "mme":4,
         "mee":5,
         "ee":6,
         "mm":7,
         "em":8,
         "all":9
}

# ZjetsPH_merged_processed_1.root
# Diboson
# WjetsPH
# ZjetsPH
# ZjetsPH_merged_processed__1.root
# DibosonPH
# XGamma
# Triboson
# PythiaB
# ttbar
# Wjets
# Zjets
# lowMassDY

# bkgdic = {"XGamma":{"color":R.kMagenta},
#           "Wjets":{"color":R.kBlue-7},
#           "Zjets":{"color":R.kRed-7},
#           "WjetsPH":{"color":R.kBlue-7},
#           "ZjetsPH":{"color":R.kRed-7},
#           "Diboson":{"color":R.kOrange+10},
#           "DibosonPH":{"color":R.kOrange+10},
#           "higgs":{"color":R.kPink-9},
#           "lowMassDY":{"color":R.kCyan-4},
#           "Triboson":{"color":R.kViolet-7},
#           "ttbar":{"color":R.kYellow+2},
#           "data18":{"color":R.kBlack}
# }

bkgdic = {"Wjets":{"color":R.kMagenta},
          "Zjets2":{"color":R.kBlue-7},
          "diboson2L":{"color":R.kRed-7},
          "diboson3L":{"color":R.kBlue-7},
          "diboson4L":{"color":R.kRed-5},
          "Diboson":{"color":R.kOrange+10},
          "higgs":{"color": R.kBlue+2},
          "singletop":{"color":R.kGreen-1},
          "topOther":{"color":R.kRed+3},
          "triboson":{"color":R.kOrange-5},
          "ttbar":{"color":R.kMagenta+5},
          "data18":{"color":R.kBlack}
}


# bkgdic = {"PHWW":{"color":R.kGreen},
#           "PHWZ":{"color":R.kGreen+4},
#           "PHZZ":{"color":R.kGreen-5},
#           "Vgamma":{"color":R.kMagenta},
#           "Wjets_extension":{"color":R.kBlue+1},
#           "Wjets":{"color":R.kBlue-7},
#           "Zjets_extension":{"color":R.kRed},
#           "Zjets":{"color":R.kRed-7},
#           "diboson":{"color":R.kOrange+10},
#           "higgs":{"color":R.kPink-9},
#           "lowMassDY":{"color":R.kCyan-4},
#           "singleTop":{"color":R.kRed+4},
#           "topOther":{"color":R.kSpring-9},
#           "triboson":{"color":R.kViolet-7},
#           "ttbar":{"color":R.kYellow+2},
#           "data18":{"color":R.kBlack},
#           "WmuHNL50_30G":{"color":R.kGreen}
# }

def getTriggerThreshold(tname):
    thr = []
    #print(tname)
    reg = re.findall(r'_\d*([e]*[mu]*\d{1,})_{0,}',tname)
    for r in reg:
        #print(int(re.sub('\D', '', r)))
        thr.append(int(re.sub('\D', '', r)))
    return max(thr)



trgdic = {"2015":{"1L":["HLT_e24_lhmedium_L1EM20VH",
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
                  ]},
}

import re
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



def convertRDFCutflowToTex(cutflow1,cutflow2):
    i = 0
    tabstr = ""
    for c in cutflow1:
        cname = c.GetName()
        c2 = cutflow2.At(cname)
        if i == 0:
            nevc1 = c.GetAll()
            nevc2 = c2.GetAll()
        cname = cname.replace(">","$>$")
        cname = cname.replace("<","$<$")
        tabstr += "%-30s & $%.0f$ & $%.0f$ & $%.2f$ & $%.2f$ & $%.0f$ & $%.0f$ & $%.2f$ & $%.2f$ \\\ \n"%(cname,c.GetPass(),c.GetAll(),c.GetEff(),(c.GetPass()/nevc1)*100.,c2.GetPass(),c2.GetAll(),c2.GetEff(),(c2.GetPass()/nevc2)*100.)
        i += 1
    print(tabstr)


def writeHistsToFile(histo, writetofile = True):
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
        cl = R.gROOT.GetClass(key.GetClassName());
        if cl.InheritsFrom("TTree"):
            obj = key.ReadObj()
            if obj.GetName() in ["CutBookkeepers","MetaTree"]: 
                key = it.Next()
                continue
            return obj.GetName()
        else:
            key = it.Next()
            continue
    f1.Close()
    return "noname"


def getDataFrames(mypath, nev = 0): 
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    df = {}
    files = {}
    for of in onlyfiles:
        if not "merged" in of or not of.endswith(".root"): continue
        sp = of.split("_")
        typ = ""
        for s in sp:
            if "merged" in s or s.isnumeric(): break
            typ += s
        if not typ in files.keys():
            files[typ] = {"files":[], "treename":""}
        treename = getTreeName(mypath+"/"+of)
        if treename == "noname":
            print("ERROR \t Could not find any TTree in %s"%(mypath+"/"+of))
            continue
        files[typ]["treename"] = treename
        files[typ]["files"].append(mypath+"/"+of)
        
        #print(typ)
        #if not typ == "singleTop": continue
        #df[typ] = R.Experimental.MakeNTupleDataFrame("mini",mypath+"/"+of)#("%s_NoSys"%typ,mypath+"/"+of)
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

def runANA(mypath_mc, mypath_data, everyN, fldic, histo, allhisto, nEvents = 0, create_histogram = False):
    nh = 100
    if not isfile("histograms.root"):
        histo = getHistograms("histograms.root")
        return
    else:
        #mypath = "/storage/eirikgr/ANAntuples/PHYS_3LBkgs_mc16e/HNL3L_NOV03/merged/"
        if isdir(mypath_mc):
            df_mc = getDataFrames(mypath_mc)
            print("Loading %s into dataframe with keys %s" %(mypath_mc,",".join(df_mc.keys())))
        else:
            df_mc = {}

        #mypath = "/storage/eirikgr/ANAntuples/PHYS_Data/"
        if isdir(mypath_data):
            df_data = getDataFrames(mypath_data)
            print("Loading %s into dataframe with keys %s" %(mypath_data,",".join(df_data.keys())))
        else:
            df_data = {}
    
        
        
        df = {**df_mc}#{**df_mc,**df_data}
       
        
        print(df.keys())
       
        for k in df.keys():
            
           
            if k != "higgs":
                continue
                
            
                
            print(df[k].GetColumnNames())
            
            print("Number of events in %s = %i" %(k,df[k].Count().GetValue()))

            #if not k in ["data18"]: continue

            isData = "data" in k

            if not isData:
                df[k] = df[k].Define("scaletolumi","(RandomRunNumber) < 320000 ? 36207.65 : (((RandomRunNumber) > 320000 && (RandomRunNumber) < 348000) ? 44307.4 : 58450.1)")
            #else:
            #    run_cutstr = ""
            #    for rn in good_runs:
            #        run_cutstr += "(RunNumber == %s ||" %rn
            #    run_cutstr = run_cutstr[:-2]+")"
            #    print(run_cutstr)

            df[k] = df[k].Define("new_xsec","(DatasetNumber == 308981) ? (0.30649*69.594)/80000. : 0.0")

            # Baseline leptons
            df[k] = df[k].Define("ele_BL","lepFlavor==1 && lepPassOR > 0 && (lepEta <= 2.47 && lepEta >= -2.47) && ((lepZ0SinTheta)<=0.5 && (lepZ0SinTheta)>=-0.5)") #((lepZ0SinTheta)<=0.5 && (lepZ0SinTheta)>=-0.5) &&
            df[k] = df[k].Define("muo_BL","lepFlavor==2 && lepPassOR > 0 && (lepEta <= 2.7  && lepEta >= -2.7) && lepLoose > 0 && ((lepZ0SinTheta)<=0.5 && (lepZ0SinTheta)>=-0.5)") #((lepZ0SinTheta)<=0.5 && (lepZ0SinTheta)>=-0.5) &&

            df[k] = df[k].Define("nlep_BL","ROOT::VecOps::Sum(ele_BL)+ROOT::VecOps::Sum(muo_BL)")

            
            
            # Signal leptons
            df[k] = df[k].Define("ele_SG","ele_BL && lepIsoLoose_VarRad && lepTight && (lepD0Sig <= 5 && lepD0Sig >= -5)") #&& lepTight && (lepD0Sig <= 5 && lepD0Sig >= -5)
            df[k] = df[k].Define("muo_SG","muo_BL && lepIsoLoose_VarRad && (lepD0Sig <= 3 && lepD0Sig >= -3)") #&& (lepD0Sig <= 3 && lepD0Sig >= -3)

            df[k] = df[k].Define("nlep_SG","ROOT::VecOps::Sum(ele_SG)+ROOT::VecOps::Sum(muo_SG)")

            
            
            if not isData:

                df[k] = df[k].Define("is2015","RandomRunNumber <= 284500")
                df[k] = df[k].Define("is2016","(RandomRunNumber > 284500 && RandomRunNumber < 320000)")
                df[k] = df[k].Define("is2017","(RandomRunNumber > 320000 && RandomRunNumber < 348000)")
                df[k] = df[k].Define("is2018","RandomRunNumber > 348000")

                #df[k] = df[k].Define("lepwgt_BL","getSF(lepBLRecoSF[ele_BL || muo_BL])")
                df[k] = df[k].Define("lepwgt_SG","getSF(lepRecoSF[ele_SG || muo_SG])")

                #df[k] = df[k].Define("trgwgt_BL","getSF(lepBLTrigSF[ele_BL || muo_BL])")
                df[k] = df[k].Define("trgwgt_SG","getSF(lepTrigSF[ele_SG || muo_SG])")

                #df[k] = df[k].Define("wgt_BL","(new_xsec ? (new_xsec) : (genWeight))*eventWeight*jvtWeight*bTagWeight*pileupWeight*scaletolumi*lepwgt_BL*trgwgt_BL")
                df[k] = df[k].Define("wgt_SG","(new_xsec ? (new_xsec) : (genWeight))*eventWeight*jvtWeight*bTagWeight*pileupWeight*scaletolumi*lepwgt_SG*trgwgt_SG")

                #df[k] = df[k].Define("wgt_EV_BL","(eventWeight*jvtWeight*bTagWeight*pileupWeight*scaletolumi*lepwgt_BL*trgwgt_BL)")
                df[k] = df[k].Define("wgt_EV_SG","(eventWeight*jvtWeight*bTagWeight*pileupWeight*scaletolumi*lepwgt_SG*trgwgt_SG)")

            else:
                df[k] = df[k].Define("is2015","(RunNumber >= 276262 && RunNumber <= 284484)")
                df[k] = df[k].Define("is2016","(RunNumber >= 297730 && RunNumber <= 311481)")
                df[k] = df[k].Define("is2017","(RunNumber >= 325713 && RunNumber <= 340453)")
                df[k] = df[k].Define("is2018","RunNumber >= 348885")

                #df[k] = df[k].Define("wgt_BL","1.0")
                df[k] = df[k].Define("wgt_SG","1.0")
                df[k] = df[k].Define("wgt_EV","1.0")


            
            #df[k].Define("lepIsTrigMatched_2L","is2015 ? trigmatch_2015_2L : (is2016 ? trigmatch_2016_2L : (is2017 ? trigmatch_2017_2L : trigmatch_2018_2L))")
            #df[k].Define("lepIsTrigMatched_3L","is2015 ? trigmatch_2015_3L : (is2016 ? trigmatch_2016_3L : (is2017 ? trigmatch_2017_3L : trigmatch_2018_3L))")

            #print("Nev(pileupWeight == 0) : ",df[k].Filter("pileupWeight == 0").Count().GetValue())
            # Check trigger matching!
            for tr in trigstr.keys():
                if tr == "3L": continue
                for yr in trigstr[tr].keys():
                    #print("trigmatch_%s_%s"%(yr,tr))
                    df[k] = df[k].Define("trigmatch_%s_%s"%(yr,tr),trigstr[tr][yr])
                    df[k] = df[k].Define("triggered_%s_%s"%(yr,tr),evtrigstr[tr][yr])

            for nlep in ["1L","2L"]: #"1L"
                print(nlep)
                df[k] = df[k].Define("lepIsTrigMatched_%s"%nlep,"is2015 ? trigmatch_2015_%s : (is2016 ? trigmatch_2016_%s : (is2017 ? trigmatch_2017_%s : trigmatch_2018_%s))"%(nlep,nlep,nlep,nlep))
                df[k] = df[k].Define("eventIsTriggered_%s"%nlep,"is2015 ? triggered_2015_%s : (is2016 ? triggered_2016_%s : (is2017 ? triggered_2017_%s : triggered_2018_%s))"%(nlep,nlep,nlep,nlep))

            #df[k] = df[k].Filter("eventIsTriggered_1L","1L trigger")
            #df[k] = df[k].Filter("ROOT::VecOps::Sum(lepIsTrigMatched_1L[ele_BL || muo_BL]) > 0","Trigger Matched")

            if not nEvents:
                this_nEvents = int(df[k].Count().GetValue())
                nEvents += this_nEvents
                print("Loading %s with %.0f events. Now %.0f events"%(k,this_nEvents,nEvents))
            else:
                print("Loading %s"%(k))    

            #histo["nlep_BL_%s"%k] = df[k].Histo1D(("nlep_BL_%s"%k,"nlep_BL_%s"%k,10,0,10),"nlep_BL","wgt_SG")
            if create_histogram == True:
                #histo["nlep_SG_%s"%k] = df[k].Histo1D(("nlep_SG_%s"%k,"nlep_SG_%s"%k,10,0,10),"nlep_SG","wgt_SG")
                   pass
            
            df[k] = df[k].Filter("nlep_BL == 3","3 BL leptons")
            df[k] = df[k].Filter("nlep_SG == 3","3 SG leptons")

            df[k] = df[k].Define("isZlep1","getZlep1()")
            df[k] = df[k].Define("isZlep2","getZlep2()")
            df[k] = df[k].Define("isWlep1","getWlep1()")
            
            # Jets
            df[k] = df[k].Define("jet_BL","jetPt >= 20 && (jetEta <= 2.8 && jetEta >= -2.8)")
            df[k] = df[k].Define("jet_SG","jet_BL && (jetPt > 60 || (jetPt <=60 && jetJVT <= 0.91 && jetJVT >= -0.91))")

            
            
            
            
            """
            RMM matrix feature calculations with histogram creation
            
            """
            
            for row in range(N_row):
                if row == 0: # Calculate e_T^miss and m_T for all objects
                    
                    for column in range(N_col):
                        if column == 0: 
                            # Set e_T_miss
                            df[k] = df[k].Define("e_T_miss", "met_Et")
                            histo[f"e_T_miss_%s"%(k)] = df[k].Histo1D(("h_%s_%s"%(f"e_T_miss",k),"h_%s_%s;m_{T}^{2}(23) [GeV];Entries"%(f"e_T_miss",k),200,0,1000),f"e_T_miss","wgt_SG")
                        else:
                            # Set m_T for all particles
                            particle_info = rmm_structure[column]
                            name = particle_info[0]
                            pt = particle_info[1]
                            eta = particle_info[2]
                            phi = particle_info[3]
                            m = particle_info[4]
                            index = particle_info[5]
                            df[k] = df[k].Define(f"m_T_{name}", f"getM_T({pt},{eta},{phi},{m},{index})")
                            histo[f"m_T_{name}_%s"%(k)] = df[k].Histo1D(("h_%s_%s"%(f"m_T_{name}",k),"h_%s_%s;m_{T}^{2}(23) [GeV];Entries"%(f"e_T_miss",k),200,0,1000),f"m_T_{name}","wgt_SG")
                else:
                    for column in range(N_col):
                        if column == 0:
                            # Set h_L for all particles
                            particle_info = rmm_structure[row]
                            name = particle_info[0]
                            pt = particle_info[1]
                            eta = particle_info[2]
                            phi = particle_info[3]
                            m = particle_info[4]
                            index = particle_info[5]
                            df[k] = df[k].Define(f"h_L_{name}", f"geth_L({pt},{eta},{phi},{m},{index})")
                            histo[f"h_L_{name}_%s"%(k)] = df[k].Histo1D(("h_%s_%s"%(f"h_L_{name}",k),"h_%s_%s;m_{T}^{2}(23) [GeV];Entries"%(f"h_L_{name}",k),50,0,1),f"h_L_{name}","wgt_SG")
                           
                            
                        elif column == row:
                            particle_info = rmm_structure[column]
                            name = particle_info[0]
                            pt = particle_info[1]
                            eta = particle_info[2]
                            phi = particle_info[3]
                            m = particle_info[4]
                            index = particle_info[5]
                            if index == 0:
                                # If particle is the first of its type, calculate e_T of particle
                                df[k] = df[k].Define(f"e_T_{name}", f"getET_part({pt},{m},{index})")
                                histo[f"e_T_{name}_%s"%(k)] = df[k].Histo1D(("h_%s_%s"%(f"e_T_{name}",k),"h_%s_%s;m_{T}^{2}(23) [GeV];Entries"%(f"e_T_{name}",k),200,0,1000),f"e_T_{name}","wgt_SG")
                            
                            else:
                                # If particle is not the first of its type, calculate the difference in e_T
                                df[k] = df[k].Define(f"delta_e_t_{name}", f"delta_e_T({pt},{m},{index})")
                                histo[f"delta_e_t_{name}_%s"%(k)] = df[k].Histo1D(("h_%s_%s"%(f"delta_e_t_{name}",k),"h_%s_%s;m_{T}^{2}(23) [GeV];Entries"%(f"delta_e_t_{name}",k),200,0,1),f"delta_e_t_{name}","wgt_SG")
                                
                                
                            
                        elif column > row:
                            # For invariant mass
                            
                            # Particle 1
                            particle_info1 = rmm_structure[row]
                            name1 = particle_info1[0]
                            pt1 = particle_info1[1]
                            eta1 = particle_info1[2]
                            phi1 = particle_info1[3]
                            m1 = particle_info1[4]
                            index1 = particle_info1[5]
                            
                            # Particle 2
                            particle_info2 = rmm_structure[column]
                            name2 = particle_info2[0]
                            pt2 = particle_info2[1]
                            eta2 = particle_info2[2]
                            phi2 = particle_info2[3]
                            m2 = particle_info2[4]
                            index2 = particle_info2[5]
                            
                            histo_name = f"m_{name1}_{name2}"
                            
                            df[k] = df[k].Define(histo_name, f"getM({pt1},{eta1}, {phi1}, {m1}, {pt2}, {eta2}, {phi2}, {m}, {index1}, {index2})")
                            histo[f"{histo_name}_%s"%(k)] = df[k].Histo1D(("h_%s_%s"%(f"{histo_name}",k),"h_%s_%s;m_{T}^{2}(23) [GeV];Entries"%(f"{histo_name}",k),200,0,1000),f"{histo_name}","wgt_SG")
                                
                                
                            
                            
                        elif row > column:
                            # For h longitudal stuff
                            
                            # Particle 1
                            particle_info1 = rmm_structure[row]
                            name1 = particle_info1[0]
                            pt1 = particle_info1[1]
                            eta1 = particle_info1[2]
                            phi1 = particle_info1[3]
                            m1 = particle_info1[4]
                            index1 = particle_info1[5]
                            
                            # Particle 2
                            particle_info2 = rmm_structure[column]
                            name2 = particle_info2[0]
                            pt2 = particle_info2[1]
                            eta2 = particle_info2[2]
                            phi2 = particle_info2[3]
                            m2 = particle_info2[4]
                            index2 = particle_info2[5]
                            
                            histo_name = f"h_{name1}_{name2}"
                            
                            df[k] = df[k].Define(f"{histo_name}", f"geth({pt1},{eta1}, {phi1}, {m1}, {pt2}, {eta2}, {phi2}, {m}, {index1}, {index2})")
                            histo[f"{histo_name}_%s"%(k)] = df[k].Histo1D(("h_%s_%s"%(f"{histo_name}",k),"h_%s_%s;m_{T}^{2}(23) [GeV];Entries"%(f"{histo_name}",k),200,0,1000),f"{histo_name}","wgt_SG")
                                
            
           
            

            
            
            
                



        

    for k in histo.keys():
        allhisto.append(histo[k])


    print("Calculating %i histograms"%len(allhisto))
    #sys.exit()
    start = time.time()
    R.RDF.RunGraphs(allhisto)
    end = time.time()
    print("%10i | %.2f"%(len(allhisto),(end - start)))

    hfile = R.TFile("histograms.root","RECREATE")
    hfile.cd()

    writeHistsToFile(histo, True)


    
    return df, histo

if __name__ == "__main__":
    
    N_j = 0
    N_l = 3
    
    N_col = N_j + N_l + 1
    N_row = N_col
    """
        1: [  "jet_0", 
             "jetPt[jet_SG > 0]", 
             "jetEta[jet_SG > 0]", 
             "jetPhi[jet_SG > 0]", 
             "jetM[jet_SG > 0]", 
             0],

         2: ["jet_1", 
             "jetPt[jet_SG > 0]", 
             "jetEta[jet_SG > 0]", 
             "jetPhi[jet_SG > 0]", 
             "jetM[jet_SG > 0]", 
             1],
    """
    
    rmm_structure = {
        

         1: ["lep_0", 
             "lepPt[ele_SG > 0 || muo_SG > 0]", 
             "lepEta[ele_SG > 0 || muo_SG > 0]", 
             "lepPhi[ele_SG > 0 || muo_SG > 0]", 
             "lepM[ele_SG > 0 || muo_SG > 0]", 
             0],

         2: ["lep_1", 
             "lepPt[ele_SG > 0 || muo_SG > 0]", 
             "lepEta[ele_SG > 0 || muo_SG > 0]", 
             "lepPhi[ele_SG > 0 || muo_SG > 0]", 
             "lepM[ele_SG > 0 || muo_SG > 0]", 
             1],

         3: ["lep_2", 
             "lepPt[ele_SG > 0 || muo_SG > 0]", 
             "lepEta[ele_SG > 0 || muo_SG > 0]", 
             "lepPhi[ele_SG > 0 || muo_SG > 0]", 
             "lepM[ele_SG > 0 || muo_SG > 0]", 
             2]
                    
                    
    }
    
    
    
    good_runs = []

    
    histo = {}
    allhisto = []
    nEvents = 618282964
    nSlots = R.GetThreadPoolSize();
    print("Number of slots = %i"%nSlots);
    everyN = int(100 * nSlots)

    df, histo = runANA("/storage/shared/data/master_students/William_Sakarias/data/PHYS_3LBkgs_mc16e/","/storage/shared/data/master_students/William_Sakarias/data/data18/",everyN,fldic,histo,allhisto, create_histogram=True)
    
    
    #toplot = ["Diboson","WjetsPH","ZjetsPH","DibosonPH","PythiaB","Triboson","Wjets","XGamma","Zjets","lowMassDY","ttbar"]
    writeHistsToFile(histo, False)

    toplot = []
    for bkg in bkgdic.keys():
        toplot.append(bkg)
        
        
    names = histo.keys()

    print(len(names))
    new_feats = []
    for name in names:
        if name[-5:] == "higgs":
            new_feats.append(name[:-6])

    print(new_feats)
    
    for key in new_feats:
        try:
            p = pt.Plot(histo,key,toplot)
            p.can.SaveAs(f"../../../Figures/histo_var_check/{key}.pdf")
        except:
            print(f"Could not make plot for name {key}")


    
    
    

