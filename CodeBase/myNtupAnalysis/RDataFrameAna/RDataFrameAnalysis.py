import re
import sys
import time
import glob
import array
import requests
import ROOT as R
import matplotlib
import numpy as np
import pandas as pd
from os import listdir
from usedPaths import *
from pathlib import Path
from typing import Tuple
import plottingTool as pt
from pyHelperFuncs import *
import matplotlib.pyplot as plt
from samples import configure_samples
from os.path import isfile, join, isdir
from div_dicts import triggers, rmm_structure



d_samp, d_type, d_reg = configure_samples()  # False,False,True,False,False)

R.EnableImplicitMT(1000)

R.gROOT.ProcessLine(".L helperFunctions.cxx+")
R.gSystem.AddDynamicPath(str(DYNAMIC_PATH))
R.gInterpreter.Declare(
    '#include "helperFunctions.h"'
)  # Header with the definition of the myFilter function
R.gSystem.Load("helperFunctions_cxx.so")  # Library with the myFilter function

# from IPython.display import display, HTML

# display(HTML("<style>.container { width:100% !important; }</style>"))


def runANA(
    mypath_mc: str,
    mypath_data: str,
    everyN: int,
    fldic: dict,
    histo: dict,
    allhisto: list,
    nEvents=0,
    create_histogram=False,
) -> Tuple[dict, dict]:

    nh = 100

    # mypath = "/storage/eirikgr/ANAntuples/PHYS_3LBkgs_mc16e/HNL3L_NOV03/merged/"

    if isdir(mypath_mc):
        df_mc = getDataFrames1(mypath_mc)
        print(
            "Loading %s into dataframe with keys %s"
            % (mypath_mc, ",".join(df_mc.keys()))
        )
    else:
        df_mc = {}
    
    #exit()
    # mypath = "/storage/eirikgr/ANAntuples/PHYS_Data/"
    if isdir(mypath_data):
        df_data = getDataFrames1(mypath_data)
        print(
            "Loading %s into dataframe with keys %s"
            % (mypath_data, ",".join(df_data.keys()))
        )
    else:
        df_data = {}

    df = {**df_mc, **df_data}
    
    print(" ")
   
    
    for k in df.keys():

        if k not in samples: #["topOther"]:#
            continue
        
        
        
       
        
        # df[k] = df[k].Range(0,100)
            
        # print("Number of events in %s = %i" % (k, df[k].Count().GetValue()))

        # if not k in ["data18"]: continue

        isData = "data" in k

        if not isData:
            """df[k] = df[k].Define(
                "scaletolumi",
                "(RandomRunNumber) < 320000 ? 36207.65 : (((RandomRunNumber) > 320000 && (RandomRunNumber) < 348000) ? 44307.4 : 58450.1)",
            )"""
            df[k] = df[k].Define(
                "scaletolumi",
                "(RandomRunNumber) < 320000 ? 36207.65 : (((RandomRunNumber) > 320000 && (RandomRunNumber) < 348000) ? 44307.4 : (((RandomRunNumber) > 348000 && (RandomRunNumber) < 400000) ? 58450.1 : 1258.27))",
            )

        df[k] = df[k].Define(
            "new_xsec", "(DatasetNumber == 308981) ? (0.30649*69.594)/80000. : 0.0"
        )

        # Baseline leptons
        df[k] = df[k].Define(
            "ele_BL",
            "lepFlavor==1 && lepPassOR > 0 && (lepEta <= 2.47 && lepEta >= -2.47) && ((lepZ0SinTheta)<=0.5 && (lepZ0SinTheta)>=-0.5)",
        )  # ((lepZ0SinTheta)<=0.5 && (lepZ0SinTheta)>=-0.5) &&
        df[k] = df[k].Define(
            "muo_BL",
            "lepFlavor==2 && lepPassOR > 0 && (lepEta <= 2.7  && lepEta >= -2.7) && lepLoose > 0 && ((lepZ0SinTheta)<=0.5 && (lepZ0SinTheta)>=-0.5)",
        )  # ((lepZ0SinTheta)<=0.5 && (lepZ0SinTheta)>=-0.5) &&

        df[k] = df[k].Define(
            "nlep_BL", "ROOT::VecOps::Sum(ele_BL)+ROOT::VecOps::Sum(muo_BL)"
        )

        # Signal leptons
        df[k] = df[k].Define(
            "ele_SG",
            "ele_BL && lepIsoLoose_VarRad && lepTight && (lepD0Sig <= 5 && lepD0Sig >= -5)",
        )  # && lepTight && (lepD0Sig <= 5 && lepD0Sig >= -5)
        df[k] = df[k].Define(
            "muo_SG",
            "muo_BL && lepIsoLoose_VarRad && (lepD0Sig <= 3 && lepD0Sig >= -3)",
        )  # && (lepD0Sig <= 3 && lepD0Sig >= -3)

        df[k] = df[k].Define(
            "nlep_SG", "ROOT::VecOps::Sum(ele_SG)+ROOT::VecOps::Sum(muo_SG)"
        )

        df[k] = df[k].Define("isGoodLep", "ele_SG || muo_SG")

        if not isData:

            
            
            df[k] = df[k].Define("is2015", "RandomRunNumber <= 284500")
            df[k] = df[k].Define(
                "is2016", "(RandomRunNumber > 284500 && RandomRunNumber < 320000)"
            )
            df[k] = df[k].Define(
                "is2017", "(RandomRunNumber > 320000 && RandomRunNumber < 348000)"
            )
            df[k] = df[k].Define(
                "is2018", "(RandomRunNumber > 348000 && RandomRunNumber < 400000)"
            )
            df[k] = df[k].Define("is2022", "(RandomRunNumber <= 427882)")

            # df[k] = df[k].Define("lepwgt_BL","getSF(lepBLRecoSF[ele_BL || muo_BL])")
            df[k] = df[k].Define("lepwgt_SG", "getSF(lepRecoSF[ele_SG || muo_SG])")

            # df[k] = df[k].Define("trgwgt_BL","getSF(lepBLTrigSF[ele_BL || muo_BL])")
            df[k] = df[k].Define("trgwgt_SG", "getSF(lepTrigSF[ele_SG || muo_SG])")

            # df[k] = df[k].Define("wgt_BL","(new_xsec ? (new_xsec) : (genWeight))*eventWeight*jvtWeight*bTagWeight*pileupWeight*scaletolumi*lepwgt_BL*trgwgt_BL")
            # df[k] = df[k].Define("wgt_SG","(new_xsec ? (new_xsec) : (genWeight))*eventWeight*jvtWeight*bTagWeight*scaletolumi*lepwgt_SG*trgwgt_SG") #*pileupWeight
            # df[k] = df[k].Define("wgt_SG","(genWeight)*eventWeight*jvtWeight*bTagWeight*scaletolumi*lepwgt_SG*trgwgt_SG*beamSpotWeight") #*pileupWeight
            if "data22" in k or "mc21a" in mypath_mc:
                df[k] = df[k].Define(
                    "wgt_SG",
                    "(genWeight)*eventWeight*jvtWeight*bTagWeight*scaletolumi*leptonWeight*globalDiLepTrigSF*beamSpotWeight",
                )  # *pileupWeight

                # df[k] = df[k].Define("wgt_EV_BL","(eventWeight*jvtWeight*bTagWeight*pileupWeight*scaletolumi*lepwgt_BL*trgwgt_BL)")
                # df[k] = df[k].Define("wgt_EV_SG","(eventWeight*jvtWeight*bTagWeight*scaletolumi*lepwgt_SG*trgwgt_SG*beamSpotWeight)") #*pileupWeight
                df[k] = df[k].Define(
                    "wgt_EV_SG",
                    "(eventWeight*jvtWeight*bTagWeight*scaletolumi*leptonWeight*globalDiLepTrigSF*beamSpotWeight)",
                )  # *pileupWeight
            elif "Z" in k and "jets" in k:
                df[k] = df[k].Define(
                    "wgt_SG",
                    "((genWeight)*eventWeight*jvtWeight*bTagWeight*scaletolumi*leptonWeight*globalDiLepTrigSF*pileupWeight)*1000.",
                )  # *pileupWeight

                # df[k] = df[k].Define("wgt_EV_BL","(eventWeight*jvtWeight*bTagWeight*pileupWeight*scaletolumi*lepwgt_BL*trgwgt_BL)")
                # df[k] = df[k].Define("wgt_EV_SG","(eventWeight*jvtWeight*bTagWeight*scaletolumi*lepwgt_SG*trgwgt_SG*beamSpotWeight)") #*pileupWeight
                df[k] = df[k].Define(
                    "wgt_EV_SG",
                    "(eventWeight*jvtWeight*bTagWeight*scaletolumi*leptonWeight*globalDiLepTrigSF*pileupWeight)*1000.",
                )  # *pileupWeight
            else:
                df[k] = df[k].Define(
                    "wgt_SG",
                    "(genWeight)*eventWeight*jvtWeight*bTagWeight*scaletolumi*leptonWeight*globalDiLepTrigSF*pileupWeight",
                )  # *pileupWeight

                # df[k] = df[k].Define("wgt_EV_BL","(eventWeight*jvtWeight*bTagWeight*pileupWeight*scaletolumi*lepwgt_BL*trgwgt_BL)")
                # df[k] = df[k].Define("wgt_EV_SG","(eventWeight*jvtWeight*bTagWeight*scaletolumi*lepwgt_SG*trgwgt_SG*beamSpotWeight)") #*pileupWeight
                df[k] = df[k].Define(
                    "wgt_EV_SG",
                    "(eventWeight*jvtWeight*bTagWeight*scaletolumi*leptonWeight*globalDiLepTrigSF*pileupWeight)",
                )  # *pileupWeight
        else:
            df[k] = df[k].Define(
                "is2015", "(RunNumber >= 276262 && RunNumber <= 284484)"
            )
            df[k] = df[k].Define(
                "is2016", "(RunNumber >= 297730 && RunNumber <= 311481)"
            )
            df[k] = df[k].Define(
                "is2017", "(RunNumber >= 325713 && RunNumber <= 340453)"
            )
            df[k] = df[k].Define(
                "is2018", "(RunNumber >= 348885 && RunNumber <  370000)"
            )
            df[k] = df[k].Define("is2022", "(RunNumber >= 427882)")

            # df[k] = df[k].Define("wgt_BL","1.0")
            df[k] = df[k].Define("wgt_SG", "1.0")
            df[k] = df[k].Define("wgt_EV", "1.0")

        # df[k].Define("lepIsTrigMatched_2L","is2015 ? trigmatch_2015_2L : (is2016 ? trigmatch_2016_2L : (is2017 ? trigmatch_2017_2L : trigmatch_2018_2L))")
        # df[k].Define("lepIsTrigMatched_3L","is2015 ? trigmatch_2015_3L : (is2016 ? trigmatch_2016_3L : (is2017 ? trigmatch_2017_3L : trigmatch_2018_3L))")

        # print("Nev(pileupWeight == 0) : ",df[k].Filter("pileupWeight == 0").Count().GetValue())
        # Check trigger matching!
       
        histo["nlep_BL_%s" % k] = df[k].Histo1D(
            ("nlep_BL_%s" % k, "nlep_BL_%s" % k, 10, 0, 10), "nlep_BL", "wgt_SG"
        )
        histo["nlep_SG_%s" % k] = df[k].Histo1D(
            ("nlep_SG_%s" % k, "nlep_SG_%s" % k, 10, 0, 10), "nlep_SG", "wgt_SG"
        )

        # df[k] = df[k].Filter("eventIsTriggered_1L","1L trigger")
        # df[k] = df[k].Filter("ROOT::VecOps::Sum(lepIsTrigMatched_1L[ele_BL || muo_BL]) > 0","Trigger Matched")

        if not nEvents:
            this_nEvents = int(df[k].Count().GetValue())
            nEvents += this_nEvents
            print(
                "Loading %s with %.0f events. Now %.0f events"
                % (k, this_nEvents, nEvents)
            )
        else:
            print("Loading %s" % (k))

        df[k] = df[k].Filter("nlep_BL == 2", "2 BL leptons")
        df[k] = df[k].Filter("nlep_SG == 2", "2 SG leptons")

        print("Number of events in %s = %i" % (k, df[k].Count().GetValue()))

        """
        Trigger filtering
        """
        """
        # 2015 only triggers

        trigs2015 = triggers["2015"]
        trig2015 = trigs2015["trig"]
        trigmatch2015 = trigs2015["trigmatch"]

        df[k] = df[k].Filter(trig2015, "2015trig")
        df[k] = df[k].Filter(trigmatch2015, "2015trigmatch")
        
        

        # 2016 only triggers

        trigs2016 = triggers["2015"]
        trig2016 = trigs2016["trig"]
        trigmatch2016 = trigs2016["trigmatch"]

        df[k] = df[k].Filter(trig2016, "2016trig")
        df[k] = df[k].Filter(trigmatch2016, "2016trigmatch")
        
        

        # 2017 and 2018 only triggers

        trigs201718 = triggers["2017/18"]
        trig201718 = trigs201718["trig"]
        trigmatch201718 = trigs201718["trigmatch"]

        df[k] = df[k].Filter(trig201718, "2018trig")
        
        df[k] = df[k].Filter(trigmatch201718, "2018trigmatch")
        
        print("Filtering done.")
        """

        """
        pT cut for two highest pT leptons above 20 GeV
        """
        df[k] = df[k].Filter(
            "ROOT::VecOps::Sum(lepPt[isGoodLep] > 20) >= 2", "pt cut 20"
        )

        """
        Remove Z overlap
        """
        if k in ["Zeejets", "Zmmjets", "Zttjets"]:
            df[k] = df[k].Filter(
                "((DatasetNumber >= 700320 && DatasetNumber <= 700328) && bornMass <= 120000) || !((DatasetNumber >= 700320 && DatasetNumber <= 700328))",
                "Z overlap",
            )

        # Jets
        df[k] = df[k].Define(
            "jet_BL", "jetPt >= 20 && (jetEta <= 2.8 && jetEta >= -2.8)"
        )
        df[k] = df[k].Define(
            "jet_SG",
            "jet_BL && (jetPt > 60 || (jetPt <=60 && jetJVT <= 0.91 && jetJVT >= -0.91))",
        )

        df[k] = df[k].Define("njet_BL", "ROOT::VecOps::Sum(jet_BL)")
        df[k] = df[k].Define("njet_SG", "ROOT::VecOps::Sum(jet_SG)")
        
        df[k] = df[k].Define("ljet","jet_SG && jetdl1r<0.665")
        
        
        df[k] = df[k].Define("bjet85","jet_SG && jetdl1r>=0.665")
        df[k] = df[k].Define("bjet77","jet_SG && jetdl1r>=2.195")

        df[k] = df[k].Define("nbjet85","ROOT::VecOps::Sum(bjet85)")
        df[k] = df[k].Define("nbjet77","ROOT::VecOps::Sum(bjet77)")
        
        
        

        
        """    
        print(df[k].Display("bjet85").AsString())
        print(df[k].Display("bjet77").AsString())
        """
       

        # Adding column for type of channel

        # df[k] = df[k].Define("channeltype", k)

        # print(df[k].Display("lepPt").AsString())

        """
        RMM matrix feature calculations with histogram creation
        
        """

        # df_test = df[k].Range(43, 45)

        for row in range(N_row):
            if row == 0:

                # Calculate e_T^miss and m_T for all objects

                for column in range(N_col):
                    if column == 0:
                        # Set e_T_miss
                        df[k] = df[k].Define("e_T_miss", "met_Et")
                        histo[f"e_T_miss_{k}"] = df[k].Histo1D(
                            (
                                f"h_e_T_miss_{k}",
                                f"h_e_T_miss_{k};m_" + "{T}^{2}(23) [GeV];Entries",
                                70,
                                0,
                                500,
                            ),
                            f"e_T_miss",
                            "wgt_SG",
                        )
                    else:
                        # Set m_T for all particles
                        particle_info = rmm_structure[column]
                        name = particle_info[0]
                        pt = particle_info[1]
                        eta = particle_info[2]
                        phi = particle_info[3]
                        m = particle_info[4]
                        index = particle_info[5]

                        df[k] = df[k].Define(
                            f"m_T_{name}", f"getM_T({pt},{eta},{phi},{m},{index})"
                        )
                        histo[f"m_T_{name}_{k}"] = df[k].Histo1D(
                            (
                                f"h_m_T_{name}_{k}",
                                f"h_e_T_miss_{k};m_" + "{T}^{2}(23) [GeV];Entries",
                                70,
                                0,
                                500,
                            ),
                            f"m_T_{name}",
                            "wgt_SG",
                        )
            else:

                # Calculate rest of matrix

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

                        df[k] = df[k].Define(
                            f"h_L_{name}", f"geth_L({pt},{eta},{phi},{m},{index})"
                        )
                        histo[f"h_L_{name}_{k}"] = df[k].Histo1D(
                            (
                                f"h_h_L_{name}_{k}",
                                f"h_h_L_{name}_{k};m_" + "{T}^{2}(23) [GeV];Entries",
                                50,
                                0,
                                1,
                            ),
                            f"h_L_{name}",
                            "wgt_SG",
                        )

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

                            df[k] = df[k].Define(
                                f"e_T_{name}", f"getET_part({pt},{m},{index})"
                            )
                            histo[f"e_T_{name}_{k}"] = df[k].Histo1D(
                                (
                                    f"h_e_T_{name}_{k}",
                                    f"h_e_T_{name}_{k};m_"
                                    + "{T}^{2}(23) [GeV];Entries",
                                    70,
                                    0,
                                    500,
                                ),
                                f"e_T_{name}",
                                "wgt_SG",
                            )

                        else:
                            # If particle is not the first of its type, calculate the difference in e_T

                            df[k] = df[k].Define(
                                f"delta_e_t_{name}", f"delta_e_T({pt},{m},{index})"
                            )
                            histo[f"delta_e_t_{name}_{k}"] = df[k].Histo1D(
                                (
                                    f"h_delta_e_t_{name}_{k}",
                                    f"h_delta_e_t_{name}_{k};m_"
                                    + "{T}^{2}(23) [GeV];Entries",
                                    50,
                                    0,
                                    1,
                                ),
                                f"delta_e_t_{name}",
                                "wgt_SG",
                            )

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

                        df[k] = df[k].Define(
                            histo_name,
                            f"getM({pt1},{eta1}, {phi1}, {m1}, {pt2}, {eta2}, {phi2}, {m2}, {index1}, {index2})",
                        )
                        histo[f"{histo_name}_{k}"] = df[k].Histo1D(
                            (
                                f"h_{histo_name}_{k}",
                                f"h_{histo_name}_{k};m_" + "{T}^{2}(23) [GeV];Entries",
                                70,
                                0,
                                500,
                            ),
                            f"{histo_name}",
                            "wgt_SG",
                        )

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

                        df[k] = df[k].Define(
                            f"{histo_name}",
                            f"geth({pt1},{eta1}, {phi1}, {m1},  {pt2}, {eta2}, {phi2}, {m2},  {index1}, {index2})",
                        )
                        histo[f"{histo_name}_{k}"] = df[k].Histo1D(
                            (
                                f"h_{histo_name}_{k}",
                                f"h_{histo_name}_{k};;Entries",
                                50,
                                0,
                                1,
                            ),
                            f"{histo_name}",
                            "wgt_SG",
                        )

        df[k] = df[k].Define("flcomp", "flavourComp3L(lepFlavor[ele_SG || muo_SG])")
        histo[f"flcomp_{k}"] = df[k].Histo1D(
            (
                f"h_flcomp_{k}",
                f"h_flcomp_{k}",
                len(fldic.keys()),
                0,
                len(fldic.keys()),
            ),
            "flcomp",
            "wgt_SG",
        )

        df[k] = df[k].Define("ele_0_charge", "getLepCharge(lepCharge, lepFlavor, 0, 1)")
        histo_name = "ele_0_charge"
        histo[f"{histo_name}_{k}"] = (
            df[k]
            .Filter("ele_0_charge < 0 || ele_0_charge > 0")
            .Histo1D(
                (
                    f"h_{histo_name}_{k}",
                    f"h_{histo_name}_{k};;Entries",
                    3,
                    -2,
                    2,
                ),
                f"{histo_name}",
                "wgt_SG",
            )
        )

        df[k] = df[k].Define("ele_1_charge", "getLepCharge(lepCharge, lepFlavor, 1, 1)")
        histo_name = "ele_1_charge"
        histo[f"{histo_name}_%s" % (k)] = (
            df[k]
            .Filter("ele_1_charge < 0 || ele_1_charge > 0")
            .Histo1D(
                (
                    f"h_{histo_name}_{k}",
                    f"h_{histo_name}_{k};;Entries",
                    3,
                    -2,
                    2,
                ),
                f"{histo_name}",
                "wgt_SG",
            )
        )

        df[k] = df[k].Define("ele_2_charge", "getLepCharge(lepCharge, lepFlavor, 2, 1)")
        histo_name = "ele_2_charge"
        histo[f"{histo_name}_%s" % (k)] = (
            df[k]
            .Filter("ele_2_charge < 0 || ele_2_charge > 0")
            .Histo1D(
                (
                    f"h_{histo_name}_{k}",
                    f"h_{histo_name}_{k};;Entries",
                    3,
                    -2,
                    2,
                ),
                f"{histo_name}",
                "wgt_SG",
            )
        )

        df[k] = df[k].Define("muo_0_charge", "getLepCharge(lepCharge, lepFlavor, 0, 2)")
        histo_name = "muo_0_charge"
        histo[f"{histo_name}_%s" % (k)] = (
            df[k]
            .Filter("muo_0_charge < 0 || muo_0_charge > 0")
            .Histo1D(
                (
                    f"h_{histo_name}_{k}",
                    f"h_{histo_name}_{k};;Entries",
                    3,
                    -2,
                    2,
                ),
                f"{histo_name}",
                "wgt_SG",
            )
        )

        df[k] = df[k].Define("muo_1_charge", "getLepCharge(lepCharge, lepFlavor, 1, 2)")
        histo_name = "muo_1_charge"
        histo[f"{histo_name}_%s" % (k)] = (
            df[k]
            .Filter("muo_1_charge < 0 || muo_1_charge > 0")
            .Histo1D(
                (
                    f"h_{histo_name}_{k}",
                    f"h_{histo_name}_{k};;Entries",
                    3,
                    -2,
                    2,
                ),
                f"{histo_name}",
                "wgt_SG",
            )
        )

        df[k] = df[k].Define("muo_2_charge", "getLepCharge(lepCharge, lepFlavor, 2, 2)")
        histo_name = "muo_2_charge"
        histo[f"{histo_name}_%s" % (k)] = (
            df[k]
            .Filter("muo_2_charge < 0 || muo_2_charge > 0")
            .Histo1D(
                (
                    f"h_{histo_name}_{k}",
                    f"h_{histo_name}_{k};;Entries",
                    3,
                    -2,
                    2,
                ),
                f"{histo_name}",
                "wgt_SG",
            )
        )
        
        df[k] = df[k].Define("TrileptonMass", "TrileptonMass(lepPt[isGoodLep],lepEta[isGoodLep],lepPhi[isGoodLep],lepM[isGoodLep])")
        histo_name = "TrileptonMass"
        histo[f"{histo_name}_%s" % (k)] = (
            df[k]
            .Histo1D(
                (
                    f"h_{histo_name}_{k}",
                    f"h_{histo_name}_{k};;Entries",
                    70,
                    0,
                    500,
                ),
                f"{histo_name}",
                "wgt_SG",
            )
        )
        """
        k = df[k].Range(5, 30)  
        k.Filter("ele_0_charge < 0 || ele_0_charge > 0")
        p = k.Display("flcomp").AsString()#df[k].Display("lepFlavor").AsString()
        print(p)
        p = k.Display(("ele_0_charge", "ele_1_charge", "ele_2_charge", "muo_0_charge", "muo_1_charge", "muo_2_charge")).AsString()#df[k].Display("lepFlavor").AsString()
        print(p)
        exit()"""

        a = df[k].Report().Print()
        
        print(a)
       

    for k in histo.keys():
        allhisto.append(histo[k])

    print("Calculating %i histograms" % len(allhisto))
    # sys.exit()
    start = time.time()
    R.RDF.RunGraphs(allhisto)
    end = time.time()
    print("%10i | %.2f" % (len(allhisto), (end - start)))

    hfile = R.TFile("histograms.root", "RECREATE")
    hfile.cd()

    writeHistsToFile(histo, d_samp, True)

    return df, histo


def create_histograms_pdfs(
    histo: dict, all_cols: list, histo_var: Path, d_samp: dict
) -> None:
    """
    Takes the list of features in the histo dict, and creates
    histograms of those features.

    Args:
        histo (dict): dictionary containing all the histograms
        new_feats (list): list of features to make histograms out of
    """

    writeHistsToFile(histo, d_samp, False)

    toplot = []
    for bkg in bkgdic.keys():
        toplot.append(bkg)

    for hname in all_cols:

        try:

            p = pt.Plot(histo, hname, toplot)

            p.can.SaveAs(str(histo_var) + f"/{hname}.pdf")

        except:
            print(f"Could not make plot for name {hname}")


def get_numpy_df(df: dict, all_cols: list) -> list:
    """_summary_

    Args:
        df (dict): _description_
        all_cols (list): _description_

    Returns:
        list: _description_
    """

    cols = df.keys()

    dfs = []
    for k in cols:
        
        if k not in samples: #["topOther"]:#
            continue

        print(f"Transforming {k}.ROOT to numpy")
        numpy = df[k].AsNumpy(all_cols)
        print(f"Numpy conversion done for {k}.ROOT")
        df1 = pd.DataFrame(data=numpy)
        print(f"Transformation done")
        dfs.append(df1)

        # print("        ")
        # dfs.append(df1)
        # del df1
        df1.to_hdf(
            str(df_storage) + f"/two_{k}_3lep_df_forML_bkg_signal_fromRDF.hdf5", "mini"
        )

    return dfs


def fetchDfs():
    
    
    files = [
        f for f in listdir(str(df_storage)) 
        if isfile(join(str(df_storage), f)) and 
        f[-4:] != ".npy"
        and f[-4:] != ".csv"
        and f[-5:] != "_b.h5"
        and f[-4:] != ".txt"
        and f[-3:] != ".h5"
    ]
    
    
    
    
    keep = [
        "Wjets_3lep_df_forML_bkg_signal_fromRDF.hdf5",
        "diboson2L_3lep_df_forML_bkg_signal_fromRDF.hdf5",
        "diboson3L_3lep_df_forML_bkg_signal_fromRDF.hdf5",
        "diboson4L_3lep_df_forML_bkg_signal_fromRDF.hdf5",
        "higgs_3lep_df_forML_bkg_signal_fromRDF.hdf5",
        "singletop_3lep_df_forML_bkg_signal_fromRDF.hdf5",
        "topOther_3lep_df_forML_bkg_signal_fromRDF.hdf5",
        "triboson_3lep_df_forML_bkg_signal_fromRDF.hdf5",
        "ttbar_3lep_df_forML_bkg_signal_fromRDF.hdf5",
        "data18_3lep_df_forML_bkg_signal_fromRDF.hdf5",
        "Zeejets_3lep_df_forML_bkg_signal_fromRDF.hdf5",
        "Zmmjets_3lep_df_forML_bkg_signal_fromRDF.hdf5",
        "Zttjets_3lep_df_forML_bkg_signal_fromRDF.hdf5",
        "data15_3lep_df_forML_bkg_signal_fromRDF.hdf5",
        "data16_3lep_df_forML_bkg_signal_fromRDF.hdf5",
        "data17_3lep_df_forML_bkg_signal_fromRDF.hdf5",]
    
    
    
    
    keeps = []  
    names = []
        
    for file in files:
        if file in keep:
            c = file.find("_3lep_df_forML_bkg_signal_fromRDF.hdf5")
            name = file[:c]
            names.append(name)
            print(name)
            df = pd.read_hdf(str(df_storage) + "/" + file)
            
            df = df.drop([
                'nlep_BL', 'nlep_SG','flcomp', 'ele_0_charge',
                'ele_1_charge', 'ele_2_charge', 'muo_0_charge', 'muo_1_charge',
                'muo_2_charge', 'wgt_SG'], axis=1)
            print(df.columns)
            keeps.append(df)
        
       
    
    print(names)
     

    return keeps, names





if __name__ == "__main__":
    
    samples = ["data15","data16","data17","data18", "Wjets","Zmmjets","Zttjets","diboson2L","diboson3L","diboson4L","higgs","topOther","triboson","ttbar","Zeejets","singletop", "MGPy8EGA14N23LOC1N2WZ800p0p050p0p03L2L7", "MGPy8EGA14N23LOC1N2WZ450p0p0300p0p03L2L7"]

    rerun = 1
    if len(sys.argv) > 2:
        rerun = int(sys.argv[2])
    if rerun:
        files = glob.glob("*")
        for f in files:
            if f in ["histograms.root"]:
                remove(f)

    """Remove old images from histo_var_check"""
    # de = [f.unlink() for f in Path(HISTO_VAR).glob("*") if f.is_file()]

    """ Actual analysis """
    N_j = 12
    N_l = 10

    N_col = N_j + N_l + 1
    N_row = N_col
    
    

    good_runs = []

    histo = {}
    allhisto = []
    nEvents = 618282964
    nSlots = R.GetThreadPoolSize()
    print("Number of slots = %i" % nSlots)
    everyN = int(100 * nSlots)

    tot_lumi = 0.0

    hfiles = glob.glob("./histograms_*.root")
    histo = {}
    """name_allhisto_1D = []
    name_allhisto_2D = []
    if not rerun:
        all_histo = []
        for f in hfiles:
            all_histo.append(getHistograms("%s"%f))
            for key in all_histo[-1].keys():
                this_hname = "_".join(key.split("_")[:-1])
                if not this_hname in name_allhisto_1D and type(all_histo[-1][key]) is R.TH1D:
                    name_allhisto_1D.append(this_hname)
                    #print(this_hname)
                elif not this_hname in name_allhisto_2D and type(all_histo[-1][key]) is R.TH2D:
                    name_allhisto_2D.append(this_hname)
                    #print(this_hname)
            for d in all_histo:
                histo.update(d)
    else:"""
    
    df, histo = runANA(
        str(MC_AND_DATA_PATH_2LEP),
        str(MC_AND_DATA_PATH_2LEP) + "/EXOT0_Data",
        everyN,
        fldic,
        histo,
        allhisto[:],
        create_histogram=True,
    )

    all_cols = get_column_names(df, histo)

    

    print("Histogram creation started")
    create_histograms_pdfs(histo, all_cols, histo_var=HISTO_VAR, d_samp=d_samp)
    print("Histogram creation done")
    
    print(" ")
    
    print(all_cols)
    
    print("Numpy conversion started")
    numpy_dfs = get_numpy_df(df, all_cols)
    print("Numpy conversion ended")
    names = list(df.keys())
    
    
    #numpy_dfs, names = fetchDfs()
    
    print(" ")
    
    print("RMM plot creation started")
    
    for index, df in enumerate(numpy_dfs):
        plot_rmm_matrix(df, names[index], rmm_structure, RMMSIZE, lep=2)
    print("RMM plot creation ended")
    
    
    
    TOKEN = "5789363537:AAF0SErRfZ07yWrzjppg9oCCO6H8BfFLHw"
    chat_id = "5733209220"
    message = "Hello Sakarias, Event selection is done!"
    resp = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}")
    print(resp.status_code)