import re
import sys
import time
import array
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

R.EnableImplicitMT(200)

R.gROOT.ProcessLine(".L helperFunctions.cxx+")
R.gSystem.AddDynamicPath(str(dynamic_path))
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
    if not isfile("histograms.root"):
        histo = getHistograms("histograms.root")
        return
    else:
        # mypath = "/storage/eirikgr/ANAntuples/PHYS_3LBkgs_mc16e/HNL3L_NOV03/merged/"
        if isdir(mypath_mc):
            df_mc = getDataFrames(mypath_mc)
            print(
                "Loading %s into dataframe with keys %s"
                % (mypath_mc, ",".join(df_mc.keys()))
            )
        else:
            df_mc = {}

        # mypath = "/storage/eirikgr/ANAntuples/PHYS_Data/"
        if isdir(mypath_data):
            df_data = getDataFrames(mypath_data)
            print(
                "Loading %s into dataframe with keys %s"
                % (mypath_data, ",".join(df_data.keys()))
            )
        else:
            df_data = {}

        df = {**df_mc, **df_data}

        for k in df.keys():

            if k not in ["Wjets"]:  # , "ttbar"]:
                continue

            print(df[k].GetColumnNames())
            exit()

            # print("Number of events in %s = %i" % (k, df[k].Count().GetValue()))

            # if not k in ["data18"]: continue

            isData = "data" in k

            if not isData:
                df[k] = df[k].Define(
                    "scaletolumi",
                    "(RandomRunNumber) < 320000 ? 36207.65 : (((RandomRunNumber) > 320000 && (RandomRunNumber) < 348000) ? 44307.4 : 58450.1)",
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
                df[k] = df[k].Define("is2018", "RandomRunNumber > 348000")

                # df[k] = df[k].Define("lepwgt_BL","getSF(lepBLRecoSF[ele_BL || muo_BL])")
                df[k] = df[k].Define("lepwgt_SG", "getSF(lepRecoSF[isGoodLep])")

                # df[k] = df[k].Define("trgwgt_BL","getSF(lepBLTrigSF[ele_BL || muo_BL])")
                df[k] = df[k].Define("trgwgt_SG", "getSF(lepTrigSF[isGoodLep])")

                # df[k] = df[k].Define("wgt_BL","(new_xsec ? (new_xsec) : (genWeight))*eventWeight*jvtWeight*bTagWeight*pileupWeight*scaletolumi*lepwgt_BL*trgwgt_BL")
                df[k] = df[k].Define(
                    "wgt_SG",
                    "(new_xsec ? (new_xsec) : (genWeight))*eventWeight*jvtWeight*bTagWeight*pileupWeight*scaletolumi*lepwgt_SG*trgwgt_SG",
                )

                # df[k] = df[k].Define("wgt_EV_BL","(eventWeight*jvtWeight*bTagWeight*pileupWeight*scaletolumi*lepwgt_BL*trgwgt_BL)")
                df[k] = df[k].Define(
                    "wgt_EV_SG",
                    "(eventWeight*jvtWeight*bTagWeight*pileupWeight*scaletolumi*lepwgt_SG*trgwgt_SG)",
                )

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
                df[k] = df[k].Define("is2018", "RunNumber >= 348885")

                # df[k] = df[k].Define("wgt_BL","1.0")
                df[k] = df[k].Define("wgt_SG", "1.0")
                df[k] = df[k].Define("wgt_EV", "1.0")

            # df[k].Define("lepIsTrigMatched_2L","is2015 ? trigmatch_2015_2L : (is2016 ? trigmatch_2016_2L : (is2017 ? trigmatch_2017_2L : trigmatch_2018_2L))")
            # df[k].Define("lepIsTrigMatched_3L","is2015 ? trigmatch_2015_3L : (is2016 ? trigmatch_2016_3L : (is2017 ? trigmatch_2017_3L : trigmatch_2018_3L))")

            # print("Nev(pileupWeight == 0) : ",df[k].Filter("pileupWeight == 0").Count().GetValue())
            # Check trigger matching!
            for tr in trigstr.keys():
                if tr == "3L":
                    continue
                for yr in trigstr[tr].keys():
                    # print("trigmatch_%s_%s"%(yr,tr))
                    df[k] = df[k].Define("trigmatch_%s_%s" % (yr, tr), trigstr[tr][yr])
                    df[k] = df[k].Define(
                        "triggered_%s_%s" % (yr, tr), evtrigstr[tr][yr]
                    )

            for nlep in ["1L", "2L"]:  # "1L"
                print(nlep)
                df[k] = df[k].Define(
                    "lepIsTrigMatched_%s" % nlep,
                    "is2015 ? trigmatch_2015_%s : (is2016 ? trigmatch_2016_%s : (is2017 ? trigmatch_2017_%s : trigmatch_2018_%s))"
                    % (nlep, nlep, nlep, nlep),
                )
                df[k] = df[k].Define(
                    "eventIsTriggered_%s" % nlep,
                    "is2015 ? triggered_2015_%s : (is2016 ? triggered_2016_%s : (is2017 ? triggered_2017_%s : triggered_2018_%s))"
                    % (nlep, nlep, nlep, nlep),
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

            # histo["nlep_BL_%s"%k] = df[k].Histo1D(("nlep_BL_%s"%k,"nlep_BL_%s"%k,10,0,10),"nlep_BL","wgt_SG")
            if create_histogram == True:
                # histo["nlep_SG_%s"%k] = df[k].Histo1D(("nlep_SG_%s"%k,"nlep_SG_%s"%k,10,0,10),"nlep_SG","wgt_SG")
                pass

            df[k] = df[k].Filter("nlep_BL == 3", "3 BL leptons")
            df[k] = df[k].Filter("nlep_SG == 3", "3 SG leptons")

            """
            Trigger filtering
            """

            # 2015 only triggers

            trigs2015 = triggers["2015"]
            trig2015 = trigs2015["trig"]
            trigmatch2015 = trigs2015["trigmatch"]

            df[k] = df[k].Filter(trig2015)
            df[k] = df[k].Filter(trigmatch2015)

            # 2016 only triggers

            trigs2016 = triggers["2015"]
            trig2016 = trigs2016["trig"]
            trigmatch2016 = trigs2016["trigmatch"]

            df[k] = df[k].Filter(trig2016)
            df[k] = df[k].Filter(trigmatch2016)

            # 2017 and 2018 only triggers

            trigs201718 = triggers["2017/18"]
            trig201718 = trigs201718["trig"]
            trigmatch201718 = trigs201718["trigmatch"]

            df[k] = df[k].Filter(trig201718)
            df[k] = df[k].Filter(trigmatch201718)

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

            """
            p = df[k].Display("channel").AsString()
            print(p)
            print(k)
            exit()
            """

            # Adding column for type of channel

            # df[k] = df[k].Define("channeltype", k)

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
                            histo[f"e_T_miss_%s" % (k)] = df[k].Histo1D(
                                (
                                    "h_%s_%s" % (f"e_T_miss", k),
                                    "h_%s_%s;m_{T}^{2}(23) [GeV];Entries"
                                    % (f"e_T_miss", k),
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
                            histo[f"m_T_{name}_%s" % (k)] = df[k].Histo1D(
                                (
                                    "h_%s_%s" % (f"m_T_{name}", k),
                                    "h_%s_%s;m_{T}^{2}(23) [GeV];Entries"
                                    % (f"e_T_miss", k),
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
                            histo[f"h_L_{name}_%s" % (k)] = df[k].Histo1D(
                                (
                                    "h_%s_%s" % (f"h_L_{name}", k),
                                    "h_%s_%s;m_{T}^{2}(23) [GeV];Entries"
                                    % (f"h_L_{name}", k),
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
                                histo[f"e_T_{name}_%s" % (k)] = df[k].Histo1D(
                                    (
                                        "h_%s_%s" % (f"e_T_{name}", k),
                                        "h_%s_%s;m_{T}^{2}(23) [GeV];Entries"
                                        % (f"e_T_{name}", k),
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
                                histo[f"delta_e_t_{name}_%s" % (k)] = df[k].Histo1D(
                                    (
                                        "h_%s_%s" % (f"delta_e_t_{name}", k),
                                        "h_%s_%s;m_{T}^{2}(23) [GeV];Entries"
                                        % (f"delta_e_t_{name}", k),
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
                                f"getM({pt1},{eta1}, {phi1}, {m1}, {pt2}, {eta2}, {phi2}, {m2}, {index1}, {index2}, {row}, {column})",
                            )
                            histo[f"{histo_name}_%s" % (k)] = df[k].Histo1D(
                                (
                                    "h_%s_%s" % (f"{histo_name}", k),
                                    "h_%s_%s;m_{T}^{2}(23) [GeV];Entries"
                                    % (f"{histo_name}", k),
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
                            histo[f"{histo_name}_%s" % (k)] = df[k].Histo1D(
                                (
                                    "h_%s_%s" % (f"{histo_name}", k),
                                    "h_%s_%s;m_{T}^{2}(23) [GeV];Entries"
                                    % (f"{histo_name}", k),
                                    50,
                                    0,
                                    1,
                                ),
                                f"{histo_name}",
                                "wgt_SG",
                            )

            df[k] = df[k].Define("ele3_pt", "getP_T(lepPt[ele_SG], 2)")
            df[k] = df[k].Define("ele3_eta", "getEta(lepEta[ele_SG], 2)")
            df[k] = df[k].Define("ele3_phi", "getPhi(lepPhi[ele_SG], 2)")
            df[k] = df[k].Define("ele3_m", "getm(lepM[ele_SG], 2)")

            df[k] = df[k].Define("muo3_pt", "getP_T(lepPt[muo_SG], 2)")
            df[k] = df[k].Define("muo3_eta", "getEta(lepEta[muo_SG], 2)")
            df[k] = df[k].Define("muo3_phi", "getPhi(lepPhi[muo_SG], 2)")
            df[k] = df[k].Define("muo3_m", "getm(lepM[muo_SG], 2)")

            """
            p = df_test.Display(("m_ele_0_ele_2", "m_jet_0_muo_2")).AsString()
            print(p)"""

            """
            p = df[k].Display(("m_jet_0_ele_2","m_jet_0_muo_2")).AsString()
            print(p)

            p = df[k].Display(("m_jet_1_ele_2","m_jet_1_muo_0")).AsString()
            print(p)
            p = df[k].Display(("m_jet_1_muo_2","m_ele_0_muo_2")).AsString()
            
            print(p)
            p = df[k].Display(("m_ele_0_ele_2","m_ele_0_muo_0")).AsString()
            print(p)
            p = df[k].Display(("m_ele_1_ele_2","m_ele_1_muo_2")).AsString()
            print(p)
            p = df[k].Display(("m_ele_2_muo_1","m_ele_2_muo_2")).AsString()
            print(p)
            p = df[k].Display(("m_muo_0_muo_2","m_muo_1_muo_2")).AsString()
            print(p)

            p = df[k].Display(("m_ele_2_muo_0")).AsString()
            print(p)
            
            """

            print(
                "Number of events in %s = %i after filtering"
                % (k, df[k].Count().GetValue())
            )

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


def create_histograms_pdfs(histo: dict, new_feats: list) -> None:
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

    for key in new_feats:
        try:
            p = pt.Plot(histo, key, toplot)
            p.can.SaveAs(str(histo_var) + f"/{key}.pdf")
        except:
            print(f"Could not make plot for name {key}")


def get_numpy_df(df: dict, all_cols: list) -> Tuple[dict, ...]:
    """_summary_

    Args:
        df (dict): _description_
        all_cols (list): _description_

    Returns:
        Tuple[dict, ...]: _description_
    """

    cols = df.keys()

    dfs = []
    for k in cols:

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
            str(df_storage) + f"/{k}_3lep_df_forML_bkg_signal_fromRDF.hdf5", "mini"
        )

    return dfs


if __name__ == "__main__":

    """Remove old images from histo_var_check"""
    # de = [f.unlink() for f in Path(histo_var).glob("*") if f.is_file()]

    """ Actual analysis """
    N_j = 2
    N_l = 6

    N_col = N_j + N_l + 1
    N_row = N_col

    good_runs = []

    histo = {}
    allhisto = []
    nEvents = 618282964
    nSlots = R.GetThreadPoolSize()
    print("Number of slots = %i" % nSlots)
    everyN = int(100 * nSlots)

    df, histo = runANA(
        str(bkg_sample),
        str(data_sample),
        everyN,
        fldic,
        histo,
        allhisto,
        create_histogram=True,
    )

    all_cols = get_column_names(df, histo)
    print(all_cols)

    create_histograms_pdfs(histo, all_cols)

    numpy_dfs = get_numpy_df(df, all_cols)

    names = list(df.keys())
    for index, df in enumerate(numpy_dfs):
        plot_rmm_matrix(df, names[index], rmm_structure, N_row)
