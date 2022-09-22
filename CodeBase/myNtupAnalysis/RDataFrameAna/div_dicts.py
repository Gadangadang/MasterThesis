triggers = {
    "2015": {
        "trig": "trigMatch_HLT_mu18_mu8noL1 || trigMatch_HLT_2e12_lhloose_L12EM10VH || trigMatch_HLT_e17_lhloose_mu14  || is2016 || is2017 || is2018", 
        "trigmatch": "ROOT::VecOps::Sum(lepHLT_2e12_lhloose_L12EM10VH[isGoodLep] && lepPt[isGoodLep] > 12  || lepHLT_mu18_mu8noL1[isGoodLep] && lepPt[isGoodLep] > 18  || lepHLT_e17_lhloose_mu14[isGoodLep] && lepPt[isGoodLep] > 17) >= 2 || is2016 || is2017 || is2018"
    },
    "2016": {
        "trig": "trigMatch_HLT_2e17_lhvloose_nod0 || trigMatch_HLT_e17_lhloose_nod0_mu14 || trigMatch_HLT_mu22_mu8noL1  || is2017 || is2018",
        "trigmatch": "ROOT::VecOps::Sum(lepHLT_2e17_lhvloose_nod0[isGoodLep] && lepPt[isGoodLep] > 17  || lepHLT_e17_lhloose_nod0_mu14[isGoodLep] && lepPt[isGoodLep] > 17  || lepHLT_mu22_mu8noL1[isGoodLep] && lepPt[isGoodLep] > 22) >= 2 || is2017 || is2018"
    },
    "2017/18": {
        "trig": "trigMatch_HLT_2e17_lhvloose_nod0_L12EM15VHI || trigMatch_HLT_e17_lhloose_nod0_mu14 || trigMatch_HLT_mu22_mu8noL1",
        "trigmatch": "ROOT::VecOps::Sum( lepHLT_2e17_lhvloose_nod0_L12EM15VHI[isGoodLep] && lepPt[isGoodLep] > 17  || lepHLT_e17_lhloose_nod0_mu14[isGoodLep] && lepPt[isGoodLep] > 17  || lepHLT_mu22_mu8noL1[isGoodLep] && lepPt[isGoodLep] > 22 >= 2) " 
    }
}

rmm_structure = {
        1: [
            "jet_0",
            "jetPt[jet_SG]",
            "jetEta[jet_SG]",
            "jetPhi[jet_SG]",
            "jetM[jet_SG]",
            0
        ],
        2: [
            "jet_1",
            "jetPt[jet_SG]",
            "jetEta[jet_SG]",
            "jetPhi[jet_SG]",
            "jetM[jet_SG]",
            1
        ],
        3: [
            "ele_0",
            "lepPt[ele_SG]",
            "lepEta[ele_SG]",
            "lepPhi[ele_SG]",
            "lepM[ele_SG]",
            0
        ],
        4: [
            "ele_1",
            "lepPt[ele_SG]",
            "lepEta[ele_SG]",
            "lepPhi[ele_SG]",
            "lepM[ele_SG]",
            1
        ],
        
        5: [
            "muo_0",
            "lepPt[muo_SG]",
            "lepEta[muo_SG]",
            "lepPhi[muo_SG]",
            "lepM[muo_SG]",
            0
        ],
        6: [
            "muo_1",
            "lepPt[muo_SG]",
            "lepEta[muo_SG]",
            "lepPhi[muo_SG]",
            "lepM[muo_SG]",
            1
        ],
        
}

"""
5: [
            "ele_2",
            "lepPt[ele_SG]",
            "lepEta[ele_SG]",
            "lepPhi[ele_SG]",
            "lepM[ele_SG]",
            2
        ],

8: [
"muo_2",
"lepPt[muo_SG]",
"lepEta[muo_SG]",
"lepPhi[muo_SG]",
"lepM[muo_SG]",
2"""



columns = {
    "lepHLT_e24_lhvloose_nod0_2e12_lhvloose_nod0_L1EM20VH_3EM10VH",
    "lepHLT_e12_lhloose_nod0_2mu10",
    "lepHLT_2e12_lhloose_nod0_mu10",
    "lepHLT_mu20_2mu4noL1",
    "lepHLT_3mu6",
    "lepHLT_3mu6_msonly",
    "lepHLT_e17_lhloose_nod0_2e10_lhloose_nod0_L1EM15VH_3EM8VH",
    "lepHLT_e17_lhloose_2e9_lhloose",
    "lepHLT_mu18_2mu4noL1",
    "lepHLT_2e12_lhloose_mu10",
    "lepHLT_e12_lhloose_2mu10",
    "lepHLT_e24_lhmedium_L1EM20VH",
    "lepHLT_e60_lhmedium",
    "lepHLT_e120_lhloose",
    "lepHLT_mu20_iloose_L1MU15",
    "lepHLT_2e12_lhloose_L12EM10VH",
    "lepHLT_mu18_mu8noL1",
    "lepHLT_e17_lhloose_mu14",
    "lepHLT_e7_lhmedium_mu24",
    "lepHLT_e24_lhtight_nod0_ivarloose",
    "lepHLT_e24_lhmedium_nod0_L1EM20VH",
    "lepHLT_e60_medium",
    "lepHLT_mu40",
    "lepHLT_mu24_iloose_L1MU15",
    "lepHLT_mu24_ivarloose_L1MU15",
    "lepHLT_mu24_ivarmedium",
    "lepHLT_mu24_imedium",
    "lepHLT_mu26_imedium",
    "lepHLT_2e15_lhvloose_nod0_L12EM13VH",
    "lepHLT_2e17_lhvloose_nod0",
    "lepHLT_2mu10",
    "lepHLT_2mu14",
    "lepHLT_mu20_mu8noL1",
    "lepHLT_mu22_mu8noL1",
    "lepHLT_e24_lhmedium_nod0_L1EM20VHI_mu8noL1",
    "lepHLT_e26_lhtight_nod0_ivarloose",
    "lepHLT_e26_lhtight_nod0",
    "lepHLT_e60_lhmedium_nod0",
    "lepHLT_e140_lhloose_nod0",
    "lepHLT_e300_etcut",
    "lepHLT_mu26_ivarmedium",
    "lepHLT_mu50",
    "lepHLT_mu60_0eta105_msonly",
    "lepHLT_2e17_lhvloose_nod0_L12EM15VHI",
    "lepHLT_2e24_lhvloose_nod0",
    "lepHLT_e17_lhloose_nod0_mu14",
    "lepHLT_e26_lhmedium_nod0_mu8noL1",
    "lepHLT_e7_lhmedium_nod0_mu24",
    "trigMatch_1L2LTrig",
    "trigMatch_3LTrig",
    "trigMatch_1LTrig",
    "trigMatch_3LTrigOR",
    "trigMatch_1L2LTrigOR",
    "trigMatch_1LTrigOR",
    "trigMatch_2LTrig",
    "trigMatch_2LTrigOR",
    "trigMatch_HLT_e24_lhmedium_L1EM20VH",
    "trigMatch_HLT_e60_lhmedium",
    "trigMatch_HLT_e120_lhloose",
    "trigMatch_HLT_mu20_iloose_L1MU15",
    "trigMatch_HLT_2e12_lhloose_L12EM10VH",
    "trigMatch_HLT_mu18_mu8noL1",
    "trigMatch_HLT_e17_lhloose_mu14",
    "trigMatch_HLT_e7_lhmedium_mu24",
    "trigMatch_HLT_e24_lhtight_nod0_ivarloose",
    "trigMatch_HLT_e24_lhmedium_nod0_L1EM20VH",
    "trigMatch_HLT_e60_medium",
    "trigMatch_HLT_mu40",
    "trigMatch_HLT_mu24_iloose_L1MU15",
    "trigMatch_HLT_mu24_ivarloose_L1MU15",
    "trigMatch_HLT_mu24_ivarmedium",
    "trigMatch_HLT_mu24_imedium",
    "trigMatch_HLT_mu26_imedium",
    "trigMatch_HLT_2e15_lhvloose_nod0_L12EM13VH",
    "trigMatch_HLT_2e17_lhvloose_nod0",
    "trigMatch_HLT_2mu10",
    "trigMatch_HLT_2mu14",
    "trigMatch_HLT_mu20_mu8noL1",
    "trigMatch_HLT_mu22_mu8noL1",
    "trigMatch_HLT_e24_lhmedium_nod0_L1EM20VHI_mu8noL1",
    "trigMatch_HLT_e26_lhtight_nod0_ivarloose",
    "trigMatch_HLT_e26_lhtight_nod0",
    "trigMatch_HLT_e60_lhmedium_nod0",
    "trigMatch_HLT_e140_lhloose_nod0",
    "trigMatch_HLT_e300_etcut",
    "trigMatch_HLT_mu26_ivarmedium",
    "trigMatch_HLT_mu50",
    "trigMatch_HLT_mu60_0eta105_msonly",
    "trigMatch_HLT_2e17_lhvloose_nod0_L12EM15VHI",
    "trigMatch_HLT_2e24_lhvloose_nod0",
    "trigMatch_HLT_e17_lhloose_nod0_mu14",
    "trigMatch_HLT_e26_lhmedium_nod0_mu8noL1",
    "trigMatch_HLT_e7_lhmedium_nod0_mu24",
    "mu",
    "avg_mu",
    "actual_mu",
    "nVtx",
    "channel",
    "nLep_base",
    "nLep_signal",
    "lepFlavor",
    "lepCharge",
    "lepAuthor",
    "lepPt",
    "lepEta",
    "lepPhi",
    "lepM",
    "lepD0",
    "lepD0Sig",
    "lepZ0",
    "lepZ0SinTheta",
    "lepPtcone20",
    "lepPtcone30",
    "lepPtcone40",
    "lepTopoetcone20",
    "lepTopoetcone30",
    "lepTopoetcone40",
    "lepPtvarcone20",
    "lepPtvarcone30",
    "lepPtvarcone40",
    "lepPtvarcone30_TightTTVA_pt1000",
    "lepPtvarcone30_TightTTVA_pt500",
    "lepPtvarcone20_TightTTVA_pt1000",
    "lepPtvarcone20_TightTTVA_pt500",
    "lepPtcone20_TightTTVALooseCone_pt500",
    "lepPtcone20_TightTTVALooseCone_pt1000",
    "lepPtcone20_TightTTVA_pt500",
    "lepPtcone20_TightTTVA_pt1000",
    "lepNeflowisol20",
    "lepPassOR",
    "lepType",
    "lepOrigin",
    "lepIFFClass",
    "lepEgMotherType",
    "lepEgMotherOrigin",
    "lepEgMotherPdgId",
    "lepECIDS",
    "lepPassBL",
    "lepVeryLoose",
    "lepLoose",
    "lepMedium",
    "lepTight",
    "lepIsoFCHighPtCaloOnly",
    "lepIsoFCLoose",
    "lepIsoFCTight",
    "lepIsoFCLoose_FixedRad",
    "lepIsoFCTight_FixedRad",
    "lepIsoHighPtCaloOnly",
    "lepIsoTightTrackOnly_VarRad",
    "lepIsoTightTrackOnly_FixedRad",
    "lepIsoLoose_VarRad",
    "lepIsoTight_VarRad",
    "lepIsoPLVLoose",
    "lepIsoPLVTight",
    "lepTruthMatched",
    "lepTruthCharge",
    "lepTruthPt",
    "lepTruthEta",
    "lepTruthPhi",
    "lepTruthM",
    "lepTrigSF",
    "lepRecoSF",
    "lepBLTrigSF",
    "lepBLRecoSF",
    "nJet30",
    "nJet20",
    "jetPt",
    "jetEta",
    "jetPhi",
    "jetM",
    "jetJVT",
    "jetPassOR",
    "jetSignal",
    "mjj",
    "jetTileEnergy",
    "jetMV2c10",
    "jetdl1",
    "jetdl1r",
    "met_Et",
    "met_Sign",
    "met_Phi",
    "mll",
    "pileupWeight",
    "leptonWeight",
    "baselineleptonWeight",
    "eventWeight",
    "bTagWeight",
    "jvtWeight",
    "globalDiLepTrigSF",
    "globalBaselineDiLepTrigSF",
    "singleLepTrigSF",
    "singleBaselineLepTrigSF",
    "flavSymWeight",
    "genWeight",
    "genWeightUp",
    "genWeightDown",
    "PRWHash",
    "EventNumber",
    "xsec",
    "GenHt",
    "GenMET",
    "DatasetNumber",
    "RunNumber",
    "RandomRunNumber",
    "FS",
    "LHE3Weights",
    "LHE3WeightNames",
}
