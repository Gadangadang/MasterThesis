from ROOT import TColor
from ROOT import (
    kBlack,
    kWhite,
    kGray,
    kRed,
    kPink,
    kMagenta,
    kViolet,
    kBlue,
    kAzure,
    kCyan,
    kTeal,
    kGreen,
    kSpring,
    kYellow,
    kOrange,
)

# ____________________________________________________________________________
def configure_samples():

    # Blues
    # myLighterBlue=TColor.GetColor('#deebf7')
    myLighterBlue = kAzure - 2
    myLightBlue = TColor.GetColor("#9ecae1")
    myMediumBlue = TColor.GetColor("#0868ac")
    myDarkBlue = TColor.GetColor("#08306b")

    # Greens
    myLightGreen = TColor.GetColor("#c7e9c0")
    myMediumGreen = TColor.GetColor("#41ab5d")
    myDarkGreen = TColor.GetColor("#006d2c")

    # Oranges
    myLighterOrange = TColor.GetColor("#ffeda0")
    myLightOrange = TColor.GetColor("#fec49f")
    myMediumOrange = TColor.GetColor("#fe9929")

    # Greys
    myLightestGrey = TColor.GetColor("#f0f0f0")
    myLighterGrey = TColor.GetColor("#e3e3e3")
    myLightGrey = TColor.GetColor("#969696")

    # Pinks
    myLightPink = TColor.GetColor("#fde0dd")
    myMediumPink = TColor.GetColor("#fcc5c0")
    myDarkPink = TColor.GetColor("#dd3497")

    # background path bkg_suffix
    data_suffix = "_SusySkim2LJetsLegacy_v1.1_SUSY2.root"
    # data_suffix='_merged_processed.root'
    bkg_suffix = "_merged_processed.root"

    d_type = {
        "Unknown": {
            "type": "Unknown",
            "leg": "Unknown",
            "f_color": kGray + 1,
            "l_color": kGray + 1,
            "m_color": kGray + 1,
            "m_type": 20,
        },
        "KnownUnknown": {
            "type": "KnownUnknown",
            "leg": "KnownUnknown",
            "f_color": kBlack + 3,
            "l_color": kBlack + 3,
            "m_color": kBlack + 3,
            "m_type": 20,
        },
        "IsoElectron": {
            "type": "IsoElectron",
            "leg": "IsoElectron",
            "f_color": kOrange,
            "l_color": kOrange,
            "m_color": kOrange,
            "m_type": 20,
        },
        "ChargeFlipIsoElectron": {
            "type": "ChargeFlipIsoElectron",
            "leg": "ChargeFlipIsoEl",
            "f_color": kOrange + 10,
            "l_color": kOrange + 10,
            "m_color": kOrange + 10,
            "m_type": 20,
        },
        "PromptMuon": {
            "type": "PromptMuon",
            "leg": "PromptMuon",
            "f_color": kAzure,
            "l_color": kAzure,
            "m_color": kAzure,
            "m_type": 20,
        },
        "PromptPhotonConversion": {
            "type": "PrPhotonConversion",
            "leg": "PromptPhotonConv",
            "f_color": kOrange + 3,
            "l_color": kOrange + 3,
            "m_color": kOrange + 3,
            "m_type": 20,
        },
        "ElectronFromMuon": {
            "type": "ElectronFromMuon",
            "leg": "ElectronFrom#mu",
            "f_color": kOrange + 1,
            "l_color": kOrange + 1,
            "m_color": kOrange + 1,
            "m_type": 20,
        },
        "TauDecay": {
            "type": "TauDecay",
            "leg": "TauDecay.",
            "f_color": kRed,
            "l_color": kRed,
            "m_color": kRed,
            "m_type": 20,
        },
        "BHadronDecay": {
            "type": "BHadronDecay",
            "leg": "BHadronDecay",
            "f_color": kSpring + 7,
            "l_color": kSpring + 7,
            "m_color": kSpring + 7,
            "m_type": 20,
        },
        "CHadronDecay": {
            "type": "CHadronDecay",
            "leg": "CHadronDecay",
            "f_color": kSpring - 7,
            "l_color": kSpring - 7,
            "m_color": kSpring - 7,
            "m_type": 20,
        },
        "LightFlavorDecay": {
            "type": "LightFlavorDecay",
            "leg": "LightFlavorDecay",
            "f_color": kAzure + 7,
            "l_color": kAzure + 7,
            "m_color": kAzure + 7,
            "m_type": 20,
        },
        "DATA": {
            "type": "DATA",
            "leg": "DATA",
            "f_color": kBlack,
            "l_color": kBlack,
            "m_color": kBlack,
            "m_type": 20,
        },
    }
    # 'LF' : {'type':'DATA.','leg':'DATA.', 'f_color':kAzure+7,'l_color':kAzure+7,'m_color':kAzure+7,'m_type':20},
    #     'HF' : {'type':'heavy flav.','leg':'heavy flav.', 'f_color':kSpring+7,'l_color':kSpring+7,'m_color':kSpring+7,'m_type':21},
    #     'CO' : {'type':'conversion','leg':'conversion',   'f_color':kOrange+7,'l_color':kOrange+7,'m_color':kOrange+7,'m_type':22},
    #     'CF' : {'type':'charge flip','leg':'charge flip', 'f_color':kViolet+7,'l_color':kViolet+7,'m_color':kViolet+7,'m_type':23},

    # }

    d_samp = {
        "mc16a": {
            "type": "mc",
            "leg": "MC",
            "f_color": kBlack,
            "l_color": kBlack,
            "path": "data" + bkg_suffix,
            "m_type": 23,
        },
        "mc16d": {
            "type": "mc",
            "leg": "MC",
            "f_color": kBlack,
            "l_color": kBlack,
            "path": "data" + bkg_suffix,
            "m_type": 23,
        },
        "mc16e": {
            "type": "mc",
            "leg": "MC",
            "f_color": kBlack,
            "l_color": kBlack,
            "path": "data" + bkg_suffix,
            "m_type": 23,
        },
        "data15-16": {
            "type": "data",
            "leg": "Data (2015,2016)",
            "f_color": kBlack,
            "l_color": kBlack,
            "path": "data" + data_suffix,
            "m_type": 24,
            "lumi": 3219.56 + 32988.1,
            "sqrts": 13,
        },
        "data15": {
            "type": "data",
            "leg": "Data (2015)",
            "f_color": kBlack,
            "l_color": kBlack,
            "path": "data" + data_suffix,
            "m_type": 24,
            "lumi": 3219.56,
            "sqrts": 13,
        },
        "data16": {
            "type": "data",
            "leg": "Data (2016)",
            "f_color": kBlack,
            "path": "data" + data_suffix,
            "l_color": kBlack,
            "m_type": 24,
            "lumi": 32988.1,
            "sqrts": 13,
        },
        "alldata": {
            "type": "data",
            "leg": "Data (2015-2018)",
            "f_color": kBlack,
            "l_color": kBlack,
            "path": "data" + data_suffix,
            "m_type": 20,
            "lumi": 44307.4 + 58450.1 + 32988.1 + 3219.56,
            "sqrts": 13,
        },
        "data17": {
            "type": "data",
            "leg": "Data (2017)",
            "f_color": kBlack,
            "l_color": kBlack,
            "path": "data" + data_suffix,
            "m_type": 25,
            "lumi": 44307.4,
            "sqrts": 13,
        },
        "data18": {
            "type": "data",
            "leg": "Data (2018)",
            "f_color": kBlack,
            "path": "data" + data_suffix,
            "l_color": kBlack,
            "m_type": 26,
            "lumi": 58450.1,
            "sqrts": 13,
        },
        "data22": {
            "type": "data",
            "leg": "Data (2022)",
            "f_color": kBlack,
            "l_color": kBlack,
            "path": "data" + data_suffix,
            "m_type": 26,
            "lumi": 1258.27,
            "sqrts": 13.6,
        },
        #'data'     :{'type':'data','leg':'Data15-16',                'f_color':0,'l_color':0,  'path': 'data15-16'+data_suffix},
        #'data'     :{'type':'data','leg':'Data17',                   'f_color':0,'l_color':0,  'path': 'data17'+data_suffix},
        #'ttbarDilep_410472' :{'type':'bkg', 'leg':'t#bar{t} (dilep)',                  'f_color':kRed-4,   'path':'ttbarDilep_410472'+bkg_suffix},
        "ttbar": {
            "type": "bkg",
            "leg": "t#bar{t}",
            "f_color": myMediumBlue,
            "path": "ttbar" + bkg_suffix,
        },
        "singleTop": {
            "type": "bkg",
            "leg": "Single top",
            "f_color": myLightBlue,
            "path": "singleTop" + bkg_suffix,
        },
        "singletop": {
            "type": "bkg",
            "leg": "Single top",
            "f_color": myLightBlue,
            "path": "singleTop" + bkg_suffix,
        },
        "Zjets": {
            "type": "bkg",
            "leg": "Z+jets",
            "f_color": myMediumGreen,
            "path": "Zjets" + bkg_suffix,
        },
        "ZjetsPhPy": {
            "type": "bkg",
            "leg": "Z+jets (PhPy)",
            "f_color": myDarkGreen,
            "path": "Zjets" + bkg_suffix,
        },
        "Zeejets": {
            "type": "bkg",
            "leg": "Z(ee)+jets",
            "f_color": kAzure - 9,
            "path": "Zeejets" + bkg_suffix,
        },
        """WmuHNL50_60G_lt01dd": {
            "type": "sig",
            "leg": "W' #rightarrow N ",
            "f_color": kRed,
            "path": "WmuHNL50_60G_lt01dd_lepfilt_ch" + bkg_suffix,
            "l_color": kBlack,
            "m_type": 26,
            "lumi": 58450.1,
            "sqrts": 13,
        },
        "ttbarHNLfullLepMLm15": {
            "type": "sig",
            "leg": "W' #rightarrow N ",
            "f_color": kRed+11,
            "path": "ttbar_HNLfullLep_ML_m_15" + bkg_suffix,
            "l_color": kRed+11,
            
        },
        "ttbarHNLfullLepMLm75": {
            "type": "sig",
            "leg": "W' #rightarrow N ",
            "f_color": kRed+14,
            "path": "ttbar_HNLfullLep_ML_m_75" + bkg_suffix,
            "l_color": kRed+14,
           
        },
        "ttbarHNLfullLepMLp15": {
            "type": "sig",
            "leg": "W' #rightarrow N ",
            "f_color": kRed+12,
            "path": "ttbar_HNLfullLep_ML_p_15" + bkg_suffix,
            "l_color": kRed+12,
            
        },
        "ttbarHNLfullLepMLp75": {
            "type": "sig",
            "leg": "W' #rightarrow N ",
            "f_color": kRed+13,
            "path": "ttbar_HNLfullLep_ML_p_75" + bkg_suffix,
            "l_color": kRed+13,
            
        },"""
        
        
        
        "MGPy8EGA14N23LOC1N2WZ450p0p0300p0p03L2L7": {
            "type": "sig",
            "leg": "W' #rightarrow N ",
            "f_color": kRed+13,
            "path": "MGPy8EG_A14N23LO_C1N2_WZ_450p0p0_300p0p0_3L_2L7" + bkg_suffix,
            "l_color": kRed+13,
            
        },
        
        "MGPy8EGA14N23LOC1N2WZ800p0p050p0p03L2L7": {
            "type": "sig",
            "leg": "W' #rightarrow N ",
            "f_color": kRed+13,
            "path": "MGPy8EG_A14N23LO_C1N2_WZ_450p0p0_300p0p0_3L_2L7" + bkg_suffix,
            "l_color": kRed+13,
            
        },
        
        # 'Zeejets1'    :{'type':'bkg', 'leg':'Z(ee)+jets1',                   'f_color':kOrange+10,  'path':'Zeejets'+bkg_suffix},
        # 'Zeejets2'    :{'type':'bkg', 'leg':'Z(ee)+jets2',                    'f_color':kOrange+9,  'path':'Zeejets'+bkg_suffix},
        # 'Zeejets3'    :{'type':'bkg', 'leg':'Z(ee)+jets3',                    'f_color':kOrange+8,  'path':'Zeejets'+bkg_suffix},
        # 'Zeejets4'    :{'type':'bkg', 'leg':'Z(ee)+jets4',                    'f_color':kOrange+7,  'path':'Zeejets'+bkg_suffix},
        # 'Zeejets5'    :{'type':'bkg', 'leg':'Z(ee)+jets5',                    'f_color':kOrange+6,  'path':'Zeejets'+bkg_suffix},
        # 'Zeejets6'    :{'type':'bkg', 'leg':'Z(ee)+jets6',                    'f_color':kOrange+5,  'path':'Zeejets'+bkg_suffix},
        # 'Zeejets7'    :{'type':'bkg', 'leg':'Z(ee)+jets7',                    'f_color':kOrange+4, 'path':'Zeejets'+bkg_suffix},
        # 'Zeejets8'    :{'type':'bkg', 'leg':'Z(ee)+jets8',                    'f_color':kOrange+3, 'path':'Zeejets'+bkg_suffix},
        # 'Zeejets9'    :{'type':'bkg', 'leg':'Z(ee)+jets9',                    'f_color':kOrange+2,  'path':'Zeejets'+bkg_suffix},
        "Zmmjets": {
            "type": "bkg",
            "leg": "Z(mm)+jets",
            "f_color": kAzure - 8,
            "path": "Zmmjets" + bkg_suffix,
        },
        "Zttjets": {
            "type": "bkg",
            "leg": "Z(tt)+jets",
            "f_color": kAzure - 7,
            "path": "Zttjets" + bkg_suffix,
        },
        "Wjets": {
            "type": "bkg",
            "leg": "W+jets",
            "f_color": myLightGreen,
            "path": "Wjets" + bkg_suffix,
        },
        "Diboson": {
            "type": "bkg",
            "leg": "Diboson",
            "f_color": myMediumOrange,
            "path": "diboson" + bkg_suffix,
        },
        "diboson2L": {
            "type": "bkg",
            "leg": "Diboson2L",
            "f_color": myMediumOrange,
            "path": "diboson" + bkg_suffix,
        },
        "diboson3L": {
            "type": "bkg",
            "leg": "Diboson3L",
            "f_color": myLightOrange,
            "path": "diboson" + bkg_suffix,
        },
        "diboson4L": {
            "type": "bkg",
            "leg": "Diboson4L",
            "f_color": myLighterOrange,
            "path": "diboson" + bkg_suffix,
            
        },
        #'dibosonPOWHEG'  :{'type':'bkg', 'leg':'Diboson (PH)',        'f_color':kAzure-3, 'path':'diboson (PH)'+bkg_suffix},
        "PHZZ": {
            "type": "bkg",
            "leg": "ZZ (powheg)",
            "f_color": kAzure + 7,
            "path": "PHZZ" + bkg_suffix,
        },
        "PHWZ": {
            "type": "bkg",
            "leg": "WZ (powheg)",
            "f_color": kAzure - 7,
            "path": "PHWZ" + bkg_suffix,
        },
        "PHWW": {
            "type": "bkg",
            "leg": "WW (powheg)",
            "f_color": kAzure,
            "path": "PHWW" + bkg_suffix,
        },
        "triboson": {
            "type": "bkg",
            "leg": "Triboson",
            "f_color": kGreen - 9,
            "path": "triboson" + bkg_suffix,
        },
        "Triboson": {
            "type": "bkg",
            "leg": "Triboson",
            "f_color": kGreen - 9,
            "path": "triboson" + bkg_suffix,
        },
        "Wjets_extension": {
            "type": "bkg",
            "leg": "Wjets (ext)",
            "f_color": kOrange,
            "path": "Wjets_extension" + bkg_suffix,
        },
        "Zjets_extension": {
            "type": "bkg",
            "leg": "Zjets (ext)",
            "f_color": kGray + 1,
            "path": "Zjets_extension" + bkg_suffix,
        },
        "Vgamma": {
            "type": "bkg",
            "leg": "V+gamma",
            "f_color": kCyan - 7,
            "path": "Vgamma" + bkg_suffix,
        },
        "XGamma": {
            "type": "bkg",
            "leg": "V+gamma",
            "f_color": kCyan - 7,
            "path": "Vgamma" + bkg_suffix,
        },
        "higgs": {
            "type": "bkg",
            "leg": "Higgs",
            "f_color": kAzure + 6,
            "path": "higgs" + bkg_suffix,
        },
        "lowMassDY": {
            "type": "bkg",
            "leg": "Low mass DY",
            "f_color": kMagenta - 7,
            "path": "lowMassDY" + bkg_suffix,
        },
        "topOther": {
            "type": "bkg",
            "leg": "Top other",
            "f_color": kGray,
            "path": "topOther" + bkg_suffix,
        },
        #'PythiaB' :{'type':'bkg', 'leg':'Top other',                 'f_color':kGray,'path':'topOther'+bkg_suffix},
        "FNP": {
            "type": "bkg",
            "leg": "MM",
            "f_color": kWhite,
            "path": "fake" + bkg_suffix,
        },
        "LF": {
            "type": "light flav.",
            "leg": "light flav.",
            "f_color": kAzure + 7,
            "l_color": kAzure + 7,
            "m_color": kAzure + 7,
            "m_type": 20,
        },
        "HF": {
            "type": "heavy flav.",
            "leg": "heavy flav.",
            "f_color": kSpring + 7,
            "l_color": kSpring + 7,
            "m_color": kSpring + 7,
            "m_type": 21,
        },
        "CO": {
            "type": "conversion",
            "leg": "conversion",
            "f_color": kOrange + 7,
            "l_color": kOrange + 7,
            "m_color": kOrange + 7,
            "m_type": 22,
        },
        "CF": {
            "type": "charge flip",
            "leg": "charge flip",
            "f_color": kViolet + 7,
            "l_color": kViolet + 7,
            "m_color": kViolet + 7,
            "m_type": 23,
        },
        "Unknown": {
            "type": "Unknown",
            "leg": "Unknown",
            "f_color": kGray + 1,
            "l_color": kGray + 1,
            "m_color": kGray + 1,
            "m_type": 20,
        },
        "KnownUnknown": {
            "type": "KnownUnknown",
            "leg": "KnownUnknown",
            "f_color": kBlack,
            "l_color": kBlack,
            "m_color": kBlack,
            "m_type": 20,
        },
        "IsoElectron": {
            "type": "IsoElectron",
            "leg": "IsoElectron",
            "f_color": kOrange,
            "l_color": kOrange,
            "m_color": kOrange,
            "m_type": 20,
        },
        "ChargeFlipIsoElectron": {
            "type": "ChargeFlipIsoElectron",
            "leg": "ChargeFlipIsoEl",
            "f_color": kOrange + 10,
            "l_color": kOrange + 10,
            "m_color": kOrange + 10,
            "m_type": 20,
        },
        "PromptMuon": {
            "type": "PromptMuon",
            "leg": "PromptMuon",
            "f_color": kAzure,
            "l_color": kAzure,
            "m_color": kAzure,
            "m_type": 20,
        },
        "PromptPhotonConversion": {
            "type": "PromptPhotonConversion",
            "leg": "PrPhotonConv",
            "f_color": kOrange + 3,
            "l_color": kOrange + 3,
            "m_color": kOrange + 3,
            "m_type": 20,
        },
        "ElectronFromMuon": {
            "type": "ElectronFromMuon",
            "leg": "ElectronFrom#mu",
            "f_color": kOrange + 1,
            "l_color": kOrange + 1,
            "m_color": kOrange + 1,
            "m_type": 20,
        },
        "TauDecay": {
            "type": "TauDecay",
            "leg": "TauDecay.",
            "f_color": kCyan,
            "l_color": kCyan,
            "m_color": kCyan,
            "m_type": 20,
        },
        "BHadronDecay": {
            "type": "BHadronDecay",
            "leg": "BHadronDecay",
            "f_color": kSpring + 7,
            "l_color": kSpring + 7,
            "m_color": kSpring + 7,
            "m_type": 20,
        },
        "CHadronDecay": {
            "type": "CHadronDecay",
            "leg": "CHadronDecay",
            "f_color": kSpring - 7,
            "l_color": kSpring - 7,
            "m_color": kSpring - 7,
            "m_type": 20,
        },
        "LightFlavorDecay": {
            "type": "LightFlavorDecay",
            "leg": "LightFlavorDecay",
            "f_color": kAzure + 7,
            "l_color": kAzure + 7,
            "m_color": kAzure + 7,
            "m_type": 20,
        },
        "DATA": {
            "type": "DATA",
            "leg": "DATA",
            "f_color": kBlack,
            "l_color": kBlack,
            "m_color": kBlack,
            "m_type": 20,
        },
        "all": {
            "type": "MC",
            "leg": "MC",
            "f_color": kBlack,
            "l_color": kBlack,
            "m_color": kBlack,
            "m_type": 20,
        },
    }

    d_reg = {
        "REAL2L12": {"descr": "2L (p_{T} > 25), N_{jet}^{p_{T}>30} #geq 2, MET > 80"},
        "REAL2L11": {"descr": "2L (p_{T} > 25), N_{jet}^{p_{T}>30} #geq 2, MET > 80"},
        "REAL2L10": {"descr": "2L (p_{T} > 25), N_{jet}^{p_{T}>30} #geq 2, MET > 80"},
        "REAL2L09": {"descr": "2L (p_{T} > 25), N_{jet}^{p_{T}>30} #geq 2, MET > 80"},
        "REAL2L02": {"descr": "2L (p_{T} > 25), |m_{ll}-m_{z}|<10"},
        "REAL2L03": {
            "descr": "2L (p_{T} > 25), N_{jet}^{p_{T}>30} #geq 2, |m_{ll}-m_{z}|<10"
        },
        "REAL2L04": {"descr": "2L (p_{T} > 25), N_{jet}^{p_{T}>30} #geq 2"},
        "REAL2L05": {
            "descr": "2L (p_{T} > 25), N_{jet}^{p_{T}>30} #geq 2, N_{b-jet}^{p_{T}>20} == 0"
        },
        "REAL2L06": {
            "descr": "2L (p_{T} > 25), N_{jet}^{p_{T}>30} #geq 2, N_{b-jet}^{p_{T}>20} > 0"
        },
        "REAL2L07": {
            "descr": "2L (p_{T} > 25), N_{jet}^{p_{T}>30} #geq 2, |m_{ll}-m_{z}|>30"
        },
        "REAL2L08": {
            "descr": "2L (p_{T} > 25), N_{jet}^{p_{T}>30} #geq 2, MET_{sign} > 10"
        },
        "FAKE2L01": {"descr": "2L (p_{T} > 25), N_{b-jet}^{p_{T}>20} > 0, MET < 40"},
        "FAKE2L02": {"descr": "2L (p_{T} > 25), N_{b-jet}^{p_{T}>20} == 0"},
        "FAKE2L04": {"descr": "2L (p_{T} > 25), N_{jet}^{p_{T}>30} #geq 2, MET < 40"},
        "FAKE2L05": {"descr": "2L (p_{T} > 27/9)"},  # {"descr": "2L (p_{T} > 25)"},
        "FAKE2L21": {"descr": "Heavy fl. CR"},
        "FAKE2L23": {"descr": "Conversion CR"},
    }

    # python -i compareEfficiencies.py MMinput_2L2JFAR.root h_lep_pT_eff_EEOS_E_REAL2L09_2L2J_diboson_REBIN_mc16eEST2 h_lep_pT_eff_EEOS_E_REAL2L09_2L2J_ttbar_REBIN_mc16eEST2 h_lep_pT_eff_EEOS_E_REAL2L09_2L2J_Zjets_REBIN_mc16eEST2 h_lep_pT_eff_EEOS_E_REAL2L09_2L2J_singleTop_REBIN_mc16eEST2 h_lep_pT_eff_EEOS_E_trueREAL_REAL2L09_2L2J_diboson_REBIN_mc16eEST2 h_lep_pT_eff_EEOS_E_trueREAL_REAL2L09_2L2J_ttbar_REBIN_mc16eEST2 h_lep_pT_eff_EEOS_E_trueREAL_REAL2L09_2L2J_Zjets_REBIN_mc16eEST2 h_lep_pT_eff_EEOS_E_trueREAL_REAL2L09_2L2J_singleTop_REBIN_mc16eEST2

    # python -i compareEfficiencies.py MMinput_2L2JFAR.root h_lep_pT_eff_MMOS_M_REAL2L09_2L2J_diboson_REBIN_mc16eEST2 h_lep_pT_eff_MMOS_M_REAL2L09_2L2J_ttbar_REBIN_mc16eEST2 h_lep_pT_eff_MMOS_M_REAL2L09_2L2J_Zjets_REBIN_mc16eEST2 h_lep_pT_eff_MMOS_M_REAL2L09_2L2J_singleTop_REBIN_mc16eEST2 h_lep_pT_eff_MMOS_M_trueREAL_REAL2L09_2L2J_diboson_REBIN_mc16eEST2 h_lep_pT_eff_MMOS_M_trueREAL_REAL2L09_2L2J_ttbar_REBIN_mc16eEST2 h_lep_pT_eff_MMOS_M_trueREAL_REAL2L09_2L2J_Zjets_REBIN_mc16eEST2 h_lep_pT_eff_MMOS_M_trueREAL_REAL2L09_2L2J_singleTop_REBIN_mc16eEST2 h_lep_pT_eff_MMOS_M_trueREAL_REAL2L09_2L2J_all_mc16eEST2 h_lep_pT_eff_MMOS_M_REAL2L09_2L2J_all_mc16eEST2

    # d_samp = {

    #     'data15-16'     :{'type':'data','leg':'Data (2015,2016)',                     'f_color':1,'l_color':1,  'path': 'data'+data_suffix},
    #     'data17'     :{'type':'data','leg':'Data (2017)',                     'f_color':2,'l_color':2,  'path': 'data'+data_suffix},
    #     #'data'     :{'type':'data','leg':'Data15-16',                'f_color':0,'l_color':0,  'path': 'data15-16'+data_suffix},
    #     #'data'     :{'type':'data','leg':'Data17',                   'f_color':0,'l_color':0,  'path': 'data17'+data_suffix},
    #     'ttbar'    :{'type':'bkg', 'leg':'t#bar{t}',                 'f_color':myMediumBlue,   'path':'ttbar'+bkg_suffix},
    #     'singleTop':{'type':'bkg', 'leg':'Single top',               'f_color':myLightBlue,  'path':'singleTop'+bkg_suffix},
    #     'Zjets'    :{'type':'bkg', 'leg':'Z+jets',                    'f_color':myMediumGreen,  'path':'Zjets'+bkg_suffix},
    #     'Wjets'    :{'type':'bkg', 'leg':'W+jets',                    'f_color':myLightGreen,  'path':'Wjets'+bkg_suffix},
    #     'diboson'  :{'type':'bkg', 'leg':'Diboson',                  'f_color':myMediumOrange, 'path':'diboson'+bkg_suffix},
    #     'triboson' :{'type':'bkg', 'leg':'Triboson',                 'f_color':myLightOrange,'path':'triboson'+bkg_suffix},
    #     'Vgamma'   :{'type':'bkg', 'leg':'V+gamma',                   'f_color':myLighterOrange,'path':'Vgamma'+bkg_suffix},
    #     'higgs'    :{'type':'bkg', 'leg':'Higgs',                    'f_color':myLightGrey,'path':'higgs'+bkg_suffix},
    #     'lowMassDY':{'type':'bkg', 'leg':'Low mass DY',                'f_color':myLighterGrey,'path':'lowMassDY'+bkg_suffix},
    #     'fake':{'type':'bkg', 'leg':'MM',                'f_color':kGray+1,'path':'fake'+bkg_suffix},
    #     'LF' : {'type':'light flav.','leg':'light flav.', 'f_color':kAzure+7,'l_color':kAzure+7,'m_color':kAzure+7,'m_type':20},
    #     'HF' : {'type':'heavy flav.','leg':'heavy flav.', 'f_color':kSpring+7,'l_color':kSpring+7,'m_color':kSpring+7,'m_type':21},
    #     'CO' : {'type':'conversion','leg':'conversion',   'f_color':kOrange+7,'l_color':kOrange+7,'m_color':kOrange+7,'m_type':22},
    #     'CF' : {'type':'charge flip','leg':'charge flip', 'f_color':kViolet+7,'l_color':kViolet+7,'m_color':kViolet+7,'m_type':23},

    #     }

    return d_samp, d_type, d_reg
