LOAD_VAR = True #False 
SAVE_VAR = False #True
SCALER = "MinMax" # or "Standard"
BACTH_SIZE = int(8192)      
EPOCHS = 10

TYPE = "VAE" #"AE"

nbjet = 6
nljet = 6
nele = 5
nmuo = 5

RMMSIZE = 1 + nbjet + nljet + nele + nmuo
SMALL = True

TOKEN = "5789363537:AAF0SErRfZ07yWrzjp9pg9oCCO6H8BfFLHw"
chat_id = "5733209220"

rmm_structure = {
    1: [
        "ljet_0",
        "jetPt[ljet]",
        "jetEta[ljet]",
        "jetPhi[ljet]",
        "jetM[ljet]",
        0,
    ],
    2: [
        "ljet_1",
        "jetPt[ljet]",
        "jetEta[ljet]",
        "jetPhi[ljet]",
        "jetM[ljet]",
        1,
    ],
    3: [
        "ljet_2",
        "jetPt[ljet]",
        "jetEta[ljet]",
        "jetPhi[ljet]",
        "jetM[ljet]",
        2,
    ],
    4: [
        "ljet_3",
        "jetPt[ljet]",
        "jetEta[ljet]",
        "jetPhi[ljet]",
        "jetM[ljet]",
        3,
    ],
    5: [
        "ljet_4",
        "jetPt[ljet]",
        "jetEta[ljet]",
        "jetPhi[ljet]",
        "jetM[ljet]",
        4,
    ],
    6: [
        "ljet_5",
        "jetPt[ljet]",
        "jetEta[ljet]",
        "jetPhi[ljet]",
        "jetM[ljet]",
        5,
    ],
    7: [
        "bjet_0",
        "jetPt[bjet77]",
        "jetEta[bjet77]",
        "jetPhi[bjet77]",
        "jetM[bjet77]",
        0,
    ],
    8: [
        "bjet_1",
        "jetPt[bjet77]",
        "jetEta[bjet77]",
        "jetPhi[bjet77]",
        "jetM[bjet77]",
        1,
    ],
    9: [
        "bjet_2",
        "jetPt[bjet77]",
        "jetEta[bjet77]",
        "jetPhi[bjet77]",
        "jetM[bjet77]",
        2,
    ],
    10: [
        "bjet_3",
        "jetPt[bjet77]",
        "jetEta[bjet77]",
        "jetPhi[bjet77]",
        "jetM[bjet77]",
        3,
    ],
    11: [
        "bjet_4",
        "jetPt[bjet77]",
        "jetEta[bjet77]",
        "jetPhi[bjet77]",
        "jetM[bjet77]",
        4,
    ],
    12: [
        "bjet_5",
        "jetPt[bjet77]",
        "jetEta[bjet77]",
        "jetPhi[bjet77]",
        "jetM[bjet77]",
        5,
    ],
    13: [
        "ele_0",
        "lepPt[ele_SG]",
        "lepEta[ele_SG]",
        "lepPhi[ele_SG]",
        "lepM[ele_SG]",
        0,
    ],
    14: [
        "ele_1",
        "lepPt[ele_SG]",
        "lepEta[ele_SG]",
        "lepPhi[ele_SG]",
        "lepM[ele_SG]",
        1,
    ],
    15: [
        "ele_2",
        "lepPt[ele_SG]",
        "lepEta[ele_SG]",
        "lepPhi[ele_SG]",
        "lepM[ele_SG]",
        2,
    ],
    16: [
        "ele_3",
        "lepPt[ele_SG]",
        "lepEta[ele_SG]",
        "lepPhi[ele_SG]",
        "lepM[ele_SG]",
        3,
    ],
    17: [
        "ele_4",
        "lepPt[ele_SG]",
        "lepEta[ele_SG]",
        "lepPhi[ele_SG]",
        "lepM[ele_SG]",
        4,
    ],
    18: [
        "muo_0",
        "lepPt[muo_SG]",
        "lepEta[muo_SG]",
        "lepPhi[muo_SG]",
        "lepM[muo_SG]",
        0,
    ],
    19: [
        "muo_1",
        "lepPt[muo_SG]",
        "lepEta[muo_SG]",
        "lepPhi[muo_SG]",
        "lepM[muo_SG]",
        1,
    ],
    20: [
        "muo_2",
        "lepPt[muo_SG]",
        "lepEta[muo_SG]",
        "lepPhi[muo_SG]",
        "lepM[muo_SG]",
        2,
    ],
    21: [
        "muo_3",
        "lepPt[muo_SG]",
        "lepEta[muo_SG]",
        "lepPhi[muo_SG]",
        "lepM[muo_SG]",
        3,
    ],
    22: [
        "muo_4",
        "lepPt[muo_SG]",
        "lepEta[muo_SG]",
        "lepPhi[muo_SG]",
        "lepM[muo_SG]",
        4,
    ],
}


"""
rmm_structure = {
    1: [
        "jet_0",
        "jetPt[jet_SG]",
        "jetEta[jet_SG]",
        "jetPhi[jet_SG]",
        "jetM[jet_SG]",
        0,
    ],
    2: [
        "jet_1",
        "jetPt[jet_SG]",
        "jetEta[jet_SG]",
        "jetPhi[jet_SG]",
        "jetM[jet_SG]",
        1,
    ],
    3: [
        "jet_2",
        "jetPt[jet_SG]",
        "jetEta[jet_SG]",
        "jetPhi[jet_SG]",
        "jetM[jet_SG]",
        2,
    ],
    4: [
        "jet_3",
        "jetPt[jet_SG]",
        "jetEta[jet_SG]",
        "jetPhi[jet_SG]",
        "jetM[jet_SG]",
        3,
    ],
    5: [
        "ele_0",
        "lepPt[ele_SG]",
        "lepEta[ele_SG]",
        "lepPhi[ele_SG]",
        "lepM[ele_SG]",
        0,
    ],
    6: [
        "ele_1",
        "lepPt[ele_SG]",
        "lepEta[ele_SG]",
        "lepPhi[ele_SG]",
        "lepM[ele_SG]",
        1,
    ],
    7: [
        "ele_2",
        "lepPt[ele_SG]",
        "lepEta[ele_SG]",
        "lepPhi[ele_SG]",
        "lepM[ele_SG]",
        2,
    ],
    8: [
        "ele_3",
        "lepPt[ele_SG]",
        "lepEta[ele_SG]",
        "lepPhi[ele_SG]",
        "lepM[ele_SG]",
        3,
    ],
    9: [
        "ele_4",
        "lepPt[ele_SG]",
        "lepEta[ele_SG]",
        "lepPhi[ele_SG]",
        "lepM[ele_SG]",
        4,
    ],
    10: [
        "muo_0",
        "lepPt[muo_SG]",
        "lepEta[muo_SG]",
        "lepPhi[muo_SG]",
        "lepM[muo_SG]",
        0,
    ],
    11: [
        "muo_1",
        "lepPt[muo_SG]",
        "lepEta[muo_SG]",
        "lepPhi[muo_SG]",
        "lepM[muo_SG]",
        1,
    ],
    12: [
        "muo_2",
        "lepPt[muo_SG]",
        "lepEta[muo_SG]",
        "lepPhi[muo_SG]",
        "lepM[muo_SG]",
        2,
    ],
    13: [
        "muo_3",
        "lepPt[muo_SG]",
        "lepEta[muo_SG]",
        "lepPhi[muo_SG]",
        "lepM[muo_SG]",
        3,
    ],
    14: [
        "muo_4",
        "lepPt[muo_SG]",
        "lepEta[muo_SG]",
        "lepPhi[muo_SG]",
        "lepM[muo_SG]",
        4,
    ],
}
"""