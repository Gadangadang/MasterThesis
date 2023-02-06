import sys 
import argparse
from AE import *
from corr import CorrCheck
from Noise import NoiseTrial
from DummyData import DummyData
from GradNoise import GradNoise
from Utilities.pathfile import *
from ptaltering import pTAltering
from ScaleAndPrep import ScaleAndPrep
from FakeParticles import FakeParticles
from OnePercentData import OnePercentData
from PlotScaledFeats import VizualizeFeats
from ChannelTraining import ChannelTraining
from HyperParameterTuning import HyperParameterTuning
from ptaltering_VAE import pTAltering_T
from Test_signals_veri import SignalDumVeri

sys.path.insert(1, "../")


def main():
    parser = argparse.ArgumentParser(
        description="Running training, tuning and inference"
    )
    parser.add_argument("-U", "--tune", action="store_true", help="Choose to tune")
    parser.add_argument("-T", "--train", action="store_true", help="Choose to train")
    parser.add_argument("-R", "--run", action="store_true", help="Run inference")
    parser.add_argument(
        "-E",
        "--exclude",
        action="store_true",
        help="Train and inference excluding every channel one at the time",
    )
    parser.add_argument("-L", "--onep", action="store_true", help="Run only on 1% of data")
    parser.add_argument("-D", "--dummy", action="store_true", help="Run only dummy sample in validation")
    parser.add_argument("-F", "--fake", action="store_true", help="Run with fake mc samples, i.e unphysical systems")
    parser.add_argument("-N", "--noise", action="store_true", help="Run with noise, to test")
    parser.add_argument("-A", "--altering", action="store_true", help="Run with different pt scaling")
    parser.add_argument("-C", "--corr", action="store_true", help="Check correlation")
    parser.add_argument("-V", "--vizfeat", action="store_true", help="Plot distributions")
    parser.add_argument("-G", "--gradnoise", action="store_true", help="Plot distributions")
    parser.add_argument("-FT", "--fake_t", action="store_true", help="Fake particles pytorch")
    parser.add_argument("-AT", "--altering_t", action="store_true", help="Fake particles pytorch")
    parser.add_argument("-S", "--signal_test", action="store_true", help="Try on dummy signals from SUSY")

    args = parser.parse_args()

    sp = ScaleAndPrep(DATA_PATH, True, SAVE_VAR, LOAD_VAR, lep=3)
    sp.MergeScaleAndSplit()
    
    

    rae = RunAE(sp, STORE_IMG_PATH)
    
    
    if args.signal_test:
        ST = SignalDumVeri(sp, STORE_IMG_PATH)
        ST.run()
    
    if args.altering_t:
        pt_T = pTAltering_T(sp, STORE_IMG_PATH)
        pt_T.run([1.5, 3, 5, 7, 10])
    
    
    if args.corr:
        C = CorrCheck(sp, STORE_IMG_PATH)
        C.checkCorr()
        
    if args.vizfeat:
        VF = VizualizeFeats(sp, STORE_IMG_PATH)
        VF.plotfeats()
        
    if args.gradnoise:
        GN = GradNoise(sp, STORE_IMG_PATH)
        #GN.genRmmEvent()
        GN.run(False, False, "sample")
    
    if args.exclude:
        CT = ChannelTraining(sp, STORE_IMG_PATH)
        CT.run(small=SMALL)
    
    if args.dummy:
        DD = DummyData(sp, STORE_IMG_PATH)
        DD.swapEventsInChannels(0.4, 0.01)
        
    if args.onep:
        OPD = OnePercentData(sp, STORE_IMG_PATH)
        OPD.run()
        
    if args.fake:
        FP = FakeParticles(sp, STORE_IMG_PATH)
        FP.run([[1,5], [2,5], [3,5], [4,5]], 0.1)
        
    if args.noise:
        N = NoiseTrial(sp, STORE_IMG_PATH)
        N.run()
    
    if args.altering:
        pTA = pTAltering(sp, STORE_IMG_PATH)
        pTA.run([1.5, 3, 5, 7, 10])
        
    if args.tune:
        
        HPT = HyperParameterTuning(sp, STORE_IMG_PATH)
        HPT.runHpSearch(rae.X_train, rae.X_val, rae.sample_weight, small=SMALL)

    if args.train:
        
        rae.trainModel(rae.X_train, rae.X_val, rae.sample_weight)

    if args.run:
        
        rae.runInference(rae.X_val, [], True)
        rae.checkReconError(rae.channels, sig_name="no_sig_10epoch")
        
    

    


if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(len(gpus), "Physical GPU")

    main()
