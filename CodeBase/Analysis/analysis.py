import argparse
from pathfile import *
from utilities import *


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

    args = parser.parse_args()

    sp = ScaleAndPrep(DATA_PATH, True, SAVE_VAR, LOAD_VAR)
    sp.MergeScaleAndSplit()

    rae = RunAE(sp, STORE_IMG_PATH)
    
    if args.exclude:
        CT = ChannelTraining(sp, STORE_IMG_PATH)
        CT.run(small=SMALL)
    
    if args.dummy:
        DD = DummyData(sp, STORE_IMG_PATH)
        DD.swapEventsInChannels(0.4, 0.01)
        
    if args.onep:
        OPD = OnePercentData(sp, STORE_IMG_PATH)
        OPD.run()
        
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

    main()
