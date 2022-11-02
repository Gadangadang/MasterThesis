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

    args = parser.parse_args()

    sp = ScaleAndPrep(DATA_PATH, True)
    sp.MergeScaleAndSplit()

    rae = RunAE(sp, STORE_IMG_PATH)

    

    if args.tune:
        rae.hyperParamSearch(rae.X_train, rae.X_val, rae.sample_weight, small=False)
        
    if args.train:
        rae.trainModel(rae.X_train, rae.X_val, rae.sample_weight)

    if args.run:
        rae.runInference(rae.X_val, [], True)
        rae.checkReconError(rae.channels, sig_name="no_sig_10epoch")

    if args.exclude:
        rae.channelTrainings(small=False)


if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

    main()
