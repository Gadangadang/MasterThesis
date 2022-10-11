import argparse
from pathfile import *
from utilities import *



def main():
    parser = argparse.ArgumentParser(description='Running training, tuning and inference')
    parser.add_argument("-U","--tune", action='store_true', help='Choose to tune')
    parser.add_argument("-T", '--train', action='store_true', help='Choose to train')
    parser.add_argument("-R", '--run', action='store_true', help='Run inference')

    args = parser.parse_args()
    
    
    
    
    sp = ScaleAndPrep(DATA_PATH)
    sp.MergeScaleAndSplit()
 
    rae = RunAE(sp, STORE_IMG_PATH)
    
    if args.tune:
        rae.hyperParamSearch()
    
    if args.train:
        rae.trainModel()
    
    if args.run:
        rae.runInference(True)
        rae.checkReconError()



if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
    
    
    
    
    main()
    

    