import argparse
from pathfile import *
from utilities import *




if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
    tf.keras.utils.get_custom_objects()['leaky_relu'] = tf.keras.layers.LeakyReLU()
    
    """
    parser = argparse.ArgumentParser(description='Running training, tuning and inference')
    parser.add_argument("--tune", help='Choose to tune')
    parser.add_argument('--train', help='Choose to train')
    parser.add_argument('--run', help='Run inference')

    args = parser.parse_args()
    print(args)"""
    

    sp = ScaleAndPrep(DATA_PATH)
    sp.MergeScaleAndSplit()
 
    rae = RunAE(sp, STORE_IMG_PATH)
    #rae.hyperParamSearch()
    rae.runInference(True)
    rae.checkReconError()
