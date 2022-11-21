import tensorflow as tf 
from CodeBase.Analysis.Utilities.pathfile import *
from os.path import isfile, join
from os import listdir


trained_models = [f for f in listdir(TF_MODEL_PATH) if isfile(join(TF_MODEL_PATH, f))] 

for model in trained_models:
    
    start = model.find("o_")
    end = model.find(".h5")
    
    name = model[start+2:end]
    print(name)
    AE_model = tf.keras.models.load_model(
                        "tf_models/" + model
                    )

    tf.keras.utils.plot_model(AE_model, to_file="/Users/Sakarias/MasterThesis/Figures/testing/model_plots/" + f"{name}_model_plot.pdf", show_shapes=True, show_layer_names=True, expand_nested=True)
