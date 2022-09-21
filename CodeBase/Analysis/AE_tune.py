import keras_tuner as kt
import tensorflow as tf

def gridautoencoder(X_b, X_back_test):
    tuner = kt.Hyperband(
        AE_model_builder,
        objective=kt.Objective("val_mse", direction="min"),
        max_epochs=15,
        factor=3,
        directory="GridSearches",
        project_name="AE",
        overwrite=True,
    )

    tuner.search(X_b, X_b, epochs=15, batch_size=8192,
                validation_data=(X_back_test, X_back_test))
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(
        f"""
    For Encoder: \n 
    First layer has {best_hps.get('num_of_neurons1')} with activation {best_hps.get('1_act')} \n
    Second layer has {best_hps.get('num_of_neurons2')} with activation {best_hps.get('2_act')} \n
    Third layer has {best_hps.get('num_of_neurons3')} with activation {best_hps.get('3_act')} \n
    
    Latent layer has {best_hps.get("lat_num")} with activation {best_hps.get('2_act')} \n
    \n
    For Decoder: \n 
    First layer has {best_hps.get('num_of_neurons5')} with activation {best_hps.get('5_act')}\n
    Second layer has {best_hps.get('num_of_neurons6')} with activation {best_hps.get('6_act')}\n
    Third layer has {best_hps.get('num_of_neurons7')} with activation {best_hps.get('7_act')}\n
    Output layer has activation {best_hps.get('8_act')}\n
    \n
    with learning rate = {best_hps.get('learning_rate')} and alpha = {best_hps.get('alpha')}
    """
    )

    state = True
    while state == True:
        answ = input("Do you want to save model? (y/n) ")
        if answ == "y":
            name = input("name: ")
            tuner.hypermodel.build(best_hps).save(
                f"../tf_models/model_{name}.h5")
            state = False
            print("Model saved")
        elif answ == "n":
            state = False
            print("Model not saved")


def AE_model_builder(hp):
    ker_choice = hp.Choice("Kernel_reg", values=[0.5, 0.1, 0.05, 0.01])
    act_choice = hp.Choice("Atc_reg", values=[0.5, 0.1, 0.05, 0.01])

    alpha_choice = hp.Choice("alpha", values=[1., 0.5, 0.1, 0.05, 0.01])
    #get_custom_objects().update({"leakyrelu": tf.keras.layers.LeakyReLU(alpha=alpha_choice)})
    activations = {
        "relu": tf.nn.relu,
        "tanh": tf.nn.tanh,
        "leakyrelu": lambda x: tf.nn.leaky_relu(x, alpha=alpha_choice),
        "linear": tf.keras.activations.linear
    }
    inputs = tf.keras.layers.Input(shape=data_shape, name="encoder_input")
    x = tf.keras.layers.Dense(
        units=hp.Int("num_of_neurons1", min_value=20, max_value=data_shape-1, step=1),
        activation=activations.get(hp.Choice(
            "1_act", ["relu", "tanh", "leakyrelu","linear"])),
        kernel_regularizer=tf.keras.regularizers.L1(ker_choice),
        activity_regularizer=tf.keras.regularizers.L2(act_choice))(inputs)
    x_ = tf.keras.layers.Dense(
        units=hp.Int("num_of_neurons2", min_value=15, max_value=19, step=1),
        activation=activations.get(hp.Choice(
            "2_act", ["relu", "tanh", "leakyrelu","linear"])))(x)
    x1 = tf.keras.layers.Dense(
        units=hp.Int("num_of_neurons3", min_value=10, max_value=14, step=1),
        activation=activations.get(hp.Choice(
            "3_act", ["relu", "tanh", "leakyrelu","linear"])),
        kernel_regularizer=tf.keras.regularizers.L1(ker_choice),
        activity_regularizer=tf.keras.regularizers.L2(act_choice)
    )(x_)
    val = hp.Int("lat_num", min_value=1, max_value=9, step=1)
    x2 = tf.keras.layers.Dense(
        units=val, activation=activations.get(hp.Choice(
            "4_act", ["relu", "tanh", "leakyrelu","linear"]))
    )(x1)
    encoder = tf.keras.Model(inputs, x2, name="encoder")

    latent_input = tf.keras.layers.Input(shape=val, name="decoder_input")
    x = tf.keras.layers.Dense(
        units=hp.Int("num_of_neurons5", min_value=10, max_value=14, step=1),
        activation=activations.get(hp.Choice(
            "5_act", ["relu", "tanh", "leakyrelu","linear"])),
        kernel_regularizer=tf.keras.regularizers.L1(ker_choice),
        activity_regularizer=tf.keras.regularizers.L2(act_choice)
    )(latent_input)
    
    x_ = tf.keras.layers.Dense(
        units=hp.Int("num_of_neurons6", min_value=15, max_value=19, step=1),
        activation=activations.get(hp.Choice(
            "6_act", ["relu", "tanh", "leakyrelu","linear"])))(x)
    
    x1 = tf.keras.layers.Dense(
        units=hp.Int("num_of_neurons7", min_value=20, max_value=data_shape-1, step=1),
        activation=activations.get(hp.Choice(
            "7_act", ["relu", "tanh", "leakyrelu","linear"])),
        kernel_regularizer=tf.keras.regularizers.L1(ker_choice),
        activity_regularizer=tf.keras.regularizers.L2(act_choice)
    )(x_)
    output = tf.keras.layers.Dense(
        data_shape, activation=activations.get(hp.Choice(
            "8_act", ["relu", "tanh", "leakyrelu","linear"]))
    )(x1)
    decoder = tf.keras.Model(latent_input, output, name="decoder")

    outputs = decoder(encoder(inputs))
    AE_model = tf.keras.Model(inputs, outputs, name="AE_model")

    hp_learning_rate = hp.Choice("learning_rate", values=[
                                 9e-2, 9.5e-2, 1e-3, 1.5e-3])
    optimizer = tf.keras.optimizers.Adam(hp_learning_rate)
    #optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    AE_model.compile(loss="mse", optimizer=optimizer, metrics=["mse"])

    return AE_model