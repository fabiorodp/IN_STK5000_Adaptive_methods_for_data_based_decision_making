import tensorflow as tf


def mlp(input_shape, num_of_hidden_layers=0,
        hls_units=[], hls_act_functs=[],
        optimizer=tf.optimizers.Adam(),
        loss='binary_crossentropy', metrics=['accuracy'],
        output_shape=1, output_act_funct=tf.nn.sigmoid,
        dropout=0.2):
    """DNN Model for predicting multivariate targets."""
    # initializing model
    mlp = tf.keras.Sequential()
    mlp.add(tf.keras.Input(shape=input_shape))

    # creating the hidden layers according to its qtd
    if num_of_hidden_layers == 1:

        mlp.add(
            tf.keras.layers.Dense(
                units=hls_units[0],
                activation=hls_act_functs[0]
            )
        )  # hidden layer

        mlp.add(tf.keras.layers.Dropout(dropout))

    elif num_of_hidden_layers > 1:
        for i in range(num_of_hidden_layers):
            mlp.add(
                tf.keras.layers.Dense(
                    units=hls_units[i],
                    activation=hls_act_functs[i]
                )
            )  # hidden layers

            mlp.add(tf.keras.layers.Dropout(dropout))

    # creating output layer
    mlp.add(
        tf.keras.layers.Dense(
            units=output_shape,
            activation=output_act_funct
        )
    )  # output layer

    mlp.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return mlp
