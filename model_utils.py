from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Flatten, Dense, Reshape

def build_autoencoder(input_shape):
    """Defines an autoencoder model."""
    inputs = Input(shape=input_shape)
    # Encoder
    x = Conv2D(32, 3, activation='relu', strides=2, padding='same')(inputs)
    x = Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
    x = Flatten()(x)
    encoded = Dense(128, activation='relu')(x)

    # Decoder
    x = Dense(8 * 8 * 64, activation='relu')(encoded)
    x = Reshape((8, 8, 64))(x)
    x = Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
    x = Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
    decoded = Conv2DTranspose(3, 3, activation='sigmoid', padding='same')(x)

    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder