import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.callbacks import Callback

class WarmUpCallback(Callback):
    """
    Implementación del Warm-up para el peso Beta de la pérdida KL.
    Esto evita que el término KL domine al principio del entrenamiento (colapso posterior).
    Fuente: VAE_3Layers_Model.py
    """
    def __init__(self, beta_var, kappa):
        super(WarmUpCallback, self).__init__()
        self.beta_var = beta_var
        self.kappa = kappa

    def on_epoch_end(self, epoch, logs=None):
        current_beta = K.get_value(self.beta_var)
        if current_beta <= 1.0:
            K.set_value(self.beta_var, current_beta + self.kappa)

class VAE(models.Model):
    def __init__(self, original_dim, intermediate_dim1, intermediate_dim2, latent_dim):
        super(VAE, self).__init__()
        self.original_dim = original_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder_input = layers.InputLayer(input_shape=(original_dim,))
        self.dense1 = layers.Dense(intermediate_dim1, activation='linear')
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.Activation('relu')
        
        self.dense2 = layers.Dense(intermediate_dim2, activation='linear')
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.Activation('relu')
        
        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)

        # Decoder
        self.decoder_h1 = layers.Dense(intermediate_dim2, activation='relu')
        self.decoder_h2 = layers.Dense(intermediate_dim1, activation='relu')
        self.decoder_out = layers.Dense(original_dim, activation='linear')

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean), mean=0., stddev=1.0)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def call(self, inputs):
        # Forward pass del Encoder
        x = self.encoder_input(inputs)
        x = self.act1(self.bn1(self.dense1(x)))
        x = self.act2(self.bn2(self.dense2(x)))
        
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        
        # Reparameterization Trick
        z = self.sampling([z_mean, z_log_var])
        
        # Forward pass del Decoder
        x = self.decoder_h1(z)
        x = self.decoder_h2(x)
        reconstruction = self.decoder_out(x)
        
        # Añadir la pérdida KL internamente
        # Beta se gestiona externamente para el warm-up
        return reconstruction, z_mean, z_log_var