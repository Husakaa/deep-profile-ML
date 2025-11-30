import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.callbacks import Callback

class WarmUpCallback(Callback):
    """
    Implementación del Warm-up para el peso Beta de la pérdida KL.
    Fuente: VAE_3Layers_Model.py (Repo original DeepProfile)
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
        
        # --- ENCODER ---
        # Nota: En modelos subclassed, no necesitamos definir self.encoder_input explícitamente
        # para el forward pass, los inputs llegan directamente a la primera capa Dense.
        
        self.dense1 = layers.Dense(intermediate_dim1, activation='linear')
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.Activation('relu')
        
        self.dense2 = layers.Dense(intermediate_dim2, activation='linear')
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.Activation('relu')
        
        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)

        # --- DECODER ---
        self.decoder_h1 = layers.Dense(intermediate_dim2, activation='relu')
        self.decoder_h2 = layers.Dense(intermediate_dim1, activation='relu')
        self.decoder_out = layers.Dense(original_dim, activation='linear') # Linear para reconstrucción (MSE)

    def sampling(self, args):
        """
        Truco de reparametrización para permitir backpropagation.
        z = z_mean + exp(z_log_var / 2) * epsilon
        """
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean), mean=0., stddev=1.0)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def call(self, inputs):
        # --- FORWARD PASS (ENCODER) ---
        # Corrección: Pasamos 'inputs' directamente a dense1
        x = self.dense1(inputs) 
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        
        # --- SAMPLING ---
        z = self.sampling([z_mean, z_log_var])
        
        # --- FORWARD PASS (DECODER) ---
        x = self.decoder_h1(z)
        x = self.decoder_h2(x)
        reconstruction = self.decoder_out(x)
        
        # Devolvemos los 3 valores necesarios para calcular la pérdida
        return reconstruction, z_mean, z_log_var