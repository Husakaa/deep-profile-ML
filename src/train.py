import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, backend as K
from vae_model import VAE, WarmUpCallback
import os

# Configuración
INPUT_FILE = '../data/expression_data.tsv' 
OUTPUT_DIR = '../results/'
LATENT_DIM = 50  
EPOCHS = 50
BATCH_SIZE = 50
LEARNING_RATE = 0.0005

# Crear directorio si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Cargar Datos
print("Cargando datos...")

df = pd.read_csv(INPUT_FILE, sep='\t', index_col=0) 
data = df.values.astype('float32')
original_dim = data.shape[1]

# 2. Configurar Modelo
beta = K.variable(0.0) # Iniciamos beta en 0 para el warm-up
vae = VAE(original_dim, intermediate_dim1=100, intermediate_dim2=50, latent_dim=LATENT_DIM)
optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)

# 3. Loop de Entrenamiento Personalizado (para manejar la pérdida compleja)
@tf.function
def train_step(x_batch):
    with tf.GradientTape() as tape:
        reconstruction, z_mean, z_log_var = vae(x_batch)
        
        # Pérdida de Reconstrucción (MSE)
        reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(x_batch, reconstruction)) * original_dim
        
        # Pérdida KL
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        kl_loss = tf.reduce_mean(kl_loss)
        
        total_loss = reconstruction_loss + (beta * kl_loss)
        
    grads = tape.gradient(total_loss, vae.trainable_weights)
    optimizer.apply_gradients(zip(grads, vae.trainable_weights))
    return total_loss, reconstruction_loss, kl_loss

print("Iniciando entrenamiento...")
for epoch in range(EPOCHS):
    # Actualizar beta (Warm-up manual simple o usar el Callback si usamos model.fit)
    if K.get_value(beta) <= 1:
         K.set_value(beta, K.get_value(beta) + (1.0/EPOCHS))
         
    # Iterar por batches
    total_loss_avg = 0
    for i in range(0, len(data), BATCH_SIZE):
        x_batch = data[i:i+BATCH_SIZE]
        loss, rec, kl = train_step(x_batch)
        total_loss_avg += loss
        
    print(f"Epoch {epoch+1}, Loss: {total_loss_avg.numpy():.4f}, Beta: {K.get_value(beta):.2f}")

# 4. Extraer y Guardar Embeddings (Espacio Latente)
_, z_mean_pred, _ = vae(data)
embeddings_df = pd.DataFrame(z_mean_pred.numpy(), index=df.index)
embeddings_df.to_csv(f"{OUTPUT_DIR}/vae_embeddings_{LATENT_DIM}L.tsv", sep='\t')
print("Embeddings guardados.")