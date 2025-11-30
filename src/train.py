import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from vae_model import VAE
import os
import sys

# --- CONFIGURACIÓN ---
if len(sys.argv) < 2:
    print("Uso: python src/train.py <TIPO_CANCER>")
    print("Ejemplo: python src/train.py OV")
    sys.exit(1)

CANCER_TYPE = sys.argv[1]
DATA_DIR = 'data'
RESULTS_DIR = 'results'
INPUT_FILENAME = f"{CANCER_TYPE}_DATA_TOP2_JOINED_BATCH_CORRECTED_CLEANED.tsv"
INPUT_PATH = os.path.join(DATA_DIR, INPUT_FILENAME)

LATENT_DIM = 50
EPOCHS = 50
BATCH_SIZE = 50
LEARNING_RATE = 0.0005

os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. Cargar Datos Reales
print(f"--> Cargando datos de expresión para {CANCER_TYPE}...")
try:
    df = pd.read_csv(INPUT_PATH, sep='\t', index_col=0)
    print(f"    Matriz cargada: {df.shape[0]} pacientes x {df.shape[1]} genes.")
except FileNotFoundError:
    print(f"ERROR: No encuentro '{INPUT_FILENAME}' en la carpeta 'data/'.")
    sys.exit(1)

# Preprocesamiento: Escalar datos entre 0 y 1 
print("--> Normalizando datos...")
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df.values)
original_dim = data_scaled.shape[1]

# 2. Configurar Modelo
beta = K.variable(0.0)
vae = VAE(original_dim, 100, 50, LATENT_DIM)
optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)

@tf.function
def train_step(x_batch):
    with tf.GradientTape() as tape:
        reconstruction, z_mean, z_log_var = vae(x_batch)
        # Loss: Error de Reconstrucción + Divergencia KL (con peso beta)
        reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(x_batch, reconstruction)) * original_dim
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        kl_loss = tf.reduce_mean(kl_loss)
        total_loss = reconstruction_loss + (beta * kl_loss)
    
    grads = tape.gradient(total_loss, vae.trainable_weights)
    optimizer.apply_gradients(zip(grads, vae.trainable_weights))
    return total_loss

# 3. Entrenamiento
print(f"--> Entrenando DeepProfile ({EPOCHS} épocas)...")
dataset = tf.data.Dataset.from_tensor_slices(data_scaled).shuffle(1000).batch(BATCH_SIZE)

for epoch in range(EPOCHS):
    # Warm-up: Beta sube gradualmente de 0 a 1
    new_beta = min(1.0, epoch / 20.0) # Warm-up más rápido (20 épocas)
    K.set_value(beta, new_beta)
    
    loss_metric = 0
    steps = 0
    for x_batch in dataset:
        loss = train_step(x_batch)
        loss_metric += loss
        steps += 1
        
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss_metric/steps:.2f} | Beta: {K.get_value(beta):.2f}")

# 4. Evaluación y Guardado
print("--> Generando embeddings y evaluando reconstrucción...")
reconstruction, z_mean_pred, _ = vae(data_scaled)

# Cálculo de R2 (Qué tan bien conserva el modelo la información biológica)
r2 = r2_score(data_scaled.flatten(), reconstruction.numpy().flatten())
print(f"\nRESULTADO FINAL:")
print(f"R² Score (Reconstrucción): {r2:.4f}")
print("(Un R² cercano a 1.0 indica que el VAE ha capturado la biología correctamente)")

# Guardar Embeddings
embeddings_df = pd.DataFrame(z_mean_pred.numpy(), index=df.index)
output_file = os.path.join(RESULTS_DIR, f"{CANCER_TYPE}_embeddings.tsv")
embeddings_df.to_csv(output_file, sep='\t')
print(f"Embeddings guardados en: {output_file}")