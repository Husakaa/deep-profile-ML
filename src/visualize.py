import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
import os

if len(sys.argv) < 2:
    print("Uso: python src/visualize.py <TIPO_CANCER>")
    sys.exit(1)

CANCER_TYPE = sys.argv[1]
FILE_PATH = f'results/{CANCER_TYPE}_embeddings.tsv'

if not os.path.exists(FILE_PATH):
    print("Primero debes ejecutar train.py para generar los embeddings.")
    sys.exit(1)

print(f"--> Visualizando espacio latente para {CANCER_TYPE}...")
df = pd.read_csv(FILE_PATH, sep='\t', index_col=0)

# Reducir a 2D usando PCA para poder pintar
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df.values)

# Crear gráfico
plt.figure(figsize=(10, 8))
plt.scatter(principal_components[:, 0], principal_components[:, 1], 
            alpha=0.6, c='#0072B2', s=10)
plt.title(f'Espacio Latente DeepProfile (PCA) - {CANCER_TYPE}\n{df.shape[0]} Pacientes')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.grid(True, alpha=0.3)

output_img = f'results/{CANCER_TYPE}_latent_space.png'
plt.savefig(output_img)
print(f"Gráfico guardado en: {output_img}")