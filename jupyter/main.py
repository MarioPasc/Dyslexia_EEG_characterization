# -*- coding: utf-8 -*-
"""

@author: ignac
"""

# Reimportar dependencias después del reinicio del entorno
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import os
from scipy.spatial.distance import cdist
from PIL import Image
from skimage.transform import resize

# Crear carpeta para guardar imágenes si no existe
save_dir = "attractors"
os.makedirs(save_dir, exist_ok=True)

# Funciones auxiliares

def mutual_information(signal, max_lag=100, bins=64):
    ami = []
    for tau in range(1, max_lag + 1):
        x1 = signal[:-tau]
        x2 = signal[tau:]
        hist_2d, _, _ = np.histogram2d(x1, x2, bins=bins)
        pxy = hist_2d / np.sum(hist_2d)
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        px_py = np.outer(px, py)
        nonzero = pxy > 0
        ami_val = np.sum(pxy[nonzero] * np.log(pxy[nonzero] / px_py[nonzero]))
        ami.append(ami_val)
    return np.array(ami)

def estimate_tau(signal, max_lag=100):
    ami = mutual_information(signal, max_lag=max_lag)
    for i in range(1, len(ami)-1):
        if ami[i] < ami[i-1] and ami[i] < ami[i+1]:
            return i + 1
    return np.argmin(ami) + 1

def false_nearest_neighbors(signal, tau, max_dim=10, R_tol=10.0, A_tol=2.0):
    fnn_percentages = []
    N = len(signal)
    for m in range(1, max_dim + 1):
        n_points = N - (m + 1) * tau
        if n_points <= 0:
            break
        embedded_m = np.array([signal[i:i + m * tau:tau] for i in range(n_points)])
        embedded_m1 = np.array([signal[i:i + (m + 1) * tau:tau] for i in range(n_points)])
        nbrs = NearestNeighbors(n_neighbors=2).fit(embedded_m)
        distances, indices = nbrs.kneighbors(embedded_m)
        nn_indices = indices[:, 1]
        R_m = distances[:, 1]
        delta = np.abs(embedded_m1[:, -1] - embedded_m1[nn_indices, -1])
        R_tol_mask = delta / R_m > R_tol
        A_tol_mask = delta > A_tol * np.std(signal)
        fnn = np.logical_or(R_tol_mask, A_tol_mask)
        fnn_percentage = 100 * np.sum(fnn) / n_points
        fnn_percentages.append(fnn_percentage)
    return fnn_percentages

def estimate_embedding_dimension(signal, tau, max_dim=10):
    fnn = false_nearest_neighbors(signal, tau, max_dim)
    for i, val in enumerate(fnn):
        if val < 1.0:
            return i + 1
    return np.argmin(fnn) + 1

def reconstruct_attractor(signal, tau, m):
    n_points = len(signal) - (m - 1) * tau
    return np.array([signal[i:i + m * tau:tau] for i in range(n_points)])

def estimate_epsilon_global(attractor, recurrence_rate=0.15, block_size=500, n_blocks=3):
    N = attractor.shape[0]
    step = N // (n_blocks + 1)
    samples = []

    for i in range(1, n_blocks + 1):
        start = max(0, i * step - block_size // 2)
        end = min(N, start + block_size)
        samples.append(attractor[start:end])

    combined = np.vstack(samples)
    distances = cdist(combined, combined).flatten()
    distances = distances[distances > 0]  # eliminar ceros de la diagonal

    sorted_distances = np.sort(distances)
    cutoff_index = int(len(sorted_distances) * recurrence_rate)
    epsilon = sorted_distances[cutoff_index]

    return epsilon

def process_signal(signal, signal_id, save_dir='attractors', recurrence_rate=0.15, block_size=1000):
    print(f"🔁 Procesando {signal_id}...")

    # Paso 1: estimar parámetros
    print("🔍 Estimando tau y m...")
    tau = estimate_tau(signal)
    m = estimate_embedding_dimension(signal, tau)
    attractor = reconstruct_attractor(signal, tau, m)
    N = attractor.shape[0]

    # # Paso 2: guardar atractor
    # os.makedirs(save_dir, exist_ok=True)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.plot(attractor[:, 0], attractor[:, 1], attractor[:, 2], lw=0.5)
    # ax.set_title(f'Signal {signal_id} - τ={tau}, m={m}')
    # ax.axis('off')
    # filename = f"{save_dir}/attractor_{signal_id}.png"
    # plt.savefig(filename, dpi=300)
    # plt.close()

    # Paso 3: estimar epsilon
    print("📐 Estimando epsilon global...")
    epsilon = estimate_epsilon_global(attractor, recurrence_rate)

    # Paso 4: calcular RP por bloques
    print("🧮 Calculando Recurrence Plot por bloques...")
    RP = np.zeros((N, N), dtype=np.uint8)

    total_blocks = (N + block_size - 1) // block_size
    for bi, i in enumerate(range(0, N, block_size)):
        end_i = min(i + block_size, N)
        A = attractor[i:end_i]
        for bj, j in enumerate(range(0, N, block_size)):
            end_j = min(j + block_size, N)
            B = attractor[j:end_j]
            print(f"   ▶️ Bloque [{bi+1}/{total_blocks}]x[{bj+1}/{total_blocks}]...")
            dist_block = cdist(A, B)
            RP[i:end_i, j:end_j] = (dist_block < epsilon).astype(np.uint8)

    #Paso 5: guardar RP

    channel_index = int(signal_id.split('_ch')[-1])
    channel_name = ch_names[channel_index]  # Usamos la lista de nombres que ya tenés
    
    # Crear la carpeta por canal y grupo
    channel_dir = os.path.join(save_dir, channel_name, 'control')  # cambiar a 'control' cuando corresponda
    os.makedirs(channel_dir, exist_ok=True)
    
    # Redimensionar y guardar la imagen
    RP_small = resize(RP, (1024, 1024), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
    img = Image.fromarray((1 - RP_small) * 255)  # 1 = negro (recurrencia)
    filename_rp = os.path.join(channel_dir, f"RP_{signal_id}.png")
    img.save(filename_rp)


    print(f"✅ Terminado {signal_id}\n")

    return tau, m, epsilon



#INICIO DEL SCRIPT

# Nombres de electrodos
ch_names = ['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','C4','T8','TP9','CP5',
            'CP1','CP2','CP6','TP10','P7','P3','Pz','P4','P8','PO9','O1','Oz','O2','PO10','Cz']

# Cargar archivo
data_path = "C:/Users/ignac/Mi unidad/BioSIP/Datos EEG/PC_c_preprocess_40.npy"
data = np.load(data_path)  # Esperado shape: (n, 32, 68000, 5)

# Extraer dimensiones
n_subjects, n_channels, n_timepoints, n_bands = data.shape

# Crear carpeta si no existe
os.makedirs("attractors", exist_ok=True)

# Usar sólo canales 0 a 30 (excluir Cz, índice 31)
usable_channels = list(range(31))  # Índices del 0 al 30

# Usar sólo la última banda (gamma), que es índice -1 o 4
band_index = 4

# DataFrame para resultados
results_df = pd.DataFrame(columns=['subject_id', 'channel_index', 'channel_name', 'tau', 'embedding_dim', 'label', 'epsilon'])

# Función principal
def main():
    global results_df
    for subj in range(n_subjects):
        for ch in usable_channels:
            signal = data[subj, ch, :, band_index]
            signal_id = f"subj{subj:02d}_ch{ch:02d}"
            try:
                tau, m, epsilon = process_signal(signal, signal_id)
                results_df = pd.concat([results_df, pd.DataFrame([{
                    'subject_id': subj,
                    'channel_index': ch,
                    'channel_name': ch_names[ch],
                    'tau': tau,
                    'embedding_dim': m,
                    'label': 0,  # Grupo 
                    'epsilon':epsilon
                }])], ignore_index=True)
            except Exception as e:
                print(f"Error con {signal_id}: {e}")
    
    # Guardar resultados en CSV
    results_df.to_csv("takens_results_gamma_control.csv", index=False)

#Ejecutar si se llama directamente
if __name__ == "__main__":
    main()
