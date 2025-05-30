import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
import xgboost as xgb
import tkinter as tk
from tkinter import filedialog, messagebox

# Grupo\N{Invisible Times}NC = 1
ND1 = 1
ND2 = 0

# Paths (definidos via GUI)
PATCHES_DIR = ""  # pasta "patches" contendo subpastas 1-1059
TRAIN_IDS_FILE = ""  # arquivo .txt com IDs de treino, um por linha

# 1. Ler lista de IDs
def load_ids(id_file):
    with open(id_file, 'r') as f:
        ids = [line.strip() for line in f if line.strip().isdigit()]
    return ids

# 2. Carregar imagens apenas para IDs de treino/teste especificados
# folder structure: patches/<patient_id>/*.png

def load_images_by_ids(patches_dir, ids_list):
    images = []
    labels = []
    for pid in ids_list:
        patient_folder = os.path.join(patches_dir, pid)
        if not os.path.isdir(patient_folder):
            print(f"Aviso: pasta do paciente {pid} não existe.")
            continue
        for fname in os.listdir(patient_folder):
            if not fname.lower().endswith(('.png','.jpg','.jpeg')):
                continue
            path = os.path.join(patient_folder, fname)
            img = cv2.imread(path)
            if img is None:
                continue
            images.append(img)
            # extraia label da nomenclatura ou de arquivo externo, aqui apenas placeholder
            labels.append(pid)  # ou modifique para classe
    return images, labels

# 3. Segmentação via Watershed

def segment_nuclei(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, fg = cv2.threshold(dist, 0.7*dist.max(), 255, 0)
    fg = np.uint8(fg)
    unknown = cv2.subtract(opening, fg)
    _, markers = cv2.connectedComponents(fg)
    markers = markers + 1
    markers[unknown==255] = 0
    markers = cv2.watershed(img, markers)
    mask = (markers > 1).astype(np.uint8)
    return mask

# 4. Extração de descritores

def extract_features(images):
    import math
    from skimage.measure import regionprops, label
    feats = []
    for img in images:
        mask = segment_nuclei(img)
        lbl = label(mask)
        props = regionprops(lbl)
        areas, circ, ecc, dist = [], [], [], []
        centroids = [p.centroid for p in props]
        for p in props:
            area = p.area
            perim = p.perimeter if p.perimeter>0 else 1
            circularity = 4*math.pi*area/(perim**2)
            ecc_val = p.eccentricity
            dists = [np.linalg.norm(np.array(p.centroid)-np.array(c)) for c in centroids if c!=p.centroid]
            dmin = min(dists)/math.sqrt(area/math.pi) if dists else 0
            areas.append(area); circ.append(circularity)
            ecc.append(ecc_val); dist.append(dmin)
        feats.append({
            'area_mean': np.mean(areas), 'area_std': np.std(areas),
            'circ_mean': np.mean(circ), 'circ_std': np.std(circ),
            'ecc_mean': np.mean(ecc), 'ecc_std': np.std(ecc),
            'dist_mean': np.mean(dist), 'dist_std': np.std(dist)
        })
    return pd.DataFrame(feats)

# 5. Treino XGBoost incremental por paciente
def train_incremental_xgb(patches_dir, ids_list):
    # armazenar features e labels incremental
    all_feats = []
    all_labels = []
    for pid in ids_list:
        imgs, labs = load_images_by_ids(patches_dir, [pid])
        df = extract_features(imgs)
        all_feats.append(df)
        all_labels += labs
        print(f"Extraídas features paciente {pid}")
    X = pd.concat(all_feats, ignore_index=True)
    y = all_labels
    # split classique
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    print("Acurácia:", accuracy_score(y_te, preds))
    print("Matriz de confusão:\n", confusion_matrix(y_te, preds))

# GUI para selecionar arquivos e iniciar treino

def select_patches():
    global PATCHES_DIR
    PATCHES_DIR = filedialog.askdirectory(title='Selecione a pasta "patches"')
    messagebox.showinfo('OK', f'Patches: {PATCHES_DIR}')

def select_ids():
    global TRAIN_IDS_FILE
    TRAIN_IDS_FILE = filedialog.askopenfilename(title='Selecione arquivo de IDs', filetypes=[('TXT','*.txt')])
    messagebox.showinfo('OK', f'IDs: {TRAIN_IDS_FILE}')

def main_menu():
    root = tk.Tk()
    root.title('Treinamento Incremental')
    tk.Button(root, text='Selecionar patches', command=select_patches).pack(pady=5)
    tk.Button(root, text='Selecionar IDs de treino', command=select_ids).pack(pady=5)
    tk.Button(root, text='Treinar XGBoost', command=lambda: train_incremental_xgb(PATCHES_DIR, load_ids(TRAIN_IDS_FILE))).pack(pady=5)
    root.mainloop()

if __name__ == '__main__':
    main_menu()
