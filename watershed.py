import os
import cv2
import numpy as np
import pandas as pd
import math
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
from skimage import measure
from skimage.measure import regionprops
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

class App:
    def __init__(self, master):
        self.master = master
        master.title("Visualizador de Biópsias BCNB")

        self.pasta_patches = "/Users/giovanna/Documents/PUC/7º Período/PAI/Trabalhos/BCNB/paper_patches"
        self.planilha = "descritores_nucleos.xlsx"

        self.btn_frame = tk.Frame(master)
        self.btn_frame.pack(side=tk.TOP, pady=10)

        tk.Button(self.btn_frame, text="Abrir Imagem", command=self.abrir_imagem).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_frame, text="Zoom +", command=self.zoom_in).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_frame, text="Zoom -", command=self.zoom_out).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_frame, text="Converter para Cinza", command=self.converter_cinza).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_frame, text="Segmentar Núcleos", command=self.segmentar_nucleos).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_frame, text="Cálculo dos Descritores de Forma", command=self.calcular_descritores_gui).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_frame, text="Processar Todos os Pacientes", command=self.processar_todos_pacientes).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_frame, text="Executar XGBoost", command=self.executar_xgboost).pack(side=tk.LEFT, padx=5)

        self.canvas = tk.Canvas(master, width=1600, height=600)
        self.canvas.pack()

        self.descritor_output = tk.Text(master, height=10, width=160)
        self.descritor_output.pack(pady=10)

        self.zoom_factor = 1.0
        self.image_path = None
        self.original_image = None
        self.gray_image = None
        self.segmented_image = None

    def abrir_imagem(self):
        caminho = filedialog.askopenfilename(filetypes=[("Imagens", "*.png *.jpg *.jpeg")])
        if caminho:
            self.image_path = caminho
            self.original_image = Image.open(caminho)
            self.zoom_factor = 1.0
            self.gray_image = None
            self.segmented_image = None
            self.atualizar_canvas()

    def zoom_in(self):
        self.zoom_factor *= 1.2
        self.atualizar_canvas()

    def zoom_out(self):
        self.zoom_factor /= 1.2
        self.atualizar_canvas()

    def converter_cinza(self):
        if self.original_image:
            resized = self.original_image.resize(
                (int(self.original_image.width * self.zoom_factor),
                 int(self.original_image.height * self.zoom_factor))
            )
            self.gray_image = ImageOps.grayscale(resized)
            self.atualizar_canvas()

    def segmentar_nucleos(self):
        if not self.gray_image:
            return

        img_gray = np.array(self.gray_image)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_eq = clahe.apply(img_gray)

        blur = cv2.GaussianBlur(img_eq, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        self.segmented_image = Image.fromarray(thresh)
        self.atualizar_canvas()

    def atualizar_canvas(self):
        self.canvas.delete("all")
        imgs = []
        if self.original_image:
            orig = self.original_image.resize(
                (int(self.original_image.width * self.zoom_factor),
                int(self.original_image.height * self.zoom_factor))
            )
            imgs.append(orig.convert("RGB"))
        if self.gray_image:
            imgs.append(self.gray_image.convert("RGB"))
        if self.segmented_image:
            imgs.append(self.segmented_image.convert("RGB"))

        if imgs:
            total_width = sum(img.width for img in imgs) + (len(imgs) - 1) * 10
            max_height = max(img.height for img in imgs)

            # Corrige a cor de fundo para igualar a do canvas
            bg_color = self.canvas.cget("background")
            self.master.update_idletasks()
            bg_color_rgb = self.master.winfo_rgb(bg_color)
            bg_color_rgb = tuple([c // 256 for c in bg_color_rgb])

            combined = Image.new("RGB", (total_width, max_height), bg_color_rgb)
            x_offset = 0
            for img in imgs:
                combined.paste(img, (x_offset, 0))
                x_offset += img.width + 10

            self.tkimage = ImageTk.PhotoImage(combined)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tkimage)

    def atualizar_canvas(self):
        self.canvas.delete("all")
        imgs = []
        if self.original_image:
            orig = self.original_image.resize(
                (int(self.original_image.width * self.zoom_factor),
                int(self.original_image.height * self.zoom_factor))
            )
            imgs.append(orig.convert("RGB"))
        if self.gray_image:
            imgs.append(self.gray_image.convert("RGB"))
        if self.segmented_image:
            imgs.append(self.segmented_image.convert("RGB"))

        if imgs:
            total_width = sum(img.width for img in imgs) + (len(imgs) - 1) * 10
            max_height = max(img.height for img in imgs)

            # Corrige a cor de fundo para igualar a do canvas
            bg_color = self.canvas.cget("background")
            self.master.update_idletasks()
            bg_color_rgb = self.master.winfo_rgb(bg_color)
            bg_color_rgb = tuple([c // 256 for c in bg_color_rgb])

            combined = Image.new("RGB", (total_width, max_height), bg_color_rgb)
            x_offset = 0
            for img in imgs:
                combined.paste(img, (x_offset, 0))
                x_offset += img.width + 10

            self.tkimage = ImageTk.PhotoImage(combined)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tkimage)

    def calcular_descritores(self, mask):
        label_img = measure.label(mask, connectivity=2)
        props = regionprops(label_img)

        areas = []
        circularidades = []
        excentricidades = []
        razoes_distancia_raio = []
        centros = []

        for region in props:
            if region.area < 10:
                continue
            area = region.area
            perimetro = region.perimeter if region.perimeter > 0 else 1
            circularidade = (4 * math.pi * area) / (perimetro ** 2)
            excentricidade = region.eccentricity
            centro = region.centroid
            centros.append(centro)
            areas.append(area)
            circularidades.append(circularidade)
            excentricidades.append(excentricidade)

        centros = np.array(centros)
        for i, centro_i in enumerate(centros):
            outras = [distance.euclidean(centro_i, centro_j) for j, centro_j in enumerate(centros) if j != i]
            if not outras:
                continue
            menor = min(outras)
            raio = np.sqrt(areas[i] / math.pi)
            if raio > 0:
                razoes_distancia_raio.append(menor / raio)

        return {
            "Área média": np.mean(areas) if areas else 0,
            "Área desvio padrão": np.std(areas) if areas else 0,
            "Circularidade média": np.mean(circularidades) if circularidades else 0,
            "Circularidade desvio padrão": np.std(circularidades) if circularidades else 0,
            "Excentricidade média": np.mean(excentricidades) if excentricidades else 0,
            "Excentricidade desvio padrão": np.std(excentricidades) if excentricidades else 0,
            "Dist/raio média": np.mean(razoes_distancia_raio) if razoes_distancia_raio else 0,
            "Dist/raio desvio padrão": np.std(razoes_distancia_raio) if razoes_distancia_raio else 0
        }

    def calcular_descritores_gui(self):
        if not self.segmented_image or not self.image_path:
            return

        mascara_binaria = np.array(self.segmented_image.convert("L"))
        _, mascara_binaria = cv2.threshold(mascara_binaria, 127, 1, cv2.THRESH_BINARY)

        resultado = self.calcular_descritores(mascara_binaria)

        self.descritor_output.delete(1.0, tk.END)
        for chave, valor in resultado.items():
            self.descritor_output.insert(tk.END, f"{chave}: {valor:.4f}\n")

    def processar_todos_pacientes(self):
        dados = []
        for paciente_id in sorted([p for p in os.listdir(self.pasta_patches) if p.isdigit()], key=lambda x: int(x)):
            pasta_paciente = os.path.join(self.pasta_patches, paciente_id)
            if not os.path.isdir(pasta_paciente):
                continue

            melhor_patch = None
            max_nucleos = 0
            melhor_mask = None

            for arquivo in os.listdir(pasta_paciente):
                if not arquivo.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                caminho_img = os.path.join(pasta_paciente, arquivo)
                try:
                    img = cv2.imread(caminho_img)
                    if img is None:
                        raise ValueError("Imagem não pôde ser carregada.")
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    kernel = np.ones((3, 3), np.uint8)
                    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
                    sure_bg = cv2.dilate(opening, kernel, iterations=3)
                    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
                    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
                    sure_fg = np.uint8(sure_fg)
                    unknown = cv2.subtract(sure_bg, sure_fg)
                    _, markers = cv2.connectedComponents(sure_fg)
                    markers = markers + 1
                    markers[unknown == 255] = 0
                    cv2.watershed(img, markers)
                    num_nucleos = len(np.unique(markers)) - 2
                    if num_nucleos > max_nucleos:
                        max_nucleos = num_nucleos
                        melhor_patch = arquivo
                        melhor_mask = (markers > 1).astype(np.uint8)
                except Exception as e:
                    print(f"Erro ao processar imagem {arquivo} do paciente {paciente_id}: {e}")

            if melhor_mask is not None:
                descritores = self.calcular_descritores(melhor_mask)
                dados.append({
                    "ID Paciente": int(paciente_id),
                    "Nome do Arquivo": melhor_patch,
                    **descritores
                })
                print(f"Paciente {paciente_id} processado com sucesso com {max_nucleos} núcleos.")
            else:
                print(f"Aviso: Nenhum patch válido encontrado para o paciente {paciente_id}.")

        if dados:
            df_descritores = pd.DataFrame(dados)
            df_descritores = df_descritores.sort_values(by="ID Paciente").reset_index(drop=True)

            try:
                # Lê a planilha clínica
                df_clinico = pd.read_excel("patient-clinical-data.xlsx")

                # Assume que a 1ª coluna é o ID do paciente e que há uma coluna "ALN status"
                id_col = df_clinico.columns[0]  # primeira coluna = ID
                if "ALN status" not in df_clinico.columns:
                    print("Erro: coluna 'ALN status' não encontrada na planilha patient-clinical-data.xlsx.")
                else:
                    # Cria dicionário para busca eficiente por ID
                    mapa_aln = df_clinico.set_index(id_col)["ALN status"].to_dict()

                    # Adiciona coluna "ALN status" usando o ID Paciente
                    df_descritores["ALN status"] = df_descritores["ID Paciente"].map(mapa_aln)

                    print("Coluna 'ALN status' incorporada com sucesso usando os IDs.")

            except Exception as e:
                print(f"Erro ao tentar abrir ou processar a planilha patient-clinical-data.xlsx: {e}")

            df_descritores.to_excel(self.planilha, index=False)
            print(f"Planilha {self.planilha} salva com sucesso.")
        else:
            print("Nenhum dado foi processado.")


    def executar_xgboost(self):
        try:
            # Caminho fixo para a pasta com os arquivos de IDs
            caminho_ids = "/Users/giovanna/Documents/PUC/7º Período/PAI/Trabalhos/BCNB/dataset-splitting"

            # Lê planilha de descritores
            df = pd.read_excel(self.planilha)
            df = df[df['ALN status'].notnull()].copy()

            # Mapeia classes
            mapa_classes = {"N0": 0, "N+(1-2)": 1, "N+(>2)": 2}
            df['Classe'] = df['ALN status'].map(mapa_classes)
            df = df[df['Classe'].notnull()].copy()

            # Lê os arquivos de IDs
            def carregar_ids(arquivo):
                with open(arquivo, 'r') as f:
                    return set(int(line.strip()) for line in f if line.strip().isdigit())

            ids_train = carregar_ids(os.path.join(caminho_ids, "train_id.txt"))
            ids_valid = carregar_ids(os.path.join(caminho_ids, "val_id.txt"))
            ids_test  = carregar_ids(os.path.join(caminho_ids, "test_id.txt"))
            ids_treino_total = ids_train.union(ids_valid)

            # Divide os conjuntos
            df_train = df[df['ID Paciente'].isin(ids_treino_total)]
            df_test = df[df['ID Paciente'].isin(ids_test)]

            # Verifica cobertura das classes
            print("Distribuição de classes no conjunto de TESTE:")
            print(df_test['Classe'].value_counts())
            print("IDs de pacientes no teste:", df_test['ID Paciente'].tolist())

            # Seleciona atributos e rótulos
            colunas_descritoras = [col for col in df.columns if "média" in col or "desvio" in col]
            X_train = df_train[colunas_descritoras].values
            y_train = df_train['Classe'].values
            X_test = df_test[colunas_descritoras].values
            y_test = df_test['Classe'].values

            # Treinamento com XGBoost
            from xgboost import XGBClassifier
            modelo = XGBClassifier(objective="multi:softmax", num_class=3, eval_metric='mlogloss')
            modelo.fit(X_train, y_train)

            y_pred = modelo.predict(X_test)

            print("\n=== Diagnóstico por paciente no conjunto de TESTE ===")
            for i in range(len(df_test)):
                id_paciente = df_test.iloc[i]["ID Paciente"]
                classe_real = int(y_test[i])
                classe_prevista = int(y_pred[i])
                print(f"Paciente {id_paciente} | Real: {classe_real} | Previsto: {classe_prevista}")

            # Debug: ver quais classes foram previstas e estavam no teste
            import numpy as np
            print("Valores únicos em y_test:", np.unique(y_test))
            print("Valores únicos em y_pred:", np.unique(y_pred))

            from sklearn.metrics import classification_report, confusion_matrix
            import matplotlib.pyplot as plt
            import seaborn as sns

            classes_reais = list(mapa_classes.values())
            nomes_classes = list(mapa_classes.keys())

            relatorio = classification_report(
                y_test,
                y_pred,
                labels=classes_reais,
                target_names=nomes_classes,
                zero_division=0
            )

            matriz = confusion_matrix(y_test, y_pred, labels=classes_reais)

            # Exibe no terminal e interface
            print("Relatório de Classificação:\n", relatorio)
            self.descritor_output.delete(1.0, tk.END)
            self.descritor_output.insert(tk.END, relatorio)

            plt.figure(figsize=(6, 5))
            sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues',
                        xticklabels=nomes_classes, yticklabels=nomes_classes)
            plt.xlabel('Predito')
            plt.ylabel('Real')
            plt.title('Matriz de Confusão - XGBoost')
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Erro ao executar XGBoost: {e}")
            self.descritor_output.delete(1.0, tk.END)
            self.descritor_output.insert(tk.END, f"Erro ao executar XGBoost: {e}")



if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()