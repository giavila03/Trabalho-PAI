import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import cv2
import numpy as np
import pandas as pd
from skimage import measure
from skimage.measure import regionprops
from scipy.spatial import distance
import math
import os
import subprocess
import platform

class App:
    def __init__(self, master):
        self.master = master
        master.title("Visualizador de Biópsias BCNB")

        self.planilha = "descritores_nucleos.xlsx"

        # Frame dos botões
        self.btn_frame = tk.Frame(master)
        self.btn_frame.pack(side=tk.TOP, pady=10)

        tk.Button(self.btn_frame, text="Abrir Imagem", command=self.abrir_imagem).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_frame, text="Zoom +", command=self.zoom_in).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_frame, text="Zoom -", command=self.zoom_out).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_frame, text="Converter para Cinza", command=self.converter_cinza).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_frame, text="Segmentar Núcleos", command=self.segmentar_nucleos).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_frame, text="Cálculo dos Descritores de Forma", command=self.calcular_descritores).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_frame, text="Abrir e Salvar Planilha", command=self.salvar_e_abrir_planilha).pack(side=tk.LEFT, padx=5)

        self.canvas = tk.Canvas(master, width=1500, height=500)
        self.canvas.pack()

        self.descritor_output = tk.Text(master, height=10, width=150)
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
        img_cv = np.array(self.gray_image)
        blur = cv2.GaussianBlur(img_cv, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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
            combined = Image.new("RGB", (total_width, max_height), (220, 220, 220))
            x_offset = 0
            for img in imgs:
                combined.paste(img, (x_offset, 0))
                x_offset += img.width + 10

            self.tkimage = ImageTk.PhotoImage(combined)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tkimage)

    def calcular_descritores(self):
        if not self.segmented_image or not self.image_path:
            return

        mascara_binaria = np.array(self.segmented_image.convert("L"))
        _, mascara_binaria = cv2.threshold(mascara_binaria, 127, 1, cv2.THRESH_BINARY)

        label_img = measure.label(mascara_binaria, connectivity=2)
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

        resultado = {
            "Área média": np.mean(areas) if areas else 0,
            "Área std": np.std(areas) if areas else 0,
            "Circularidade média": np.mean(circularidades) if circularidades else 0,
            "Circularidade std": np.std(circularidades) if circularidades else 0,
            "Excentricidade média": np.mean(excentricidades) if excentricidades else 0,
            "Excentricidade std": np.std(excentricidades) if excentricidades else 0,
            "Dist/raio média": np.mean(razoes_distancia_raio) if razoes_distancia_raio else 0,
            "Dist/raio std": np.std(razoes_distancia_raio) if razoes_distancia_raio else 0
        }

        self.descritor_output.delete(1.0, tk.END)
        for chave, valor in resultado.items():
            self.descritor_output.insert(tk.END, f"{chave}: {valor:.4f}\n")

        self.salvar_descritores_em_planilha(resultado)

    def salvar_descritores_em_planilha(self, resultado):
        nome_arquivo = os.path.basename(self.image_path)
        try:
            id_paciente = int(nome_arquivo.split("_")[0])
        except:
            id_paciente = 0

        nova_linha = {
            "ID Paciente": id_paciente,
            "Nome do Arquivo": nome_arquivo,
            **resultado
        }

        if os.path.exists(self.planilha):
            df = pd.read_excel(self.planilha)
            if nome_arquivo in df["Nome do Arquivo"].values:
                return
            df = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)
        else:
            df = pd.DataFrame([nova_linha])

        df = df.sort_values(by=["ID Paciente", "Nome do Arquivo"]).reset_index(drop=True)
        df.to_excel(self.planilha, index=False)

    def salvar_e_abrir_planilha(self):
        # Cria planilha vazia se não existir
        if not os.path.exists(self.planilha):
            colunas = [
                "ID Paciente", "Nome do Arquivo", "Área média", "Área std",
                "Circularidade média", "Circularidade std",
                "Excentricidade média", "Excentricidade std",
                "Dist/raio média", "Dist/raio std"
            ]
            df_vazio = pd.DataFrame(columns=colunas)
            df_vazio.to_excel(self.planilha, index=False)

        # Abre a planilha com o programa padrão
        if platform.system() == "Windows":
            os.startfile(self.planilha)
        elif platform.system() == "Darwin":
            subprocess.call(["open", self.planilha])
        else:
            subprocess.call(["xdg-open", self.planilha])


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
