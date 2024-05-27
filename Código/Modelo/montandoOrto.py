import cv2
import rasterio as rasterio
from rasterio.windows import Window
import geopandas as gpd
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from shapely import Polygon
from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv
import os
import shutil
import matplotlib.pyplot as plt
from PIL import Image


def main():
    folder_imagens = '/home/anderson/Documentos/Roney/Código/Modelo/testandoRecortes'
    path_imagens = os.listdir(os.path.join(folder_imagens))
    print(path_imagens)
    orto_file = '/media/anderson/Jaiza/Roney/Rotulos/Pré-analisados/orto_ebee_SCAND-SOCAMAI290-20136z1_17-10-2023_17-10-2023_17-10-2023/orto_ebee_SCAND-SOCAMAI290-20136z1_17-10-2023_17-10-2023_17-10-2023.tif'
    ortoMontado = []

    for img in path_imagens:
        imagensOriginal, coordenadas= img, img.split('_')[-2:]
        x , y = int(coordenadas[0]) , int(coordenadas[1][:-4])

        for i in path_imagens: 
            newImagensOriginal, newCoordenadas= img, img.split('_')[-2:]
            newX , newY = int(coordenadas[0]) , int(coordenadas[1][:-4])
            if x == newX:
                ortoMontado = np.hstack(newImagensOriginal)

        
        
    imagem_pil = Image.fromarray(ortoMontado)

    # Salvar a imagem
    imagem_pil.save('ortoMontado2.tif')

    # Exibir a imagem
    imagem_pil.show()

main()
