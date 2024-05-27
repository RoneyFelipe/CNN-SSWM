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


def get_image_patch(dataset, x, y, w, h):    
    img = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(3):
        band = dataset.read(i+1, window=Window(y, x, w, h))
        nw, nh = band.shape[0], band.shape[1]
        if nw == 0 or nh == 0:
            return None
        img[0:nw, 0:nh, i] = band
    
    return img


def main():
    t = 256
    step = t // 2
    orto_file =  '/home/anderson/Documentos/Roney/Código/Modelo/orto_ebee_BARRA-SOCAABR5-91814z554_05-10-2023_06-10-2023_06-10-2023.tif'
    print("Orto lido:")
    print(orto_file)

    filename_orto = os.path.basename(orto_file)[:-4]

    dataset = rasterio.open(orto_file)
    crs = dataset.crs
    print(dataset.index)
    print(crs)

    width, height = dataset.height, dataset.width
    print(f'Dimensões da ortofoto: {(dataset.width, dataset.height)}')

    for x in tqdm(range(0, width-t, step)):
        for y in range(0, height-t, step):
            print(x, y, t)
            # Salvando a imagem colorida
            patch_img = get_image_patch(dataset, x, y, t, t)
            patch_img = Image.fromarray(patch_img) 
            patch_img.save(os.path.join('/home/anderson/Documentos/Roney/Código/Modelo/recortes/', f'{filename_orto}_{x}_{y}.jpg'))

main()
