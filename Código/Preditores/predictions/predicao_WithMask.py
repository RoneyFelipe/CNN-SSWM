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
from mmseg.apis import MMSegInferencer

def predictionModel(model, path_img):

    result = inference_model(model, path_img)

    result = show_result_pyplot(model, path_img, result, show=0)

    return result

def get_image_patch(dataset, x, y, w, h):   

    img = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(3):
        band = dataset.read(i+1, window=Window(y, x, w, h))
        nw, nh = band.shape[0], band.shape[1]
        if nw == 0 or nh == 0:
            return None
        img[0:nw, 0:nh, i] = band

    return img


def get_img_from_shapefile(gpd, width, height, func_latlon_xy):
    image = np.zeros((width, height), dtype=np.uint8)
    
    for idx, geometry in enumerate(gpd.geometry):
        if geometry is None: continue
        pts = np.array([func_latlon_xy(point[0], point[1])
                       for point in geometry.exterior.coords[:]], np.int32)[:, ::-1]
        
        cv2.fillPoly(image, [pts], 1)

        for interior in geometry.interiors:
            pts = np.array([func_latlon_xy(point[0], point[1])
                            for point in interior.coords[:]], np.int32)[:, ::-1]
            
            cv2.fillPoly(image, [pts], 0)
    return image


def main():
    t = 256
    step = t // 2
    path_folder = '/media/anderson/Jaiza/Roney/Ortos/Rotulos_np/'

    config_file = '/home/anderson/mmsegmentation/configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512_mamona.py'
    checkpoint_file = '/home/anderson/mmsegmentation/work_dirs/segformer_mit-b0_8xb2-160k_ade20k-512x512_mamona/iter_20000.pth'
    orto_files = os.listdir(path_folder)
    orto_files.sort()
    cont = 1

    model = init_model(config_file, checkpoint_file, device='cuda:0')

    print("Total de Ortos as serem processados: ", len(orto_files))
    print("Ortos a serem processados:")

    print(orto_files)
    
    print("---------------------------- Iniciando processamento ---------------------------- ")

    for file in orto_files:
        print("-------- Orto número: ", cont, "----------")
        # Caminho da ortofoto
        path_orto = os.listdir(os.path.join(path_folder,file))
        orto_file =  os.path.join(path_folder,file,[orto_file for orto_file in path_orto if orto_file.endswith('.tif')][0])
        talhao_file =  os.path.join(path_folder,file,[orto_file for orto_file in path_orto if orto_file.endswith('.shp')][1])

        print("Orto lido:")
        print(orto_file)
        print("Talhão lido:")
        print(talhao_file)
        filename_orto = os.path.basename(orto_file)[:-4]

        cont = cont + 1

        # Abrindo a ortofoto
        dataset = rasterio.open(orto_file)

        # Pegando o sistema de coordenadas (CRS) da ortofoto. Esse sistema funciona como uma maneira de representar a lat e lon
        crs = dataset.crs
        print(crs)

        # Obtendo as dimensões da ortofoto
        height ,width = dataset.height, dataset.width # Existe uma inversão entre a biblioteca rasterio e outras de processamento de imagens
        print(f'Dimensões da ortofoto: {(dataset.width, dataset.height)}')

        # Abrindo o shapefile de máscara
        gpd_mask = gpd.read_file(talhao_file)
        gpd_mask = gpd_mask.explode(ignore_index=True)

        # Colocando o shapefile no mesmo sistema de coordenadas da ortofoto
        gpd_mask = gpd_mask.to_crs(crs)

        # Abrindo a imagem que representa as máscaras
        img_mask = get_img_from_shapefile(gpd_mask, width, height, dataset.index)
        print(type(img_mask))
        print(type(dataset))
        print(img_mask.shape)

        # Percorrendo toda a imagem
        prediction = np.zeros((height, width, 3), dtype=np.uint8)
        
        for x in tqdm(range(0, height-t, step)):
            for y in range(0, width-t, step):

                if x+t >= height:
                    x = y-t

                if y+t >= width:
                    y = y-t
              
                patch_mask = img_mask[x:x+t, y:y+t]
                #if np.sum(patch_mask) == 0: continue
                
                patch_img = get_image_patch(dataset, x, y, t, t)
                pixels_brancos = np.logical_and(np.logical_and(patch_img[:,:,0]==255, patch_img[:,:,1]==255), patch_img[:,:,2]==255)
                perc_pixels_brancos = np.sum(pixels_brancos) / (t*t)

                if perc_pixels_brancos >= 0.95:
                    continue

                pred = predictionModel(model, patch_img)
                prediction[x:x+t, y:y+t] = pred
               
        print(x,y, height, width)
        prediction = Image.fromarray(prediction)
        prediction.save(os.path.join('/home/anderson/Documentos/Roney/ortosaida/DecimaVersao/', f'{filename_orto}.jpg'))


main()