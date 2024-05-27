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

# Função para obter uma imagem com os polígonos desenhados
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
    path_folder = '/home/anderson/Documentos/Roney/Pré-analisados/'

    orto_files = os.listdir(path_folder)
    print("Total de Ortos as serem processados: ", len(orto_files))
    print("Ortos a serem processados:")
    print(orto_files)
    cont = 1
    print("---------------------------- Iniciando processamento ---------------------------- ")
    for file in orto_files:
        print("-------- Orto número: ", cont, "----------")
        # Caminho da ortofoto
        path_orto = os.listdir(os.path.join(path_folder,file))
        orto_file =  os.path.join(path_folder,file,[orto_file for orto_file in path_orto if orto_file.endswith('.tif')][0])
        print("Orto lido:")
        print(orto_file)
        filename_orto = os.path.basename(orto_file)[:-4]

        # Caminho das máscaras
        path_daninha = os.listdir(os.path.join(path_folder,file,[i for i in path_orto if 'daninhas' in i][0]))

        try:
            path_mask = os.path.join(path_folder,file,[i for i in path_orto if 'daninhas' in i][0],[i for i in path_daninha if 'mask_graminea.shp' in i][0])
        except:
            print("Erro a encontrar a mask_name.shp")
            continue
        
        print("Mask lida:")
        print(path_mask)


        # Caminho da rotulação
        try:
            path_gt = os.path.join(path_folder,file,[i for i in path_orto if 'daninhas' in i][0],[i for i in path_daninha if 'Classe_graminea.shp' in i][0])
        except:
            print("Erro a encontrar a Classe_name.shp")
            continue

        cont = cont + 1
        # Caminho para salvar os recortes
        pathout = '/home/anderson/Documentos/Roney/Dataset/Dataset/dataset_graminea/'
        pathout_rgb = os.path.join(pathout, 'rgb')
        pathout_lbl = os.path.join(pathout, 'labels')

        os.makedirs(pathout_rgb, exist_ok=True)
        os.makedirs(pathout_lbl, exist_ok=True)

        # Abrindo a ortofoto
        dataset = rasterio.open(orto_file)

        # Pegando o sistema de coordenadas (CRS) da ortofoto. Esse sistema funciona como uma maneira de representar a lat e lon
        crs = dataset.crs

        print(crs)

        # Obtendo as dimensões da ortofoto
        width, height = dataset.height, dataset.width # Existe uma inversão entre a biblioteca rasterio e outras de processamento de imagens
        print(f'Dimensões da ortofoto: {(dataset.width, dataset.height)}')

        # Abrindo o shapefile de máscara
        gpd_mask = gpd.read_file(path_mask)
        gpd_mask = gpd_mask.explode(ignore_index=True)

        # Colocando o shapefile no mesmo sistema de coordenadas da ortofoto
        gpd_mask = gpd_mask.to_crs(crs)

        # Abrindo a imagem que representa as máscaras
        img_mask = get_img_from_shapefile(gpd_mask, width, height, dataset.index)

        # Abrindo o shapefile de anotações
        gpd_gt = gpd.read_file(path_gt)
        gpd_gt = gpd_gt.explode(ignore_index=True)

        # Colocando o shapefile no mesmo sistema de coordenadas da ortofoto
        gpd_gt = gpd_gt.to_crs(crs)

        # Abrindo a imagem que representa as máscaras
        img_gt = get_img_from_shapefile(gpd_gt, width, height, dataset.index)

        # Percorredo toda a imagem
        for x in tqdm(range(0, width-t, step)):
            for y in range(0, height-t, step):
                # Recortando um patch na máscara e no gt
                patch_mask = img_mask[x:x+t, y:y+t]
                patch_gt = img_gt[x:x+t, y:y+t]
                
                # Se não tiver máscara, pulamos esse patch
                if np.sum(patch_mask) == 0: continue
                
                # Salvando a imagem colorida
                patch_img = get_image_patch(dataset, x, y, t, t)
                patch_img = Image.fromarray(patch_img)
                patch_img.save(os.path.join(pathout_rgb, f'{filename_orto}_{x}_{y}.jpg'))
                
                # Salvando o patch
                patch_gt[patch_mask == 0] = 255 # Colocando 255 onde não foi rotulado. Durante o treino, iremos ignorar esses pixels
                patch_gt = Image.fromarray(patch_gt)
                # Criando uma paleta de cores para visualizar quando abrir a imagem no PC. Caso contrário, só veríamos uma imagem preta, porque 0: fundo e 1: classe.
                palette = [255]*(256*3)
                palette[0:6] = [0,0,0, 255,0,0]
                patch_gt.putpalette(palette)
                patch_gt.save(os.path.join(pathout_lbl, f'{filename_orto}_{x}_{y}.png'))

main()
