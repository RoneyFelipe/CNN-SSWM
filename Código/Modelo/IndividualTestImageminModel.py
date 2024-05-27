
from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv
import os
import shutil
import matplotlib.pyplot as plt
from PIL import Image

def main():
    
    config_file = '/home/anderson/mmsegmentation/configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512_colonhao.py'
    checkpoint_file = '/home/anderson/mmsegmentation/work_dirs/segformer_mit-b0_8xb2-160k_ade20k-512x512/iter_80000.pth'

    # build the model from a config file and a checkpoint file
    model = init_model(config_file, checkpoint_file, device='cuda:0')

    folder_split_dataset = '/home/anderson/Documentos/Roney/Dataset/split_dataset/colonhao'
    path_split_dataset = os.listdir(folder_split_dataset)

    folder_test_imagens = os.listdir(os.path.join(folder_split_dataset,[i for i in path_split_dataset if 'test' in i][0]))
    path_test_imagens = os.listdir(os.path.join(folder_split_dataset,[i for i in path_split_dataset if 'test' in i][0], [i for i in folder_test_imagens if 'rgb' in i][0]))

    print('Testando imagens da pasta: ',os.path.join(folder_split_dataset,[i for i in path_split_dataset if 'test' in i][0], [i for i in folder_test_imagens if 'rgb' in i][0]))
    for i in range(len(path_test_imagens)):
        result = inference_model(model, os.path.join(folder_split_dataset,[i for i in path_split_dataset if 'test' in i][0], [i for i in folder_test_imagens if 'rgb' in i][0],path_test_imagens[i]))
        show_result_pyplot(model, os.path.join(folder_split_dataset,[i for i in path_split_dataset if 'test' in i][0], [i for i in folder_test_imagens if 'rgb' in i][0],path_test_imagens[i]), result, show=False, out_file='/home/anderson/Documentos/Roney/Dataset/Inferences_model/colonhao/'+path_test_imagens[i])
        print('Imagem testada: ', path_test_imagens[i])
    print('Fim da inferencia')
   
    
    
main()