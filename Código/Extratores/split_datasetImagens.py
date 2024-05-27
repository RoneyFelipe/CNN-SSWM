import os
import shutil

def unir_vetor(vetor):
    return ''.join(map(str, vetor))

def remover_repetidos(lista):
    return list(set(lista))


def main():
    folder_dataset = '/home/anderson/Documentos/Roney/Dataset/Dataset/datasetMamona_Revisado/classificado_destino_dataset_mamona/dataset_mamona/destino'

    path_dataset = os.listdir(folder_dataset)
    path_labels = os.listdir(os.path.join(folder_dataset,[i for i in path_dataset if 'label' in i][0]))
    ortos = []
    original_names = []

    for patch in path_labels:
        names_patch = patch.split('_')
        original_names.append(names_patch)
        names_patch = (names_patch[:-2])
        names_patch = unir_vetor(names_patch)
        ortos.append(names_patch)
        print(names_patch)
        

    ortos = remover_repetidos(ortos)
    n_train = int(len(ortos)*0.7)
    n_val = int(len(ortos)*0.25)
    
    ortos_train = ortos[:n_train]
    ortos_val = ortos[n_train:n_train+n_val]
    ortos_test = ortos[n_train+n_val:]
    
    print('-------------')
    
    cont_train = 0
    cont_val = 0
    cont_test = 0

    for i in range(len(ortos_train)):

        for j in range(len(original_names)):
            print('-----------')
            print(unir_vetor((original_names[j][:-2])))
            if ortos_train[i] == unir_vetor((original_names[j][:-2])):
                cont_train+=1
                shutil.copy(folder_dataset+'/labels/'+path_labels[j], '/home/anderson/Documentos/Roney/Dataset/split_dataset/mamonaRevisado/train/labels/')
                shutil.copy(folder_dataset+'/rgb/'+path_labels[j][:-3]+'jpg', '/home/anderson/Documentos/Roney/Dataset/split_dataset/mamonaRevisado/train/rgb/')

    for i in range(len(ortos_val)):

        for j in range(len(original_names)):
            print('-----------')
            print(unir_vetor((original_names[j][:-2])))
            if ortos_val[i] == unir_vetor((original_names[j][:-2])):
                cont_val+=1
                shutil.copy(folder_dataset+'/labels/'+path_labels[j], '/home/anderson/Documentos/Roney/Dataset/split_dataset/mamonaRevisado/val/labels/')
                shutil.copy(folder_dataset+'/rgb/'+path_labels[j][:-3]+'jpg', '/home/anderson/Documentos/Roney/Dataset/split_dataset/mamonaRevisado/val/rgb/')

    for i in range(len(ortos_test)):

        for j in range(len(original_names)):
            print('-----------')
            print(unir_vetor((original_names[j][:-2])))
            if ortos_test[i] == unir_vetor((original_names[j][:-2])):
                print('oi')
                cont_test+=1
                shutil.copy(folder_dataset+'/labels/'+path_labels[j], '/home/anderson/Documentos/Roney/Dataset/split_dataset/mamonaRevisado/test/labels/')
                shutil.copy(folder_dataset+'/rgb/'+path_labels[j][:-3]+'jpg', '/home/anderson/Documentos/Roney/Dataset/split_dataset/mamonaRevisado/test/rgb/')

    print(len(ortos))
    
    print('percentual entre split:')
    print('Treino: ',n_train)
    print('Validação: ',n_val)
    
    print('Ortos lidos: ',ortos)
    print('Ortos de treino:',ortos_train)
    print('Ortos de validacao:',ortos_val)
    print('Ortos de test:',ortos_test)
    print('Total de imagens train:',cont_train)
    print('Total de imagens val:',cont_val)
    print('Total de imagens test:',cont_test)
    print('Total de imagens do dataset:',len(original_names))


main()