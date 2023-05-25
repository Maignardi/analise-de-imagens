import cv2
import os
import matplotlib.pyplot as plt
cascade_path = 'haarcascade_eye.xml'
cascade = cv2.CascadeClassifier(cascade_path)
def detectar_objetos(imagem):
    # Converter a imagem para escala de cinza
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    # Realizar a detecção de objetos
    objetos = cascade.detectMultiScale(imagem_cinza, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Verificar se houve detecção de objetos
    if len(objetos) > 0:
        return True
    else:
        return False
# Diretório contendo as imagens do dataset
diretorio_dataset = "images/6.jpg"
imagens_dataset = os.listdir(diretorio_dataset)
# Variáveis para contar sucessos e falhas
sucessos = 0
falhas = 0
# Iterar sobre as imagens do dataset
for imagem_nome in imagens_dataset:
    # Caminho completo para a imagem
    imagem_path = os.path.join(diretorio_dataset, imagem_nome)
    # Carregar a imagem
    imagem = cv2.imread(imagem_path)
    # Verificar se a imagem foi carregada corretamente
    if imagem is None:
        print(f"Falha ao carregar a imagem: {imagem_nome}")
        falhas += 1
        continue
    # Chamar a função para detectar objetos na imagem
    if detectar_objetos(imagem):
        print(f"Sucesso: {imagem_nome}")
        sucessos += 1
    else:
        print(f"Falha: {imagem_nome}")
        falhas += 1
# Calcular a porcentagem de acertos e erros
total_imagens = len(imagens_dataset)
porcentagem_acertos = (sucessos / total_imagens) * 100
porcentagem_erros = (falhas / total_imagens) * 100
# Imprimir o resultado
print(f"Total de imagens: {total_imagens}")
print(f"Sucessos: {sucessos} ({porcentagem_acertos:.2f}%)")
print(f"Falhas: {falhas} ({porcentagem_erros:.2f}%)")
# Plotar o gráfico
labels = ['Sucessos', 'Falhas']
porcentagens = [porcentagem_acertos, porcentagem_erros]
plt.bar(labels, porcentagens)
plt.xlabel('Resultado')
plt.ylabel('Porcentagem')
plt.title('Detecção de objetos')
plt.show()
