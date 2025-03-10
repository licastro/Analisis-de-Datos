# -*- coding: utf-8 -*-
"""
Trabajo Práctico 2
Grupo 7: Castro Lucia, Padulo R. Javier, Flores Leandro
Laboratorio de Datos
Análisis de imágenes MNIST-C
"""

#%% Importar bibliotecas

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split, KFold
from sklearn import tree
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

#%% Cargar datos

data_imgs = np.load('/home/Estudiante/Descargas/mnistc_images.npy')
data_labels = np.load('/home/Estudiante/Descargas/mnistc_labels.npy')

#%%

def graficar_ejemplos_por_clase(data, labels):
    # Seleccionar un índice aleatorio para cada clase
    indices = [np.random.choice(np.where(labels == i)[0]) for i in range(10)]
    
    # Crear subplots para mostrar las imágenes
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(data[indices[i]], cmap='gray')
        ax.set_title(f"Etiqueta: {i}", fontsize=14)
        ax.axis('off')
    
    plt.tight_layout()      
    plt.show()

def calcular_promedios(images, labels):
    promedios = []
    for digit in range(10):
        digit_images = images[labels == digit]
        promedio = np.mean(digit_images, axis=0)
        promedios.append(promedio)
    return np.array(promedios)

def calcular_desviacion(images, labels): 
    desviaciones = []
    for digit in range(10):
        digit_images = images[labels == digit]
        desviacion = np.std(digit_images, axis=0)
        desviaciones.append(desviacion)
    return np.array(desviaciones)

def mostrar_imagenes_promedio(promedios, color='hot'):
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))  
    axes = axes.flatten()
    for j, i in enumerate(range(10)):
        axes[j].imshow(promedios[i], cmap=color)  
        axes[j].set_title(f'Dígito {i}', fontsize=14)
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()

def mostrar_imagenes_desviaciones(desviaciones):
    # Crear la figura y los subgráficos con un espacio controlado entre ellos
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), gridspec_kw={'hspace': -0.1, 'wspace': 0.05})
    axes = axes.flatten()

    for i, desviacion in enumerate(desviaciones):
        ax = axes[i]
        im = ax.imshow(desviacion, cmap='jet')  
        ax.set_title(f'STD: {i}', fontsize=14)  
        ax.axis('off')  
    # Añadir la barra de color verticalmente
    plt.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', shrink=0.7)
    plt.show()  
    
    
def mostrar_variabilidad_desviaciones(promedios):
    desviacion_estandar_promedios = np.std(promedios, axis=0)
    plt.imshow(desviacion_estandar_promedios, cmap="jet")
    plt.title("Desviación estándar entre los promedios de los dígitos")
    plt.colorbar()
    plt.show()
    return desviacion_estandar_promedios
    
def mostrar_pixeles_mayor_variabilidad(variabilidad_entre_clases):
    percentiles = [95, 90, 85, 80, 75]
    fig, axes = plt.subplots(1, 5, figsize=(15, 8))
    axes = axes.flatten()
    for j, i in enumerate(percentiles):  
        umbral = np.percentile(variabilidad_entre_clases, i)
        pixeles_con_alta_variabilidad = variabilidad_entre_clases.squeeze() > umbral
        axes[j].imshow(pixeles_con_alta_variabilidad, cmap='gray')  
        axes[j].set_title(f'Percentil {i}')
        axes[j].axis('off')
    plt.show()
    return variabilidad_entre_clases > np.percentile(variabilidad_entre_clases, 85)
    
def mostrar_promedio_mascara_pixels(promedios, pixeles_con_alta_variabilidad):
    plt.figure(figsize=(3, 4))
    plt.imshow(pixeles_con_alta_variabilidad, cmap="gray")
    plt.title("Máscara de pìxeles con mayor variabilidad (percentil 85)")
    plt.axis('off')
    plt.show()
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 8)) 
    axes = axes.flatten()
    for i in range(10):
        promedio_mascarado = np.copy(promedios[i])
        promedio_mascarado[~pixeles_con_alta_variabilidad] = 0  
        axes[i].imshow(promedio_mascarado, cmap='hot')
        axes[i].set_title(f'Dígito {i}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

#%%
# Graficar un ejemplo aleatorio por cada clase (dígito del 0 al 9)
# Esta función selecciona un índice aleatorio de cada clase y muestra una imagen correspondiente
# en un grid de 2x5, con la etiqueta del dígito en cada imagen.

graficar_ejemplos_por_clase(data_imgs, data_labels)

#%%
# Calcular los promedios de las imágenes por clase
promedios = calcular_promedios(data_imgs, data_labels)
# Calcular la desviación estándar de las imágenes por clase
desviaciones = calcular_desviacion(data_imgs, data_labels)

# Mostrar las imágenes promedio de cada dígito (del 0 al 9)
mostrar_imagenes_promedio(promedios, color='gray')

# Mostrar las desviaciones estándar de las imágenes por clase
mostrar_imagenes_desviaciones(desviaciones)

#%% Cálculo de distancias euclidianas entre los promedios de cada clase

def calcular_distancias_euclidianas(promedios):
    n_clases = len(promedios)
    distancias = np.zeros((n_clases, n_clases))
    for i in range(n_clases):
        for j in range(n_clases):
            distancias[i, j] = euclidean(promedios[i].flatten(), promedios[j].flatten())
    return distancias

def visualizar_matriz_distancias(distancias, etiquetas):
    plt.figure(figsize=(10, 8))
    sns.heatmap(distancias, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=etiquetas, yticklabels=etiquetas)
    plt.title("Distancias Euclidianas entre Imágenes Promedio")
    plt.xlabel("Dígitos")
    plt.ylabel("Dígitos")
    plt.show()

distancias_euclidianas = calcular_distancias_euclidianas(promedios)
visualizar_matriz_distancias(distancias_euclidianas, etiquetas=[str(i) for i in range(10)])

#%%
## Detectar y mostrar outliers

def calcular_outliers(imagenes, etiquetas, clases=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    outliers = {}

    for clase in clases:
        imagenes_clase = imagenes[etiquetas == clase]
        media_clase = np.mean(imagenes_clase, axis=0)
        
        # Calcular las puntuaciones de outlier (distancia cuadrada al promedio)
        outlier_scores = [np.sum((img - media_clase)**2) for img in imagenes_clase]
        
        # Encontrar el índice de la imagen con la puntuación más alta
        outlier_idx = np.argmax(outlier_scores)
        outliers[clase] = imagenes_clase[outlier_idx]
    return outliers


def mostrar_outliers(outliers):
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))  
    axes = axes.flatten()

    for i, (clase, outlier) in enumerate(outliers.items()):
        axes[i].imshow(outlier, cmap="gray")
        axes[i].set_title(f"Outlier dígito {clase}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

outliers = calcular_outliers(data_imgs, data_labels)
mostrar_outliers(outliers)


#%%
# Calcular la desviación estándar entre los promedios de los dígitos
variabilidad_entre_clases = mostrar_variabilidad_desviaciones(promedios)

# Identificar los píxeles con alta variabilidad (percentil 85)
pixeles_con_alta_variabilidad = mostrar_pixeles_mayor_variabilidad(variabilidad_entre_clases)

# Mostrar los promedios de los dígitos con la máscara de píxeles de alta variabilidad
mostrar_promedio_mascara_pixels(promedios, pixeles_con_alta_variabilidad)

#%% CLASIFICACION MULTICLASE

## Seleccionamos los digitos para grupo 7 

dígitos_grupo = [1, 2, 3, 7, 8]
indices_filtrados = np.isin(data_labels, dígitos_grupo)
X_filtrado = data_imgs[indices_filtrados]
y_filtrado = data_labels[indices_filtrados]

X_dev, X_holdout, y_dev, y_holdout = train_test_split(X_filtrado, y_filtrado, test_size=0.3, random_state=42)

X_dev = X_dev.reshape(X_dev.shape[0], -1)
X_holdout = X_holdout.reshape(X_holdout.shape[0], -1)

#%%  ÁRBOL DE DECISIÓN ENTROPY

alturas = list(range(1, 29))   
n_splits = 5

kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

resultados = np.zeros((n_splits, len(alturas)))
tiempos = np.zeros((n_splits, len(alturas)))  # Arreglo para almacenar los tiempos

for i, (train_index, test_index) in enumerate(kf.split(X_dev)):
    kf_X_train, kf_X_test = X_dev[train_index], X_dev[test_index]
    kf_y_train, kf_y_test = y_dev[train_index], y_dev[test_index]
    
    for j, hmax in enumerate(alturas):
        arbol = tree.DecisionTreeClassifier(max_depth=hmax, criterion='entropy', random_state=42)
        
        start_time = time.time()  # Inicia el temporizador
        arbol.fit(kf_X_train, kf_y_train)
        pred = arbol.predict(kf_X_test)
        end_time = time.time()  # Termina el temporizador
        
        # Guardar el tiempo y la exactitud
        tiempos[i, j] = end_time - start_time
        accuracy = metrics.accuracy_score(kf_y_test, pred)
        resultados[i, j] = accuracy

scores_promedio = resultados.mean(axis=0)
tiempos_promedio = tiempos.mean(axis=0)

for i, hmax in enumerate(alturas):
    print(f"  Profundidad {hmax}: Exactitud promedio = {scores_promedio[i]:.4f}, Tiempo promedio = {tiempos_promedio[i]:.4f} segundos")

profundidad_mayor_exactitud = alturas[np.argmax(scores_promedio)]
print(f"  Profundidad con mayor exactitud: {profundidad_mayor_exactitud}\n")

#%%
# Primer gráfico: Todo el rango
fig, ax1 = plt.subplots(figsize=(6, 4))

# Exactitud en el eje izquierdo
color = 'tab:red'
ax1.set_xlabel("Profundidad")
ax1.set_ylabel("Exactitud", color=color)
ax1.plot(alturas, scores_promedio, '-o', color=color, label="Exactitud")
ax1.tick_params(axis='y', labelcolor=color)

# Tiempo en el eje derecho
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel("Tiempo (segundos)", color=color)
ax2.plot(alturas, tiempos_promedio, '-o', color=color, label="Tiempo")
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()

# Segundo gráfico: Rango ampliado
fig, ax1 = plt.subplots(figsize=(6, 4))

# Exactitud en el eje izquierdo
color = 'tab:red'
ax1.set_xlabel("Profundidad")
ax1.set_ylabel("Exactitud", color=color)
ax1.plot(alturas, scores_promedio, '-o', color=color, label="Exactitud")
ax1.tick_params(axis='y', labelcolor=color)

# Tiempo en el eje derecho
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel("Tiempo (segundos)", color=color)
ax2.plot(alturas, tiempos_promedio, '-o', color=color, label="Tiempo")
ax2.tick_params(axis='y', labelcolor=color)

# Ajustar los límites
ax1.set_xlim(6, 15)  # Limitar eje x
ax1.set_ylim(0.82, 0.88)  # Limitar eje y para exactitud

fig.tight_layout()
plt.show()

#%%

exactitudes_promedio = []
rango = [50, 55, 60, 65, 70, 75, 80]

for random_state in rango:
    print(f"Calculando con random_state = {random_state}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    resultados = []

    for train_index, test_index in kf.split(X_dev):
        kf_X_train, kf_X_test = X_dev[train_index], X_dev[test_index]
        kf_y_train, kf_y_test = y_dev[train_index], y_dev[test_index]

        arbol = tree.DecisionTreeClassifier(max_depth=profundidad_mayor_exactitud, criterion='entropy', random_state=42)
        arbol.fit(kf_X_train, kf_y_train)
        pred = arbol.predict(kf_X_test)
        
        accuracy = metrics.accuracy_score(kf_y_test, pred)
        resultados.append(accuracy)

    scores_promedio = np.mean(resultados)
    exactitudes_promedio.append(scores_promedio)
    print(f"Profundidad {profundidad_mayor_exactitud}: Exactitud promedio = {scores_promedio:.4f}")
#%%
plt.ylim(0.8,0.9)
plt.plot(rango, exactitudes_promedio, '-o')
plt.xlabel("Random State")
plt.ylabel("Exactitud Promedio")
plt.title("Exactitud Promedio con Diferentes Semillas Aleatorias")
plt.show()

#%%

## III. COMPARANDO ARBOLES GINI Y ENTROPY VARIANDO K FOLD Y PROFUNDIDADES

alturas = list(range(1, profundidad_mayor_exactitud+1)) 

criterios = ['entropy', 'gini']
mejor_criterio = None
mejor_prof = None
mejor_score = 0
mejor_n_splits = None
scores_list=[]

for criterio in criterios:
    for n_splits in [3, 5, 10]:  
        print(f"Evaluando con {n_splits} pliegues y criterio '{criterio}':")
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        resultados = np.zeros((n_splits, len(alturas)))
        
        for i, (train_index, test_index) in enumerate(kf.split(X_dev)):
            kf_X_train, kf_X_test = X_dev[train_index], X_dev[test_index]
            kf_y_train, kf_y_test = y_dev[train_index], y_dev[test_index]
            
            for j, hmax in enumerate(alturas):
                arbol = tree.DecisionTreeClassifier(max_depth=hmax, criterion=criterio, random_state=42)
                arbol.fit(kf_X_train, kf_y_train)
                pred = arbol.predict(kf_X_test)
                
                accuracy = metrics.accuracy_score(kf_y_test, pred)
                resultados[i, j] = accuracy
        
        scores_promedio = resultados.mean(axis=0)
        scores_list.append(scores_promedio)
        
        for i, hmax in enumerate(alturas):
            print(f"  Profundidad {hmax}: Exactitud promedio = {scores_promedio[i]:.4f}")
        
        profundidad_mayor_exactitud = alturas[np.argmax(scores_promedio)]
        mejor_score_temp = np.max(scores_promedio)
        
        print(f"  Mejor profundidad: {profundidad_mayor_exactitud}")
        
        if mejor_score_temp > mejor_score:
            mejor_score = mejor_score_temp
            mejor_criterio = criterio
            mejor_prof = profundidad_mayor_exactitud
            mejor_n_splits = n_splits
        
        print(f"  Exactitud en el conjunto eval con {criterio}: {mejor_score_temp:.4f}\n")

plt.xlim(3,10)
plt.ylim(0.7,1)
plt.plot(alturas,scores_list[0], 'go-', label="3 pliegues")
plt.plot(alturas,scores_list[1], 'ro-', label="5 pliegues")
plt.plot(alturas,scores_list[2], 'bo-', label="10 pliegues")
plt.legend()

plt.xlabel("Profundidad")
plt.ylabel("Exactitud")
plt.title('Criterio Entropy')
plt.show()

plt.xlim(3,10)
plt.ylim(.7, 1)

plt.plot(alturas,scores_list[3], 'go-',label="3 pliegues")
plt.plot(alturas,scores_list[4], 'ro-',label="5 pliegues")
plt.plot(alturas,scores_list[5], 'bo-',label="10 pliegues")
plt.legend()

plt.xlabel("Profundidad")
plt.ylabel("Exactitud")
plt.title('Criterio Gini')
plt.show()

print(f"\nEl mejor modelo global fue con el criterio '{mejor_criterio}', profundidad {mejor_prof}, y {mejor_n_splits} pliegues.")


#%%

## IV.

feature_names = [f"Pixel {i}" for i in range(784)]

arbol_elegido = tree.DecisionTreeClassifier(max_depth=7, criterion='gini', random_state=42)
arbol_elegido.fit(X_dev, y_dev)

y_pred_eval = arbol_elegido.predict(X_holdout)
score_arbol_elegido_eval = metrics.accuracy_score(y_holdout, y_pred_eval)

print(f"Exactitud en el conjunto de evaluación: {score_arbol_elegido_eval:.4f}")

plt.figure(figsize=(20, 20))
tree.plot_tree(arbol_elegido,
          feature_names=feature_names,  
          class_names=[str(c) for c in sorted(set(y_dev))],
          filled=True, rounded=True, fontsize=10)
plt.title("Árbol de Decisión Óptimo")
plt.show()

CM1 = metrics.confusion_matrix(y_holdout, y_pred_eval, labels=dígitos_grupo)
plt.figure(figsize=(8, 6))
sns.heatmap(CM1, annot=True, fmt='d', cmap="Blues", xticklabels=dígitos_grupo, yticklabels=dígitos_grupo, cbar=False)
plt.title("Matriz de Confusión Árbol de Decisión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()

print("\nReporte de clasificación de árbol de decisión elegido, crietrio gini, profundidad 7:")
print(metrics.classification_report(y_holdout, y_pred_eval))

#%%

##KNN

Nrep = 5  
valores_n = range(1, 20)  

resultados_test = np.zeros((Nrep, len(valores_n)))
resultados_train = np.zeros((Nrep, len(valores_n)))

for i in range(Nrep):
    
    for k in valores_n:

        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_dev, y_dev) 
        
        Y_pred = model.predict(X_holdout)
        Y_pred_train = model.predict(X_dev)
        
        acc_test = metrics.accuracy_score(y_holdout, Y_pred)
        acc_train = metrics.accuracy_score(y_dev, Y_pred_train)
        
        resultados_test[i, k-1] = acc_test
        resultados_train[i, k-1] = acc_train

promedios_train = np.mean(resultados_train, axis=0) 
promedios_test = np.mean(resultados_test, axis=0)

plt.plot(valores_n, promedios_train, label='Train', marker='o')
plt.plot(valores_n, promedios_test, label='Test', marker='o')
plt.legend()
plt.title('Exactitud del modelo KNN')
plt.xlabel('Cantidad de vecinos')
plt.ylabel('Exactitud (accuracy)')
plt.show()

model = KNeighborsClassifier(n_neighbors = 3) 
model.fit(X_dev, y_dev)
Y_pred = model.predict(X_holdout) 

print("Exactitud del modelo en el conjunto eval:", metrics.accuracy_score(y_holdout, Y_pred))

CM2 = metrics.confusion_matrix(y_holdout, Y_pred, labels=dígitos_grupo)
plt.figure(figsize=(8, 6))
sns.heatmap(CM2, annot=True, fmt='d', cmap="Blues", xticklabels=dígitos_grupo, yticklabels=dígitos_grupo, cbar=False)
plt.title("Matriz de Confusión KNN")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()

print("\nReporte de clasificación de modelo KNN:")
print(metrics.classification_report(y_holdout, Y_pred))
