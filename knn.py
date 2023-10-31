import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import warnings

clientes = pd.read_csv("creditos.csv")

buenos = clientes[clientes["cumplio"]==1]
malos = clientes[clientes["cumplio"]==0]

print("--Pagadores:")
print(buenos)
print("Total de pagadores: ", len(buenos), "\n\n")
print("--Deudores")
print(malos)
print("Total de deudores: ", len(malos))

plt.scatter(buenos["edad"], buenos["credito"],
            marker="*", s=150, color="brown",
            label="Sí pagó (Clase: 1)")

plt.scatter(malos["edad"], malos["credito"],
            marker="*", s=150, color="green", 
            label="No pagó (Clase: 0)")

plt.ylabel("Monto del crédito")
plt.xlabel("Edad")
plt.legend(bbox_to_anchor=(1, 0.2)) 
plt.show()

datos = clientes[["edad", "credito"]]
clase = clientes["cumplio"]

escalador = preprocessing.MinMaxScaler()

datos = escalador.fit_transform(datos)

print("\n\n--Datos escalados: ")
print(datos, "\n")

clasificador = KNeighborsClassifier(n_neighbors=3)

clasificador.fit(datos, clase)

warnings.filterwarnings('ignore', category=UserWarning)

#Nuevo solicitante
edad1 = 53
monto1 = 350000

#Escalar los datos del nuevo solicitante
solicitante = escalador.transform([[edad1, monto1]])

#Calcular clase y probabilidades
print("Solicitante 1")
print("Clase:", clasificador.predict(solicitante))
print("Probabilidades por clase", clasificador.predict_proba(solicitante))
print("\n")

#Nuevo solicitante
edad2 = 20
monto2 = 350000

#Escalar los datos del nuevo solicitante
solicitante = escalador.transform([[edad2, monto2]])

#Calcular clase y probabilidades
print("Solicitante 2")
print("Clase:", clasificador.predict(solicitante))
print("Probabilidades por clase",
      clasificador.predict_proba(solicitante))
print("\n")

#Nuevo solicitante
edad3 = 30
monto3 = 350000

#Escalar los datos del nuevo solicitante
solicitante = escalador.transform([[edad3, monto3]])

#Calcular clase y probabilidades
print("Solicitante 3")
print("Clase:", clasificador.predict(solicitante))
print("Probabilidades por clase",
      clasificador.predict_proba(solicitante))
print("\n")

#Nuevo solicitante
edad4 = 40
monto4 = 350000

#Escalar los datos del nuevo solicitante
solicitante = escalador.transform([[edad4, monto4]])

#Calcular clase y probabilidades
print("Solicitante 4")
print("Clase:", clasificador.predict(solicitante))
print("Probabilidades por clase",
      clasificador.predict_proba(solicitante))
print("\n")

#Nuevo solicitante
edad5 = 53
monto5 = 30000

#Escalar los datos del nuevo solicitante
solicitante = escalador.transform([[edad5, monto5]])

#Calcular clase y probabilidades
print("Solicitante 5")
print("Clase:", clasificador.predict(solicitante))
print("Probabilidades por clase",
      clasificador.predict_proba(solicitante))
print("\n")

#Código para graficar
plt.scatter(buenos["edad"], buenos["credito"],
            marker="*", s=150, color="brown", label="Sí pagó (Clase: 1)")
plt.scatter(malos["edad"], malos["credito"],
            marker="*", s=150, color="green", label="No pagó (Clase: 0)")

plt.scatter(edad1, monto1, marker="P", s=250, color="green", label="Solicitante1\n")
plt.scatter(edad2, monto2, marker="p", s=250, color="green", label="Solicitante2\n")
plt.scatter(edad3, monto3, marker="p", s=250, color="green", label="Solicitante3\n")
plt.scatter(edad4, monto4, marker="p", s=250, color="green", label="Solicitante4\n")
plt.scatter(edad5, monto5, marker="p", s=250, color="green", label="Solicitante5\n")

plt.ylabel("Monto del crédito")
plt.xlabel("Edad")
plt.legend(bbox_to_anchor=(1, 0.3))
plt.show()

#Datos sinténticos de todos los posibles solicitantes
creditos = np.array([np.arange(100000, 600010, 1000)]*43).reshape(1, -1)
edades = np.array([np.arange(18, 61)]*501).reshape(1, -1)
todos = pd.DataFrame(np.stack((edades, creditos), axis=2)[0],
                     columns=["edad", "credito"])

#Escalar los datos
solicitantes = escalador.transform(todos)

#Predecir todas las clases
clases_resultantes = clasificador.predict(solicitantes)

#Código para graficar
buenos = todos[clases_resultantes==1]
malos = todos[clases_resultantes==0]
plt.scatter(buenos["edad"], buenos["credito"],
            marker="*", s=150, color="brown", label="Sí pagará (Clase: 1)")
plt.scatter(malos["edad"], malos["credito"],
            marker="*", s=150, color="green", label="No pagará (Clase: 0)")
plt.ylabel("Monto del crédito")
plt.xlabel("Edad")
plt.legend(bbox_to_anchor=(1, 0.2))
plt.show()