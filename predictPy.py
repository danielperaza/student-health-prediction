# Version_1.0
# Octubre 2024
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import matplotlib.ticker as mticker
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns


class Analisis_Predictivo:

    def __init__(self, datos: pd.DataFrame, X_train=None, X_test=None, y_train= None, y_test = None, predecir=None,
                 modelo=None, random_state=None):
        '''
        datos: Datos completos y listos para construir un modelo   

        X_train, X_test, y_train, y_test Son las tablas de datos de training y testing

        predecir: Nombre de la variable a predecir

        modelo: Instancia de una Clase de un método de clasificación(KNN,Árboles,SVM,etc).
        Si no especifica un modelo no podrá utilizar el método fit

        random_state: Semilla aleatoria para la división de datos(training-testing).
        '''
        self.__datos = datos
        self.__X_train = X_train
        self.__X_test = X_test
        self.__y_train = y_train
        self.__y_test = y_test
        self.__predecir = predecir
        self.__nombre_clases = list(np.unique(self.__datos[predecir].values))
        self.__modelo = modelo
        self.__random_state = random_state
    
    @property
    def datos(self):
        return self.__datos
    
    @property
    def X_train(self):
        return self.__X_train
    
    @property
    def X_test(self):
        return self.__X_test
    
    @property
    def y_train(self):
        return self.__y_train
    
    @property
    def y_test(self):
        return self.__y_test
    
    @property
    def predecir(self):
        return self.__predecir
    
    
    @property
    def nombre_clases(self):
        return self.__nombre_clases
    
    @property
    def modelo(self):
        return self.__modelo
    
    @property
    def random_state(self):
        return self.__random_state
    
    
    @datos.setter
    def datos(self, datos):
        self.__datos = datos
    
    @predecir.setter
    def predecir(self, predecir):
        self.__predecir = predecir
        
        
    @nombre_clases.setter
    def nombre_clases(self, nombre_clases):
        self.__nombre_clases = nombre_clases
        
    @modelo.setter
    def modelo(self, modelo):
        self.__modelo = modelo
        
    @random_state.setter
    def random_state(self, random_state):
        self.__random_state = random_state
        

    def fit_predict(self):
        if(self.modelo != None):
            self.modelo.fit(self.X_train, self.y_train)
            return self.modelo.predict(self.X_test)

    def fit_predict_resultados(self, imprimir=True):
        if(self.modelo != None):
            prediccion = self.fit_predict()
            MC = confusion_matrix(self.y_test, prediccion, labels= self.nombre_clases)
            indices = self.indices_general(MC, self.nombre_clases)
            if imprimir == True:
                for k in indices:
                    print("\n%s:\n%s" % (k, str(indices[k])))

            return indices

    def indices_general(self, MC, nombres=None):
        "Método para calcular los índices de calidad de la predicción"
        precision_global = np.sum(MC.diagonal()) / np.sum(MC)
        error_global = 1 - precision_global
        precision_categoria = pd.DataFrame(MC.diagonal()/np.sum(MC, axis=1)).T
        if nombres != None:
            precision_categoria.columns = nombres
        return {"Matriz de Confusión": MC,
                "Precisión Global": precision_global,
                "Error Global": error_global,
                "Precisión por categoría": precision_categoria}

    def distribucion_variable_predecir(self, ax=None):
        "Método para graficar la distribución de la variable a predecir"
        variable_predict = self.predecir
        data = self.datos
        if ax == None:
            fig, ax = plt.subplots(1, 1, figsize = (15,10), dpi = 50)
        colors = list(dict(**mcolors.CSS4_COLORS))
        df = pd.crosstab(index=data[variable_predict],
                         columns="valor") / data[variable_predict].count()
        countv = 0
        titulo = "Distribución de la variable %s" % variable_predict
        for i in range(df.shape[0]):
            ax.barh(1, df.iloc[i], left=countv, align='center',
                    color=colors[11+i], label=df.iloc[i].name)
            countv = countv + df.iloc[i]
        ax.set_xlim(0, 1)
        ax.set_yticklabels("")
        ax.set_ylabel(variable_predict)
        ax.set_title(titulo)
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(['{:.0%}'.format(x) for x in ticks_loc])
        countv = 0
        for v in df.iloc[:, 0]:
            ax.text(np.mean([countv, countv+v]) - 0.03, 1,
                    '{:.1%}'.format(v), color='black', fontweight='bold')
            countv = countv + v
        ax.legend(loc='upper center', bbox_to_anchor=(
            1.08, 1), shadow=True, ncol=1)

    def poder_predictivo_categorica(self, var: str, ax=None):
        "Método para ver la distribución de una variable categórica respecto a la predecir"
        data = self.datos
        variable_predict = self.predecir
        if ax == None:
            fig, ax = plt.subplots(1, 1, figsize = (15,10), dpi = 50)
        df = pd.crosstab(index=data[var], columns=data[variable_predict])
        df = df.div(df.sum(axis=1), axis=0)
        titulo = "Distribución de la variable %s según la variable %s" % (
            var, variable_predict)
        df.plot(kind='barh', stacked=True, legend=True,
                ax=ax, xlim=(0, 1), title=titulo, width=0.8)
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(['{:.0%}'.format(x) for x in ticks_loc])
        ax.legend(loc='upper center', bbox_to_anchor=(
            1.08, 1), shadow=True, ncol=1)
        for bars in ax.containers:
            plt.setp(bars, width=.9)
        for i in range(df.shape[0]):
            countv = 0
            for v in df.iloc[i]:
                ax.text(np.mean([countv, countv+v]) - 0.03, i, '{:.1%}'.format(v),
                        color='black', fontweight='bold')
                countv = countv + v

    def poder_predictivo_numerica(self, var: str, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))

        sns.histplot(
            data=self.datos,
            x=var,
            hue=self.predecir,
            stat="density",
            common_norm=False,
            kde=True,
            alpha=0.35,
            ax=ax
        )

        ax.set_title(var)
def indices_general(MC, nombres = None):
    precision_global = np.sum(MC.diagonal()) / np.sum(MC)
    error_global     = 1 - precision_global
    precision_categoria  = pd.DataFrame(MC.diagonal()/np.sum(MC,axis = 1)).T
    if nombres!=None:
        precision_categoria.columns = nombres
    return {"Matriz de Confusión":MC, 
            "Precisión Global":   precision_global, 
            "Error Global":       error_global, 
            "Precisión por categoría":precision_categoria}
