import os
import math
import joblib
import pandas as pd

from scipy.io import arff
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# CONFIGURACIÓN DE FLASK
# ==========================================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "models"

# Máximo 10 MB por archivo
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MODEL_FOLDER"] = MODEL_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

EXTENSIONES_PERMITIDAS = {".csv", ".arff"}
ALGORITMOS_PERMITIDOS = {"id3", "knn"}


# ==========================================
# FUNCIONES DE VALIDACIÓN
# ==========================================

def extension_permitida(nombre_archivo):
    """
    Comprueba que la extensión sea exactamente
    .csv o .arff.
    """

    extension = os.path.splitext(nombre_archivo)[1].lower()

    return extension in EXTENSIONES_PERMITIDAS


def eliminar_archivo(ruta):
    """
    Elimina un archivo inválido de uploads.
    """

    try:
        if ruta and os.path.exists(ruta):
            os.remove(ruta)
    except OSError:
        pass


def cargar_csv(ruta):
    """
    Intenta leer archivos CSV guardados en UTF-8
    o ISO-8859-1.

    También permite separador de coma o punto y coma.
    """

    codificaciones = [
        "utf-8-sig",
        "utf-8",
        "latin-1"
    ]

    errores = []

    for codificacion in codificaciones:

        try:
            # sep=None permite detectar coma o punto y coma
            return pd.read_csv(
                ruta,
                encoding=codificacion,
                sep=None,
                engine="python"
            )

        except (
            UnicodeDecodeError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError
        ) as error:

            errores.append(str(error))

    raise ValueError(
        "No se pudo leer el archivo CSV. Comprueba que tenga "
        "encabezados en la primera fila, que todas las filas "
        "tengan la misma cantidad de columnas y que utilice "
        "coma o punto y coma como separador."
    )


def cargar_arff(ruta):
    """
    Lee un archivo ARFF y convierte los valores
    tipo bytes a texto.
    """

    try:
        datos, meta = arff.loadarff(ruta)

        df = pd.DataFrame(datos)

        for columna in df.select_dtypes(
            include=["object"]
        ).columns:

            df[columna] = df[columna].apply(
                lambda valor: valor.decode(
                    "utf-8",
                    errors="replace"
                )
                if isinstance(valor, bytes)
                else valor
            )

        return df

    except Exception as error:
        raise ValueError(
            f"El archivo ARFF no tiene una estructura válida: {error}"
        )


def cargar_dataset(ruta):
    """
    Carga un archivo únicamente si su extensión
    es CSV o ARFF.
    """

    extension = os.path.splitext(ruta)[1].lower()

    if extension == ".csv":
        df = cargar_csv(ruta)

    elif extension == ".arff":
        df = cargar_arff(ruta)

    else:
        raise ValueError(
            "Formato no compatible. Solo se permiten CSV y ARFF."
        )

    return df


def validar_dataset(df):
    """
    Comprueba que el archivo tenga una estructura válida
    para un problema de clasificación supervisada.

    Se considera que la última columna es la clase.
    """

    # Eliminar filas completamente vacías
    df = df.dropna(how="all")

    if df.empty:
        raise ValueError(
            "El dataset está vacío."
        )

    # Verificar cantidad mínima de columnas
    if df.shape[1] < 2:
        raise ValueError(
            "El dataset debe tener al menos dos columnas: "
            "una columna de atributos y una columna de clase."
        )

    # Verificar cantidad mínima de registros
    if df.shape[0] < 10:
        raise ValueError(
            "El dataset debe contener al menos 10 registros."
        )

    # Limpiar espacios de los encabezados
    df.columns = [
        str(columna).strip()
        for columna in df.columns
    ]

    # Verificar nombres de columnas vacíos
    columnas_vacias = [
        columna
        for columna in df.columns
        if (
            columna == ""
            or columna.lower().startswith("unnamed:")
        )
    ]

    if columnas_vacias:
        raise ValueError(
            "El dataset contiene columnas sin nombre. "
            "Todos los atributos deben tener un encabezado."
        )

    # Comprobar encabezados duplicados
    if df.columns.duplicated().any():
        raise ValueError(
            "El dataset contiene nombres de columnas repetidos."
        )

    # Separar atributos y clase
    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()

    nombre_clase = df.columns[-1]

    # Verificar que existan atributos
    if X.shape[1] == 0:
        raise ValueError(
            "No se encontraron atributos para entrenar."
        )

    # Verificar valores vacíos en atributos
    if X.isnull().any().any():
        columnas_con_vacios = X.columns[
            X.isnull().any()
        ].tolist()

        raise ValueError(
            "El dataset contiene valores vacíos en los atributos: "
            + ", ".join(columnas_con_vacios)
        )

    # Verificar valores vacíos en la clase
    if y.isnull().any():
        raise ValueError(
            f"La columna de clase '{nombre_clase}' "
            "contiene valores vacíos."
        )

    # Convertir la clase a texto uniforme si es categórica
    if (
        pd.api.types.is_object_dtype(y)
        or isinstance(y.dtype, pd.CategoricalDtype)
    ):
        y = y.astype(str).str.strip()

        if (y == "").any():
            raise ValueError(
                f"La columna de clase '{nombre_clase}' "
                "contiene valores vacíos."
            )

    cantidad_clases = y.nunique()

    # Deben existir al menos dos clases
    if cantidad_clases < 2:
        raise ValueError(
            f"La columna '{nombre_clase}' debe contener "
            "al menos dos clases diferentes."
        )

    # Rechazar columnas donde cada valor sea diferente
    if cantidad_clases == len(y):
        raise ValueError(
            f"La última columna, '{nombre_clase}', tiene un valor "
            "diferente en cada registro. Probablemente no es una "
            "columna de clasificación."
        )

    # Detectar un posible problema de regresión
    if (
        pd.api.types.is_numeric_dtype(y)
        and cantidad_clases > max(20, int(len(y) * 0.20))
    ):
        raise ValueError(
            f"La columna '{nombre_clase}' contiene demasiados "
            "valores numéricos diferentes. Parece una variable "
            "continua para regresión, pero esta aplicación solamente "
            "entrena modelos de clasificación."
        )

    # Verificar cantidad de ejemplos por clase
    conteo_clases = y.value_counts()

    clases_insuficientes = conteo_clases[
        conteo_clases < 2
    ]

    if not clases_insuficientes.empty:
        nombres = [
            str(clase)
            for clase in clases_insuficientes.index
        ]

        raise ValueError(
            "Cada clase debe aparecer por lo menos dos veces. "
            "Clases con pocos registros: "
            + ", ".join(nombres)
        )

    # Verificar que HOLD-OUT pueda incluir todas las clases
    cantidad_prueba = math.ceil(len(df) * 0.30)
    cantidad_entrenamiento = len(df) - cantidad_prueba

    if cantidad_prueba < cantidad_clases:
        raise ValueError(
            "El dataset no tiene suficientes registros para colocar "
            "todas las clases en el conjunto de prueba del 30%."
        )

    if cantidad_entrenamiento < cantidad_clases:
        raise ValueError(
            "El dataset no tiene suficientes registros para colocar "
            "todas las clases en el conjunto de entrenamiento."
        )

    # Comprobar valores infinitos en columnas numéricas
    columnas_numericas = X.select_dtypes(
        include=["number"]
    ).columns

    if len(columnas_numericas) > 0:

        tiene_infinito = X[
            columnas_numericas
        ].isin([float("inf"), float("-inf")]).any().any()

        if tiene_infinito:
            raise ValueError(
                "El dataset contiene valores infinitos."
            )

    return df, X, y


# ==========================================
# RUTAS
# ==========================================

@app.route("/classify")
def classify():
    return render_template("classify.html")


@app.route("/training", methods=["GET", "POST"])
def training():

    if request.method == "GET":
        return render_template("training.html")

    ruta = None

    try:
        # Obtener archivo y algoritmo
        archivo = request.files.get("dataset")
        algoritmo = request.form.get(
            "algorithm",
            ""
        ).strip().lower()

        # ======================================
        # VALIDAR ARCHIVO
        # ======================================

        if archivo is None or archivo.filename == "":
            raise ValueError(
                "Selecciona un archivo CSV o ARFF."
            )

        if not extension_permitida(archivo.filename):
            raise ValueError(
                "Formato no permitido. "
                "Solo se aceptan archivos CSV o ARFF."
            )

        # ======================================
        # VALIDAR ALGORITMO
        # ======================================

        if algoritmo == "":
            raise ValueError(
                "Selecciona un algoritmo."
            )

        if algoritmo not in ALGORITMOS_PERMITIDOS:
            raise ValueError(
                "Algoritmo no válido. "
                "Selecciona ID3 o KNN."
            )

        # ======================================
        # GUARDAR ARCHIVO
        # ======================================

        nombre_archivo = secure_filename(
            archivo.filename
        )

        if nombre_archivo == "":
            raise ValueError(
                "El nombre del archivo no es válido."
            )

        # Volver a comprobar la extensión después
        # de limpiar el nombre
        if not extension_permitida(nombre_archivo):
            raise ValueError(
                "La extensión del archivo no es válida."
            )

        ruta = os.path.join(
            app.config["UPLOAD_FOLDER"],
            nombre_archivo
        )

        archivo.save(ruta)

        print("==============================")
        print("Archivo guardado en:", ruta)
        print("Algoritmo:", algoritmo)

        # ======================================
        # LEER Y VALIDAR DATASET
        # ======================================

        df = cargar_dataset(ruta)

        df, X, y = validar_dataset(df)

        print("\n==============================")
        print("INFORMACIÓN DEL DATASET")
        print("==============================")

        print("Filas:", df.shape[0])
        print("Columnas:", df.shape[1])

        print("\nAtributos:")
        print(X.columns.tolist())

        print("\nClase:")
        print(df.columns[-1])

        print("\nClases encontradas:")
        print(y.unique())

        # ======================================
        # CONVERTIR VARIABLES CATEGÓRICAS
        # ======================================

        X = pd.get_dummies(
            X,
            drop_first=False
        )

        if X.shape[1] == 0:
            raise ValueError(
                "No quedaron atributos después del preprocesamiento."
            )

        # KNN e ID3 necesitan valores numéricos
        try:
            X = X.astype(float)

        except ValueError:
            raise ValueError(
                "No fue posible convertir los atributos a valores "
                "numéricos para entrenar el modelo."
            )

        # ======================================
        # HOLD-OUT 70 / 30
        # ======================================

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=42,
            stratify=y
        )

        print("\n==============================")
        print("HOLD-OUT")
        print("==============================")

        print("Entrenamiento:", len(X_train))
        print("Prueba:", len(X_test))

        # ======================================
        # SELECCIONAR ALGORITMO
        # ======================================

        if algoritmo == "id3":

            modelo = DecisionTreeClassifier(
                criterion="entropy",
                random_state=42
            )

        else:

            if len(X_train) < 3:
                raise ValueError(
                    "KNN necesita al menos tres registros "
                    "en el conjunto de entrenamiento."
                )

            modelo = KNeighborsClassifier(
                n_neighbors=3
            )

        # ======================================
        # ENTRENAR Y PREDECIR
        # ======================================

        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)

        # ======================================
        # CALCULAR MÉTRICAS
        # ======================================

        accuracy = round(
            accuracy_score(y_test, y_pred) * 100,
            2
        )

        precision = round(
            precision_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            ) * 100,
            2
        )

        recall = round(
            recall_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            ) * 100,
            2
        )

        f1 = round(
            f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            ) * 100,
            2
        )

        support = len(y_test)

        # ======================================
        # GUARDAR MODELO
        # ======================================

        nombre_dataset = os.path.splitext(
            nombre_archivo
        )[0]

        nombre_modelo = (
            f"{nombre_dataset}_{algoritmo}.joblib"
        )

        ruta_modelo = os.path.join(
            app.config["MODEL_FOLDER"],
            nombre_modelo
        )

        paquete_modelo = {
            "modelo": modelo,
            "columnas": X.columns.tolist(),
            "clase": df.columns[-1],
            "clases": y.unique().tolist(),
            "algoritmo": algoritmo,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support
        }

        joblib.dump(
            paquete_modelo,
            ruta_modelo
        )

        print("\nModelo guardado en:", ruta_modelo)

        print("\n==============================")
        print("RESULTADOS")
        print("==============================")

        print("Algoritmo:", algoritmo)
        print("Accuracy:", accuracy, "%")
        print("Precision:", precision, "%")
        print("Recall:", recall, "%")
        print("F1-Score:", f1, "%")
        print("Support:", support)

        return render_template(
            "training.html",
            mensaje="Modelo entrenado correctamente.",
            nombre_modelo=nombre_modelo,
            accuracy=f"{accuracy}%",
            precision=f"{precision}%",
            recall=f"{recall}%",
            f1_score=f"{f1}%",
            support=support
        )

    except ValueError as error:

        eliminar_archivo(ruta)

        return render_template(
            "training.html",
            mensaje=str(error)
        )

    except pd.errors.EmptyDataError:

        eliminar_archivo(ruta)

        return render_template(
            "training.html",
            mensaje="El archivo está vacío."
        )

    except pd.errors.ParserError:

        eliminar_archivo(ruta)

        return render_template(
            "training.html",
            mensaje=(
                "El archivo no tiene una estructura tabular válida. "
                "Todas las filas deben tener la misma cantidad "
                "de columnas."
            )
        )

    except Exception as error:

        eliminar_archivo(ruta)

        print("ERROR:", error)

        return render_template(
            "training.html",
            mensaje=(
                "No fue posible procesar el dataset: "
                f"{error}"
            )
        )


# ==========================================
# ARCHIVOS DEMASIADO GRANDES
# ==========================================

@app.errorhandler(RequestEntityTooLarge)
def archivo_demasiado_grande(error):

    return render_template(
        "training.html",
        mensaje="El archivo supera el límite permitido de 10 MB."
    ), 413


# ==========================================
# EJECUTAR APLICACIÓN
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)
