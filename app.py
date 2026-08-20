import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from werkzeug.utils import secure_filename
import os
import pandas as pd
from scipy.io import arff
from pathlib import Path
import markdown
from markupsafe import escape
from flask import Flask, render_template, request, redirect, url_for
from algorithms.explicacion_ia import generar_explicacion

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "models"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MODEL_FOLDER"] = MODEL_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

def cargar_dataset(ruta):

    extension = os.path.splitext(ruta)[1].lower()

    # CSV
    if extension == ".csv":

        df = pd.read_csv(ruta)

    # ARFF
    elif extension == ".arff":

        datos, meta = arff.loadarff(ruta)

        df = pd.DataFrame(datos)

        # Convertir datos tipo bytes a texto
        for columna in df.select_dtypes(include=["object"]).columns:

            df[columna] = df[columna].apply(
                lambda x: x.decode("utf-8")
                if isinstance(x, bytes)
                else x
            )

    else:

        raise ValueError(
            "Formato de archivo no compatible"
        )

    return df


@app.route("/training.html", methods=["GET", "POST"])
def training():

    if request.method == "POST":

        # Obtener archivo enviado desde HTML
        archivo = request.files.get("dataset")

        # Obtener algoritmo seleccionado
        algoritmo = request.form.get("algorithm")

        # Comprobar que se seleccionó un archivo
        if archivo is None or archivo.filename == "":
            return render_template(
                "training.html",
                mensaje="Selecciona un archivo CSV o ARFF"
            )

        # Comprobar extensión
        if not archivo.filename.lower().endswith((".csv", ".arff")):
            return render_template(
                "training.html",
                mensaje="El archivo debe ser CSV o ARFF"
            )

        # Comprobar algoritmo
        if algoritmo is None or algoritmo == "":
            return render_template(
                "training.html",
                mensaje="Selecciona un algoritmo"
            )

        # Nombre seguro del archivo
        nombre_archivo = secure_filename(
            archivo.filename
        )

        # Ruta donde se guardará
        ruta = os.path.join(
            app.config["UPLOAD_FOLDER"],
            nombre_archivo
        )

        # Guardar archivo
        archivo.save(ruta)

        print("==============================")
        print("Archivo guardado en:", ruta)
        print("Algoritmo:", algoritmo)
        # Leer dataset
        df = cargar_dataset(ruta)
        # ======================================
        # SEPARAR ATRIBUTOS Y CLASE
        # ======================================

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]


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

        X = pd.get_dummies(X)


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

        elif algoritmo == "knn":

            modelo = KNeighborsClassifier(
                n_neighbors=3
            )

        else:

            return render_template(
                "training.html",
                mensaje="Algoritmo no válido"
            )


        # ======================================
        # ENTRENAR
        # ======================================

        modelo.fit(X_train, y_train)


        # ======================================
        # REALIZAR PREDICCIONES
        # ======================================

        y_pred = modelo.predict(X_test)



        nombre_dataset = os.path.splitext(nombre_archivo)[0]

        nombre_modelo = f"{nombre_dataset}_{algoritmo}.joblib"

        ruta_modelo = os.path.join(
            app.config["MODEL_FOLDER"],
            nombre_modelo
        )

        paquete_modelo = {
           "modelo": modelo,
           "columnas": X.columns.tolist(),
           "clase": df.columns[-1],
           "algoritmo": algoritmo
       }

        joblib.dump(
            paquete_modelo,
            ruta_modelo
        )

        print("Modelo guardado en:", ruta_modelo)

        # ======================================
        # CALCULAR MÉTRICAS
        # ======================================

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        support = len(y_test)


        # ======================================
        # CONVERTIR A PORCENTAJE
        # ======================================

       
        accuracy = round(
            accuracy * 100,
            2
        )

        precision = round(
            precision * 100,
            2
        )

        recall = round(
            recall * 100,
            2
        )

        f1 = round(
            f1 * 100,
            2
        )


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
           mensaje="Modelo entrenado correctamente",
           accuracy=str(accuracy) + "%",
           precision=str(precision) + "%",
           recall=str(recall) + "%",
           f1_score=str(f1) + "%",
           support=support
        )
    return render_template("training.html")





CARPETA_BASE = Path(__file__).parent
CARPETA_MODELOS = CARPETA_BASE / "models"
CARPETA_DATOS = CARPETA_BASE / "data"

# Nombres comunes para la columna que contiene la clasificación.
COLUMNAS_OBJETIVO = [
    "species",
    "clase",
    "class",
    "target",
    "objetivo",
    "resultado",
    "label"
]


def obtener_modelos():
    """Obtiene los modelos .joblib disponibles."""
    CARPETA_MODELOS.mkdir(exist_ok=True)

    return sorted([
        archivo.name
        for archivo in CARPETA_MODELOS.glob("*.joblib")
    ])


def obtener_bases_datos():
    """Obtiene los archivos CSV disponibles."""
    CARPETA_DATOS.mkdir(exist_ok=True)

    return sorted([
        archivo.name
        for archivo in CARPETA_DATOS.glob("*.csv")
    ])


def obtener_ruta_segura(carpeta, nombre_archivo):
    """Evita que se acceda a archivos fuera de la carpeta indicada."""
    nombre_seguro = Path(nombre_archivo).name
    ruta = carpeta / nombre_seguro

    if not ruta.exists() or not ruta.is_file():
        raise FileNotFoundError(
            f"El archivo '{nombre_seguro}' no existe."
        )

    return ruta


def cargar_modelo(nombre_archivo):
    """Carga un modelo guardado con Joblib."""
    ruta_modelo = obtener_ruta_segura(
        CARPETA_MODELOS,
        nombre_archivo
    )

    contenido = joblib.load(ruta_modelo)

    # Si el archivo contiene un paquete o diccionario.
    if isinstance(contenido, dict):
        if "modelo" not in contenido:
            raise ValueError(
                "El archivo no contiene la clave 'modelo'."
            )

        return {
            "modelo": contenido["modelo"],
            "algoritmo": contenido.get(
                "algoritmo",
                Path(nombre_archivo).stem
            ),
            "precision": contenido.get("precision"),
            "clases": contenido.get("clases")
        }

    # Si el archivo contiene directamente el modelo entrenado.
    if not hasattr(contenido, "predict"):
        raise ValueError(
            "El archivo seleccionado no contiene un modelo válido."
        )

    return {
        "modelo": contenido,
        "algoritmo": Path(nombre_archivo).stem,
        "precision": None,
        "clases": None
    }


def cargar_base_datos(nombre_archivo):
    """Carga el CSV y obtiene sus atributos."""
    ruta_csv = obtener_ruta_segura(
        CARPETA_DATOS,
        nombre_archivo
    )

    datos = pd.read_csv(ruta_csv)

    if datos.empty:
        raise ValueError(
            "La base de datos seleccionada está vacía."
        )

    columna_objetivo = None

    # Busca automáticamente la columna objetivo.
    for columna in datos.columns:
        if columna.strip().lower() in COLUMNAS_OBJETIVO:
            columna_objetivo = columna
            break

    # Si no encuentra una columna conocida, usa la última.
    if columna_objetivo is None:
        columna_objetivo = datos.columns[-1]

    atributos = [
        columna
        for columna in datos.columns
        if columna != columna_objetivo
    ]

    if not atributos:
        raise ValueError(
            "No se encontraron atributos para clasificar."
        )

    return datos, atributos, columna_objetivo


# La dirección principal abre la pantalla de clasificación.
@app.route("/")
def inicio():
    return redirect(url_for("clasificar"))



@app.route("/classify.html", methods=["GET", "POST"])
def clasificar():
    modelos = obtener_modelos()
    bases_datos = obtener_bases_datos()

    modelo_seleccionado = request.values.get(
        "modelo",
        ""
    )

    base_seleccionada = request.values.get(
        "base_datos",
        ""
    )

    atributos = []
    columna_objetivo = None
    valores = {}
    resultado = None
    explicacion = None
    error = None

    if modelo_seleccionado and base_seleccionada:
        try:
            paquete = cargar_modelo(modelo_seleccionado)

            datos, atributos, columna_objetivo = cargar_base_datos(
                base_seleccionada
            )

            modelo = paquete["modelo"]
            algoritmo = paquete["algoritmo"]
            precision = paquete["precision"]

            cantidad_esperada = getattr(
                modelo,
                "n_features_in_",
                None
            )

            if (
                cantidad_esperada is not None
                and cantidad_esperada != len(atributos)
            ):
                raise ValueError(
                    f"El modelo necesita {cantidad_esperada} atributos, "
                    f"pero la base de datos contiene {len(atributos)}."
                )

            if request.method == "POST":
                datos_entrada = []

                for atributo in atributos:
                    valor = request.form.get(
                        atributo,
                        ""
                    ).strip()

                    valores[atributo] = valor

                    if not valor:
                        raise ValueError(
                            f"Debes ingresar un valor para: {atributo}"
                        )

                    try:
                        datos_entrada.append(float(valor))
                    except ValueError:
                        raise ValueError(
                            f"El valor de '{atributo}' debe ser numérico."
                        )

                entrada = pd.DataFrame(
                    [datos_entrada],
                    columns=atributos
                )

                prediccion = modelo.predict(entrada)
                resultado = str(prediccion[0])

                # explicacion = (
                #     f"El modelo {algoritmo} analizó los atributos "
                #     f"de la base de datos {base_seleccionada} y "
                #     f"determinó que el resultado para "
                #     f"'{columna_objetivo}' es: {resultado}."
                # )


                explicacion_markdown = generar_explicacion(
                    resultado= resultado,
                    modelo_seleccionado = modelo_seleccionado,
                    base_seleccionada = base_seleccionada,
                    valores = valores,
                    algoritmo = algoritmo,
                    precision = precision
                )

                explicacion = markdown.markdown(
                str(escape(explicacion_markdown)),
                extensions=["tables", "fenced_code"]
                )
               

        except (ValueError, FileNotFoundError) as excepcion:
            error = str(excepcion)

        except Exception as excepcion:
            error = (
                "No fue posible realizar la clasificación: "
                f"{excepcion}"
            )

    return render_template(
        "classify.html",
        modelos=modelos,
        bases_datos=bases_datos,
        modelo_seleccionado=modelo_seleccionado,
        base_seleccionada=base_seleccionada,
        columna_objetivo=columna_objetivo,
        atributos=atributos,
        valores=valores,
        resultado=resultado,
        explicacion=explicacion,
        error=error,
        pagina_actual="clasificar"
    )


if __name__ == "__main__":
    app.run(debug=True)