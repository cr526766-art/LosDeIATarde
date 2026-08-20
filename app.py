import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import pandas as pd
from scipy.io import arff

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

@app.route("/classify")
def classify():
    return render_template("classify.html")

@app.route("/training", methods=["GET", "POST"])
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

if __name__ == "__main__":
    app.run(debug=True)