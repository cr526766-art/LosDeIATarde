from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

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


# Esta ruta únicamente muestra la pantalla de entrenamiento.
# Todavía no procesa ningún formulario.
@app.route("/training.html")
def entrenar():
    return render_template(
        "training.html",
        pagina_actual="entrenar"
    )


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

                explicacion = (
                    f"El modelo {algoritmo} analizó los atributos "
                    f"de la base de datos {base_seleccionada} y "
                    f"determinó que el resultado para "
                    f"'{columna_objetivo}' es: {resultado}."
                )

                if precision is not None:
                    explicacion += (
                        f" La precisión registrada durante el "
                        f"entrenamiento fue de "
                        f"{precision * 100:.2f}%."
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