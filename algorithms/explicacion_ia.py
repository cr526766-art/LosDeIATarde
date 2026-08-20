import os

from groq import Groq


def generar_explicacion(
    resultado,
    modelo_seleccionado,
    base_seleccionada,
    valores,
    algoritmo=None,
    precision=None
):
    """Genera una explicación de la clasificación mediante Groq."""

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        return (
            "La clasificación se realizó correctamente, pero no se "
            "encontró la variable de entorno GROQ_API_KEY."
        )

    client = Groq(api_key=api_key)
    
    prompt = """

Eres un biólogo experto en análisis de datos biológicos, clasificación de
especies y aprendizaje automático. Explica el resultado a un estudiante de
forma concreta, clara, breve y científicamente responsable.

REGLAS:

- No uses tablas ni fichas técnicas.
- No repitas innecesariamente todos los datos.
- Explica cómo la combinación de atributos se relaciona con la clase predicha.
- Menciona solo los atributos que parezcan relevantes.
- No inventes unidades, rangos, porcentajes, umbrales, probabilidades,
  características biológicas ni reglas internas del modelo.
- Si faltan las reglas del modelo, aclara brevemente que es una interpretación
  biológica general, no el proceso exacto de decisión.
- La precisión representa el rendimiento general del modelo, no la
  probabilidad de que esta predicción sea correcta.
- Usa lenguaje sencillo, párrafos cortos y evita información de relleno.

Revisa si algún valor parece demasiado alto, bajo o diferente. Considera como
posibles causas un punto decimal, unidad, escala o error de captura. No afirmes
que es incorrecto sin conocer las unidades, rangos normales y datos de
entrenamiento. Si parece sospechoso, indica qué valor revisar y cómo podría
afectar la clasificación.

Responde únicamente con estas secciones:

### Resultado obtenido
### ¿Por qué se obtuvo?
### Interpretación biológica
### Revisión de los datos
### Conclusión

La respuesta debe tener entre 180 y 300 palabras.
    """

    precision_texto = (
        f"{precision * 100:.2f}%"
        if precision is not None
        else "No disponible"
    )

    mensaje_usuario = f"""
Explica esta clasificación:

Modelo: {modelo_seleccionado}
Algoritmo: {algoritmo or "No especificado"}
Base de datos: {base_seleccionada}
Valores: {valores}
Resultado: {resultado}
Precisión general: {precision_texto}

Revisa posibles valores atípicos o errores de captura antes de explicar el
resultado. Si faltan unidades, rangos o reglas internas, indícalo brevemente.
"""
 
    try:
        respuesta = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": mensaje_usuario,
                },
            ],
            model="openai/gpt-oss-120b",
            temperature=0.15,
            max_tokens=850,
        )

        contenido = respuesta.choices[0].message.content

        if not contenido:
            return "Groq no devolvió una explicación."
        return contenido.strip()

    except Exception as error:
        print(f"Error al generar la explicación: {error}")

        return (
            "La clasificación se realizó correctamente, pero no fue "
            "posible generar su explicación mediante inteligencia artificial."
        )