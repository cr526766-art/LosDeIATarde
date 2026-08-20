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
    Eres un profesor experto en inteligencia artificial y aprendizaje
    automático.

    Tu tarea es explicar el resultado de una clasificación a un estudiante.

    Sigue obligatoriamente estas reglas:

    1. No uses tablas.
    2. No hagas una ficha técnica.
    3. No repitas únicamente los datos recibidos.
    4. Explica en párrafos por qué los valores ingresados llevaron al modelo
       a elegir la clase obtenida.
    5. Explica cuáles atributos fueron relevantes para distinguir esa clase.
    6. Utiliza lenguaje sencillo, claro y educativo.
    7. No inventes porcentajes, reglas, umbrales ni valores que no fueron
       proporcionados.
    8. Si no se proporcionaron las reglas internas del modelo, aclara que la
       explicación describe la interpretación general del resultado y no el
       recorrido exacto del árbol.
    9. Organiza la respuesta solamente con estos apartados:

       ### Resultado obtenido
       ### ¿Por qué se obtuvo este resultado?
       ### Interpretación de los valores
       ### Conclusión

    La parte más importante es explicar el motivo de la clasificación.
    """

    precision_texto = (
        f"{precision * 100:.2f}%"
        if precision is not None
        else "No disponible"
    )

    mensaje_usuario = f"""
    Se realizó una clasificación mediante aprendizaje automático.

    Modelo seleccionado: {modelo_seleccionado}
    Algoritmo utilizado: {algoritmo or "No especificado"}
    Base de datos: {base_seleccionada}
    Valores ingresados: {valores}
    Clase predicha: {resultado}
    Precisión registrada: {precision_texto}

    Explica principalmente por qué el modelo relacionó estos valores con
    la clase "{resultado}". No presentes una tabla ni una lista de
    especificaciones. Desarrolla una explicación narrativa para un
    estudiante.
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
            temperature=0.2,
            max_tokens=500,
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