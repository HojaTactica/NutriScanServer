import os
import json
import time

from flask import Flask, request, jsonify
from google import genai
from google.genai import types


app = Flask(__name__)

# Límite máximo de tamaño de la petición
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# Modelo de Gemini
MODEL = "gemini-3.5-flash-lite"


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "NutriScan server funcionando"
    })


# =========================================================
# ANALIZAR FOTOGRAFÍA
# =========================================================

@app.route("/analyze", methods=["POST"])
def analyze():

    # -----------------------------------------------------
    # COMPROBAR API KEY
    # -----------------------------------------------------

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return jsonify({
            "error": "Falta configurar GEMINI_API_KEY"
        }), 500


    # -----------------------------------------------------
    # COMPROBAR IMAGEN
    # -----------------------------------------------------

    if "image" not in request.files:
        return jsonify({
            "error": "No se recibió ninguna fotografía"
        }), 400


    photo = request.files["image"]


    # -----------------------------------------------------
    # COMPROBAR FORMATO
    # -----------------------------------------------------

    if photo.mimetype not in [
        "image/jpeg",
        "image/png",
        "image/webp"
    ]:
        return jsonify({
            "error": "Formato de imagen no compatible"
        }), 400


    try:

        # -------------------------------------------------
        # LEER IMAGEN
        # -------------------------------------------------

        image_bytes = photo.read()

        print(
            "Imagen recibida:",
            len(image_bytes),
            "bytes"
        )


        # -------------------------------------------------
        # CREAR CLIENTE GEMINI
        # -------------------------------------------------

        client = genai.Client(
            api_key=api_key
        )


        # -------------------------------------------------
        # PROMPT
        # -------------------------------------------------

        
prompt = """
Eres el analista nutricional educativo de NutriScan.

Tu trabajo es interpretar etiquetas nutricionales y explicar
la información de forma clara, detallada, útil y comprensible.

Analiza la fotografía y devuelve EXCLUSIVAMENTE un JSON válido
con esta estructura:

{
  "productName": "",
  "servingSize": "",
  "calories": 0,
  "protein": 0,
  "carbohydrates": 0,
  "fats": 0,
  "sugars": 0,
  "sodium": 0,
  "ingredients": [],
  "vitamins": [],
  "minerals": [],
  "analysis": ""
}

REGLAS PARA LEER LA ETIQUETA:

- No inventes datos.
- Utiliza los valores correspondientes a UNA PORCIÓN.
- Si la etiqueta presenta valores por porción y por 100 g/ml,
  utiliza los de la porción.
- Si solamente presenta valores por 100 g/ml, no los presentes
  como valores por porción. Aclara esta limitación en el análisis.
- No confundas gramos con miligramos.
- Si un dato no aparece o no es legible, utiliza 0.
- No confundas azúcares totales con azúcares añadidos.
- No inventes ingredientes, vitaminas ni minerales.
- Si no puedes identificar el producto, deja productName vacío.

INSTRUCCIONES PARA EL ANÁLISIS:

El campo "analysis" debe contener un análisis amplio,
personalizado al producto y redactado en español.

Organiza el análisis con estos apartados:

1. ¿QUÉ ESTÁS CONSUMIENDO?
Explica qué tipo de producto parece ser y cuáles son
sus características nutricionales principales.

2. ASPECTOS NUTRICIONALES DESTACADOS
Identifica los nutrientes que más destacan.
Explica qué significan sus cantidades por porción.
Diferencia azúcares, grasas, sodio, proteínas y carbohidratos.
No califiques una cantidad como alta o baja sin contar
con información suficiente para justificarlo.

3. INGREDIENTES Y SU FUNCIÓN
Explica la función de los ingredientes que sean visibles
y reconocibles.
Señala cuáles merecen atención y por qué.
No atribuyas efectos que no estén respaldados.

4. POSIBLES EFECTOS DEL CONSUMO FRECUENTE
Explica de forma educativa qué consecuencias puede tener
el consumo frecuente de los componentes destacados.
Por ejemplo:
- Azúcares añadidos y salud dental.
- Exceso de sodio y salud cardiovascular.
- Cafeína y sueño o nerviosismo.
- Grasas saturadas y salud cardiovascular.

Menciona únicamente los aspectos que realmente correspondan
a los ingredientes y nutrientes identificados.
No afirmes que una sola porción causará una enfermedad.

5. ¿QUÉ PODRÍAS MEJORAR?
Propón alternativas y hábitos sencillos:
- Variar las bebidas y elegir agua con frecuencia.
- Incluir frutas enteras y verduras.
- Incorporar alimentos variados que aporten fibra,
  proteínas y otros nutrientes.
- Alternar productos similares con distintas composiciones.

Las recomendaciones deben relacionarse con este producto.
No sugieras dietas restrictivas, saltarse comidas,
compensar alimentos ni reducir drásticamente la alimentación.

6. CONCLUSIÓN
Resume qué es lo más importante de este producto,
qué conviene tener en cuenta y cómo consumirlo
dentro de una alimentación variada.

ESTILO:

- Habla directamente con el usuario.
- Utiliza un tono cercano, inteligente y educativo.
- Explica el porqué de cada observación.
- Evita frases genéricas que podrían aplicarse a cualquier producto.
- No exageres riesgos ni asustes al usuario.
- No diagnostiques enfermedades.
- No des recomendaciones médicas.
- No juzgues al usuario por consumir un producto.
- No afirmes que alguien consume demasiado basándote
  únicamente en una fotografía.

IMPORTANTE:
Si la imagen no contiene una etiqueta nutricional legible,
indica claramente que no es posible realizar un análisis fiable.
"""



        # =================================================
        # INTENTOS DE GEMINI
        # =================================================

        max_attempts = 4

        for attempt in range(1, max_attempts + 1):

            try:

                print(
                    f"Intento {attempt} de {max_attempts} con Gemini..."
                )


                # -----------------------------------------
                # LLAMADA A GEMINI
                # -----------------------------------------

                response = client.models.generate_content(

                    model=MODEL,

                    contents=[
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type=photo.mimetype
                        ),
                        prompt
                    ],

                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )


                # -----------------------------------------
                # RESPUESTA RECIBIDA
                # -----------------------------------------

                print(
                    "Gemini respondió correctamente."
                )


                result = json.loads(
                    response.text
                )


                print(
                    "JSON generado correctamente."
                )


                return jsonify(result)


            # =================================================
            # ERROR DE GEMINI
            # =================================================

            except Exception as error:

                error_text = str(error)

                print(
                    f"Error en intento {attempt}:"
                )

                print(error_text)


                # -----------------------------------------
                # DETERMINAR SI ES UN ERROR TRANSITORIO
                # -----------------------------------------

                temporary_error = (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                    or "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                    or "500" in error_text
                    or "INTERNAL" in error_text
                )


                # -----------------------------------------
                # SI NO ES TRANSITORIO
                # -----------------------------------------

                if not temporary_error:

                    print(
                        "El error no parece ser temporal."
                    )

                    return jsonify({
                        "error":
                            "Gemini rechazó la solicitud.",
                        "details":
                            error_text
                    }), 500


                # -----------------------------------------
                # SI YA NO QUEDAN INTENTOS
                # -----------------------------------------

                if attempt == max_attempts:

                    print(
                        "Se agotaron los intentos."
                    )

                    return jsonify({
                        "error":
                            "Gemini está temporalmente ocupado. "
                            "Intenta nuevamente en unos segundos."
                    }), 503


                # -----------------------------------------
                # ESPERA EXPONENCIAL
                # -----------------------------------------

                wait_time = 2 ** (attempt - 1)

                print(
                    f"Esperando {wait_time} segundos "
                    "antes de volver a intentar..."
                )

                time.sleep(wait_time)


    # =====================================================
    # ERROR GENERAL DEL SERVIDOR
    # =====================================================

    except Exception as error:

        print(
            "Error general del servidor:"
        )

        print(error)


        return jsonify({
            "error":
                "No se pudo analizar la fotografía.",
            "details":
                str(error)
        }), 500


# =========================================================
# INICIAR SERVIDOR
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
