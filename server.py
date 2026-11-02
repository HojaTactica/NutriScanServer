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
MODEL = "gemini-3.8-flash"


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
Analiza esta fotografía de una etiqueta nutricional.

Devuelve exclusivamente un JSON válido con esta estructura:

{
  "productName": "nombre del producto",
  "servingSize": "tamaño de porción",
  "calories": 0,
  "protein": 0,
  "carbohydrates": 0,
  "fats": 0,
  "sugars": 0,
  "sodium": 0,
  "ingredients": [],
  "vitamins": [],
  "minerals": [],
  "analysis": "explicación objetiva de la información"
}

Reglas:

- No inventes información que no aparezca en la etiqueta.
- Si no puedes identificar un dato, utiliza 0 como valor numérico.
- Si falta información, acláralo en "analysis".
- Usa gramos para proteínas, carbohidratos, grasas y azúcares.
- Usa miligramos para sodio.
- Las vitaminas y minerales deben tener esta estructura:

  {
    "name": "nombre",
    "amount": 0,
    "unit": "unidad"
  }

- No hagas recomendaciones médicas.
- Si la imagen no contiene una etiqueta nutricional legible,
  indícalo claramente en "analysis".
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
