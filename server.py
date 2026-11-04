
import os
import json
import time

from flask import Flask, request, jsonify
from google import genai
from google.genai import types


app = Flask(__name__)

# Limite maximo de la peticion
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Modelo de Gemini
MODEL = "gemini-3.5-flash-lite"


# =========================================================
# PROMPT DE ANALISIS NUTRICIONAL
# =========================================================

PROMPT = """
Eres el analista nutricional educativo de NutriScan.

Tu trabajo es interpretar etiquetas nutricionales y explicar
la informacion de forma clara, detallada, util y comprensible.

Analiza la fotografia y devuelve EXCLUSIVAMENTE un JSON valido
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
- Utiliza los valores correspondientes a UNA PORCION.
- Si la etiqueta presenta valores por porcion y por 100 g/ml,
  utiliza los de la porcion.
- Si solamente presenta valores por 100 g/ml, aclara esta
  limitacion en el analisis.
- No confundas gramos con miligramos.
- Si un dato no aparece o no es legible, utiliza 0.
- No confundas azucares totales con azucares anadidos.
- No inventes ingredientes, vitaminas ni minerales.
- Si no puedes identificar el producto, deja productName vacio.

INSTRUCCIONES PARA EL ANALISIS:

El campo "analysis" debe contener un analisis amplio,
personalizado al producto y redactado en espanol.

Organiza el analisis con estos apartados:

1. QUE ESTAS CONSUMIENDO

Explica que tipo de producto parece ser y cuales son
sus caracteristicas nutricionales principales.

2. ASPECTOS NUTRICIONALES DESTACADOS

Identifica los nutrientes que mas destacan.
Explica que significan sus cantidades por porcion.
Diferencia azucares, grasas, sodio, proteinas y carbohidratos.
No califiques una cantidad como alta o baja sin contar
con informacion suficiente para justificarlo.

3. INGREDIENTES Y SU FUNCION

Explica la funcion de los ingredientes visibles y reconocibles.
Senala cuales merecen atencion y por que.
No atribuyas efectos que no esten respaldados.

4. POSIBLES EFECTOS DEL CONSUMO FRECUENTE

Explica de forma educativa que consecuencias puede tener
el consumo frecuente de los componentes destacados.

Por ejemplo:
- Azucares anadidos y salud dental.
- Exceso de sodio y salud cardiovascular.
- Cafeina y sueno o nerviosismo.
- Grasas saturadas y salud cardiovascular.

Menciona solamente los aspectos que correspondan
a los nutrientes e ingredientes identificados.
No afirmes que una sola porcion causara una enfermedad.

5. QUE PODRIAS MEJORAR

Propone alternativas y habitos sencillos:
- Variar las bebidas y elegir agua con frecuencia.
- Incluir frutas enteras y verduras.
- Incorporar alimentos variados que aporten fibra,
  proteinas y otros nutrientes.
- Alternar productos similares con distintas composiciones.

Las recomendaciones deben relacionarse con este producto.
No sugieras dietas restrictivas, saltarse comidas,
compensar alimentos ni reducir drasticamente la alimentacion.

6. CONCLUSION

Resume que es lo mas importante de este producto,
que conviene tener en cuenta y como consumirlo
dentro de una alimentacion variada.

ESTILO:

- Habla directamente con el usuario.
- Utiliza un tono cercano, inteligente y educativo.
- Explica el porque de cada observacion.
- Evita frases genericas.
- No exageres riesgos ni asustes al usuario.
- No diagnostiques enfermedades.
- No des recomendaciones medicas.
- No juzgues al usuario por consumir un producto.
- No afirmes que alguien consume demasiado basandote
  unicamente en una fotografia.

IMPORTANTE:

Si la imagen no contiene una etiqueta nutricional legible,
indica claramente que no es posible realizar un analisis fiable.
"""


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "NutriScan server funcionando"
    })


# =========================================================
# ANALIZAR FOTOGRAFIA
# =========================================================

@app.route("/analyze", methods=["POST"])
def analyze():

    # Comprobar API KEY
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:

        return jsonify({
            "error": "Falta configurar GEMINI_API_KEY"
        }), 500

    # Comprobar imagen
    if "image" not in request.files:

        return jsonify({
            "error": "No se recibio ninguna fotografia"
        }), 400

    photo = request.files["image"]

    # Comprobar formato
    if photo.mimetype not in [
        "image/jpeg",
        "image/png",
        "image/webp"
    ]:

        return jsonify({
            "error": "Formato de imagen no compatible"
        }), 400

    try:

        # Leer imagen
        image_bytes = photo.read()

        if not image_bytes:

            return jsonify({
                "error": "La fotografia esta vacia"
            }), 400

        print(
            "Imagen recibida:",
            len(image_bytes),
            "bytes"
        )

        # Crear cliente Gemini
        client = genai.Client(
            api_key=api_key
        )

        # Intentos de Gemini
        max_attempts = 4

        for attempt in range(1, max_attempts + 1):

            try:

                print(
                    f"Intento {attempt} de {max_attempts} con Gemini..."
                )

                # Llamada a Gemini
                response = client.models.generate_content(

                    model=MODEL,

                    contents=[
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type=photo.mimetype
                        ),
                        PROMPT
                    ],

                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.4
                    )
                )

                print("Gemini respondio correctamente.")

                # Convertir respuesta a JSON
                result = json.loads(response.text)

                # Comprobar estructura
                required_fields = [
                    "productName",
                    "servingSize",
                    "calories",
                    "protein",
                    "carbohydrates",
                    "fats",
                    "sugars",
                    "sodium",
                    "ingredients",
                    "vitamins",
                    "minerals",
                    "analysis"
                ]

                for field in required_fields:

                    if field not in result:

                        raise ValueError(
                            "Falta el campo: " + field
                        )

                print("JSON generado correctamente.")

                return jsonify(result)

            except Exception as error:

                error_text = str(error)

                print(
                    f"Error en intento {attempt}:"
                )

                print(error_text)

                # Identificar errores temporales
                temporary_error = (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                    or "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                    or "500" in error_text
                    or "INTERNAL" in error_text
                )

                # Si no es temporal
                if not temporary_error:

                    print(
                        "El error no parece ser temporal."
                    )

                    return jsonify({
                        "error": "Gemini no pudo analizar la imagen.",
                        "details": error_text
                    }), 500

                # Si se agotaron los intentos
                if attempt == max_attempts:

                    print(
                        "Se agotaron los intentos."
                    )

                    return jsonify({
                        "error":
                            "Gemini esta temporalmente ocupado. "
                            "Intenta nuevamente en unos segundos."
                    }), 503

                # Espera exponencial
                wait_time = 2 ** (attempt - 1)

                print(
                    f"Esperando {wait_time} segundos..."
                )

                time.sleep(wait_time)

    except Exception as error:

        print("Error general del servidor:")
        print(error)

        return jsonify({
            "error": "No se pudo analizar la fotografia.",
            "details": str(error)
        }), 500


# =========================================================
# MANEJO DE ARCHIVOS DEMASIADO GRANDES
# =========================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "error": "La imagen supera el limite de 10 MB."
    }), 413


# =========================================================
# INICIAR SERVIDOR
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
