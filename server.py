
import os
import json
from flask import Flask, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

MODEL = "gemini-3.8-flash"

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "NutriScan server funcionando"})

@app.route("/analyze", methods=["POST"])
def analyze():

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return jsonify({"error": "Falta configurar GEMINI_API_KEY"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No se recibió ninguna fotografía"}), 400

    photo = request.files["image"]

    if photo.mimetype not in ["image/jpeg", "image/png", "image/webp"]:
        return jsonify({"error": "Formato de imagen no compatible"}), 400

    try:
        image_bytes = photo.read()

        client = genai.Client(api_key=api_key)

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
        - Si no puedes identificar un dato, utiliza 0 como valor numérico
          y aclara en analysis que no se pudo identificar.
        - Usa gramos para proteínas, carbohidratos, grasas y azúcares.
        - Usa miligramos para sodio.
        - Las vitaminas y minerales deben tener esta estructura:
          {"name": "nombre", "amount": 0, "unit": "unidad"}.
        - No hagas recomendaciones médicas.
        - Si la imagen no contiene una etiqueta nutricional legible,
          indícalo claramente en analysis.
        """

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

        result = json.loads(response.text)

        return jsonify(result)

    except Exception as error:
        print("Error:", error)
        return jsonify({"error": "No se pudo analizar la fotografía"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
