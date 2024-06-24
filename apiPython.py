
from transformers import pipeline
from flask import Flask, request, jsonify

import logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classificador = pipeline("zero-shot-classification", 
                       model="Recognai/bert-base-spanish-wwm-cased-xnli")

@app.route('/clasificar', methods=['POST'])
def clasificar_texto():
    try:
        data = request.get_json()
        texts_to_classify = data.get('texts', [])  # Lista de textos a clasificar
        candidate_labels  = data.get('tags', [])
        classifier_language = data.get('language', '')

        # Lista para almacenar los resultados de la clasificación de cada texto
        results = []
        #app.logger.debug(f"texts: {texts_to_classify}")
        if (classifier_language == 'en'): 
            for text in texts_to_classify:
                resultado = classifier(text, candidate_labels, multi_label=True)
                results.append(resultado)
        elif (classifier_language == 'es'):
            for text in texts_to_classify:
                resultado = classificador(text, candidate_labels, multi_label=True)
                results.append(resultado)

        app.logger.debug(f"result: {resultado}")
        score_filter = 0.5

        # Filtrar los resultados de cada texto
        filtered_results = []
        for result in results:
            filtered_labels = [label for label, score in zip(result['labels'], result['scores']) if score >= score_filter]
            filtered_result = {'labels': filtered_labels, 'scores': [score for score in result['scores'] if score >= score_filter], 'sequence' : result['sequence']}
            filtered_results.append(filtered_result)
        
        
        return jsonify(filtered_results)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
