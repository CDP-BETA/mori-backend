from flask import Flask, escape, request
from flask_cors import CORS
import joblib

from model import get_age, get_shap

app = Flask(__name__)
CORS(app, resources={r'*': {'origins': ['http://localhost:8080', 'https://memento-mori-zzerjae.endpoint.ainize.ai/']}})

known_keys = [
    'pneumonia',
    'hypertension',
    'chest_pain',
    'respiratory_disorder',
    'anemia',
    'diabetes',
    'hypoglycemia',
    'fever',
    'mace',
    'abdominal_pain',
    'pancreatitis',
    'married',
    'male'
]

desease = {
    "pneumonia": "폐 관련 질병이",
    "hypertension": "고혈압이",
    "chest_pain": "가슴 통증이",
    "respiratory_disorder": "호흡기 질환이",
    "anemia": "빈혈이",
    "diabetes": "당뇨병이",
    "hypoglycemia": "저혈당이",
    "fever": "열이",
    "mace": "심혈관 질환이",
    "abdominal_pain": "복부 통증이",
    "pancreatitis": "췌장염이",
}

@app.route('/predict', methods=['POST'])
def predict():
    req = request.get_json()
    if req is None or "answer" not in req:
        return {'error': "answer is missing"}, 400
    answer = req['answer']

    req_data = []
    for key in known_keys:
        if key not in answer:
            err = '%s is missing' % key
            return {'error': err}, 400
        req_data.append(answer[key])

    pred_age = get_age([req_data], model)
    shaps = get_shap(req_data, model)

    cause = []    for shap in shaps:

    return {'pred_age': round(pred_age, 1), 'shap': shaps}, 200

if __name__ == '__main__':
    model = joblib.load('/app/model.pkl')
    app.run(debug=False, port=80, host='0.0.0.0')
