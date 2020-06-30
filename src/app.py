from flask import Flask, escape, request
import joblib
from model import get_age, get_shap
app = Flask(__name__)

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
    pred_age = get_age(req_data, model)
    shap = get_shap(req_data, model)
    return {'pred_age': pred_age, 'shap': shap}, 200

if __name__ == '__main__':
    model = joblib.load('/app/model.pkl')
    app.run(debug=False, port=80, host='0.0.0.0')
