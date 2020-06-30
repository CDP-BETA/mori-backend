from flask import Flask, escape, request
import joblib

app = Flask(__name__)

known_keys = [
    'sex',
    'pneumonia',
    'high_blood_pressure',
    'chest_pain',
    'respiratory_failure',
    'anemia',
    'diabetes',
    'hypoglycemia',
    'fever',
    'cardiovascular_disease',
    'abdominal_pain',
    'pancreatitis',
    'spouse',
]

@app.route('/')
def hello():
    name = request.args.get('name', 'World')
    return f'Hello, {escape(name)}!'

@app.route('/predict', methods=['POST'])
def predict():
    req = request.get_json()
    if req is None or "answer" not in req:
        return {'error': "answer is missing"}, 400
    
    answer = req['answer']

    input = []
    for key in known_keys:
        if key not in answer:
            err = '%s is missing' % key
            return {'error': err}, 400
        input.append(answer[key])
    
    return {'input': input}, 200

if __name__ == '__main__':
  clf = joblib.load('/app/model.pkl')
  app.run(debug=False, port=80, host='0.0.0.0')
