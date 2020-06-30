import shap
import joblib

def get_age(input, model):
    result = model.predict(input)
    return result[0]

def get_shap(input, model):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input)
    explainer = shap.TreeExplainer(model)
    cols = ['Pneumonia', 'Hypertension', 'Chest_pain', 'Respiratory_disorder',
       'Anemia', 'Diabetes', 'Hypoglycemia', 'Fever', 'MACE', 'Abdominal_pain',
       'Pancreatitis', 'MARRIED', 'MALE']
    cols = list(zip(cols, input[0]))
    shap_dict = dict(zip(cols, shap_values[0])) 
    top_keys = sorted(shap_dict, key=lambda dict_key: abs(shap_dict[dict_key]), reverse=True)
    result = []
    for key in top_keys[:4]:
        result.append(list(key) + [shap_dict[key] * 100])
    return result