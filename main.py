from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd

'''Expected entry dictionary
{
    "pclass": int (1/2/3),
    "sex": str (male/female),
    "age": int,
    "embarked": str (S/C/Q)
}
'''

model = joblib.load("decision_tree_model.pkl")
app = FastAPI()

def data_prep(message: dict):
    try:
        df = pd.DataFrame([message])
        df["sex_male"] = 1 if df["sex"][0] == "male" else 0
        df["embarked_Q"] = True if df["embarked"][0] == "Q" else False
        df["embarked_S"] = True if df["embarked"][0] == "S" else False
        df.drop(["embarked", "sex"], axis=1, inplace=True)
        return df

    except Exception:
        raise HTTPException(status_code=400, detail="Error en la preparación de los datos.")

def survived_or_not(message: dict):
    data = data_prep(message)
    label = model.predict(data)[0]
    return {"label": "Survived" if int(label) == 1 else "Not survived"}

@app.get("/")
def main():
    return {"message": "Hola"}

@app.post("/survived-or-not/")
def predict_survived(message: dict):
    required_keys = ["pclass", "sex", "age", "embarked"]
    
    for key in required_keys:
        if key not in message:
            raise HTTPException(status_code=400, detail=f"Falta el parámetro: {key}")
    
    if not isinstance(message['pclass'], int) or message['pclass'] not in [1, 2, 3]:
        raise HTTPException(status_code=400, detail="El parámetro 'pclass' debe ser un entero (1, 2, o 3).")
    
    if not isinstance(message['age'], int) or message['age'] < 0:
        raise HTTPException(status_code=400, detail="El parámetro 'age' debe ser un número entero positivo.")
    
    if message['sex'] not in ['male', 'female']:
        raise HTTPException(status_code=400, detail="El parámetro 'sex' debe ser 'male' o 'female'.")
    
    if message['embarked'] not in ['C', 'Q', 'S']:
        raise HTTPException(status_code=400, detail="El parámetro 'embarked' debe ser 'C', 'Q', o 'S'.")

    model_predict = survived_or_not(message)
    return {"prediction": model_predict}