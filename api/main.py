# api/main.py
# API FastAPI pour SenSante - Assistant pré-diagnostic médical

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
from pathlib import Path
import os
from dotenv import load_dotenv
from groq import Groq

# --- Schemas Pydantic ---

class PatientInput(BaseModel):
    age:         int   = Field(..., ge=0, le=120)
    sexe:        str   = Field(...)
    temperature: float = Field(..., ge=35.0, le=42.0)
    tension_sys: int   = Field(..., ge=60, le=250)
    toux:        bool  = Field(...)
    fatigue:     bool  = Field(...)
    maux_tete:   bool  = Field(...)
    region:      str   = Field(...)

class DiagnosticOutput(BaseModel):
    diagnostic:  str   = Field(...)
    probabilite: float = Field(...)
    confiance:   str   = Field(...)
    message:     str   = Field(...)

class ExplainInput(BaseModel):
    diagnostic:  str   = Field(...)
    probabilite: float = Field(...)
    age:         int   = Field(...)
    sexe:        str   = Field(...)
    temperature: float = Field(...)
    region:      str   = Field(...)

class ExplainOutput(BaseModel):
    explication: str = Field(...)
    modele_llm:  str = Field(default="llama-3.1-8b-instant")


# --- App ---
app = FastAPI(
    title="SenSante API",
    description="Assistant pré-diagnostic médical pour le Sénégal",
    version="0.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Servir le frontend comme fichier statique
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_frontend():
    """Servir la page d'accueil."""
    return FileResponse("frontend/index.html")

# --- Groq ---
load_dotenv()
groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("Client Groq initialisé.")
else:
    print("ATTENTION : GROQ_API_KEY non trouvée. /explain sera désactivé.")

# --- Modèle ML ---
BASE_DIR = Path(__file__).resolve().parent.parent

print("Chargement du modèle...")
model        = joblib.load(BASE_DIR / "models" / "model.pkl")
le_sexe      = joblib.load(BASE_DIR / "models" / "encoder_sexe.pkl")
le_region    = joblib.load(BASE_DIR / "models" / "encoder_region.pkl")
feature_cols = joblib.load(BASE_DIR / "models" / "feature_cols.pkl")

print(f"Modèle chargé : {type(model).__name__}")
print(f"Classes : {list(model.classes_)}")


# --- Routes ---

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "SenSante API is running"}


@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):
    # 1. Encoder les variables catégoriques
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur", probabilite=0.0, confiance="aucune",
            message=f"Sexe invalide : {patient.sexe}. Utiliser M ou F."
        )

    try:
        region_enc = le_region.transform([patient.region])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur", probabilite=0.0, confiance="aucune",
            message=f"Région inconnue : {patient.region}"
        )

    # 2. Construire le vecteur de features
    features = np.array([[
        patient.age,
        sexe_enc,
        patient.temperature,
        patient.tension_sys,
        int(patient.toux),
        int(patient.fatigue),
        int(patient.maux_tete),
        region_enc
    ]])

    # 3. Prédire
    diagnostic = model.predict(features)[0]
    probas     = model.predict_proba(features)[0]
    proba_max  = float(probas.max())

    # 4. Niveau de confiance
    if proba_max >= 0.7:
        confiance = "haute"
    elif proba_max >= 0.4:
        confiance = "moyenne"
    else:
        confiance = "faible"

    # 5. Recommandation
    messages = {
        "paludisme": "Suspicion de paludisme. Consultez un médecin rapidement.",
        "grippe":    "Suspicion de grippe. Repos et hydratation recommandés.",
        "typhoide":  "Suspicion de typhoïde. Consultation médicale nécessaire.",
        "sain":      "Pas de pathologie détectée. Continuez à surveiller."
    }

    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un médecin.")
    )


SYSTEM_PROMPT = """Tu es un assistant médical sénégalais.
Tu reçois un diagnostic et des données patient.
Explique le résultat en français simple,
comme un médecin parlerait à son patient.
Sois rassurant mais recommande toujours une consultation médicale.
Maximum 3 phrases.
Ne fais JAMAIS de diagnostic toi-même.
Tu expliques uniquement le diagnostic fourni."""

@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):
    if not groq_client:
        return ExplainOutput(
            explication="Service d'explication indisponible. Clé API non configurée.",
            modele_llm="aucun"
        )

    user_prompt = (
        f"Patient : {data.sexe}, {data.age} ans, région {data.region}\n"
        f"Température : {data.temperature} C\n"
        f"Diagnostic du modèle : {data.diagnostic} "
        f"(probabilité {data.probabilite:.0%})\n"
        f"Explique ce résultat au patient."
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        explication = response.choices[0].message.content
    except Exception as e:
        explication = f"Erreur lors de l'appel au LLM : {str(e)}"

    return ExplainOutput(explication=explication)