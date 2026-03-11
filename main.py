import pandas as pd
import numpy as np
import joblib
import shap
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import os
import logging
import sys
from datetime import datetime
from pytz import timezone
from starlette.middleware.base import BaseHTTPMiddleware

# Ensure IST timezone
os.environ['TZ'] = 'Asia/Kolkata'

# Setup IST timezone logging
IST = timezone('Asia/Kolkata')

# Configure logging with IST timezone
class ISTFormatter(logging.Formatter):
    """Custom formatter for IST timestamps"""
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=IST)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.strftime("%Y-%m-%d %H:%M:%S IST")

def setup_logging():
    """Setup logging for all loggers to use IST timestamps"""
    formatter = ISTFormatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add new handlers with custom formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Configure uvicorn loggers
    for logger_name in ['uvicorn', 'uvicorn.error', 'fastapi']:
        lg = logging.getLogger(logger_name)
        lg.setLevel(logging.INFO)
        # Clear existing handlers
        for handler in lg.handlers[:]:
            lg.removeHandler(handler)
        # Use root logger's handlers
        lg.handlers = root_logger.handlers
        lg.propagate = True
    
    # Disable uvicorn access logging (we use our own middleware)
    access_logger = logging.getLogger('uvicorn.access')
    access_logger.setLevel(logging.CRITICAL)  # Suppress access logs
    access_logger.disabled = True
    
    return logging.getLogger(__name__)

logger = setup_logging()

def get_client_ip(request: Request) -> str:
    """Extract client IP from request headers or socket"""
    if 'x-forwarded-for' in request.headers:
        return request.headers['x-forwarded-for'].split(',')[0].strip()
    elif 'x-real-ip' in request.headers:
        return request.headers['x-real-ip']
    else:
        return request.client.host if request.client else "0.0.0.0"

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_ip = get_client_ip(request)
        method = request.method
        path = request.url.path
        logger.info(f"📨 [IP: {client_ip}] {method} {path}")
        
        response = await call_next(request)
        
        logger.info(f"✅ [IP: {client_ip}] {method} {path} → {response.status_code}")
        return response

try:
    from llm import start_chat_session, get_chat_response
    CHAT_AVAILABLE = True
    logger.info("[OK] LLM Chat module loaded")
except Exception as _llm_err:
    CHAT_AVAILABLE = False
    logger.warning(f"[!] LLM Chat unavailable: {_llm_err}")

app = FastAPI(
    title="PlacementPredictor+",
    description="AI-powered placement prediction with explainable insights and career routing",
    version="1.0.0"
)

# Add logging middleware first
app.add_middleware(LoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS_PATH = "placement_artifacts.pkl"

from routing_engine import RoutingEngine

model = None
le_gender = None
le_stream = None
explainer = None
routing_engine = None
shap_model = None
preprocessor = None

def load_artifacts():
    global model, le_gender, le_stream, explainer, routing_engine, shap_model, preprocessor
    if not os.path.exists(ARTIFACTS_PATH):
        raise FileNotFoundError(f"Artifacts file '{ARTIFACTS_PATH}' not found.")
    
    artifacts = joblib.load(ARTIFACTS_PATH)
    model = artifacts['model']
    shap_model = artifacts['shap_model']
    preprocessor = artifacts['preprocessor']
    le_gender = artifacts['le_gender']
    le_stream = artifacts['le_stream']
    routing_engine = artifacts.get('routing_engine')
    explainer = shap.TreeExplainer(shap_model)
    logger.info("[OK] All artifacts loaded successfully!")

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("Application Startup")
    logger.info("=" * 60)
    try:
        load_artifacts()
    except Exception as e:
        logger.error(f"[!] Startup Error: {e}")

class StudentData(BaseModel):
    Age: int = Field(..., ge=15, le=40)
    Gender: str = Field(..., description="Male or Female")
    Stream: str = Field(..., description="Stream of study")
    Internships: int = Field(..., ge=0)
    CGPA: float = Field(..., ge=0, le=10)
    Hostel: int = Field(..., description="1 if hostel, 0 if not")
    HistoryOfBacklogs: int = Field(..., description="1 if backlogs, 0 if not")
    skills: List[str] = Field(default=[], description="List of user skills for routing")
    desired_role: Optional[str] = Field(default=None, description="User's desired job role")

class PredictionResponse(BaseModel):
    prediction: int
    probability_percentage: float
    risk_level: str
    confidence: str
    recommended_job: Optional[str] = None
    missing_skills: Optional[List[str]] = None
    graph_data: Optional[str] = None

class FactorImpact(BaseModel):
    feature: str
    impact: float
    direction: str
    interpretation: str

class ExplainResponse(BaseModel):
    top_contributing_factors: List[FactorImpact]
    base_value: float
    prediction_value: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

class OptionsResponse(BaseModel):
    streams: List[str]
    skills: List[str]
    jobs: List[str]

def prepare_input(data: StudentData) -> pd.DataFrame:
    try:
        g = le_gender.transform([data.Gender])[0]
        s = le_stream.transform([data.Stream])[0]
    except Exception as e:
        # Fallback for unknown
        g = 0
        s = 0
    df = pd.DataFrame([{
        'Age': data.Age,
        'Gender': g,
        'Stream': s,
        'Internships': data.Internships,
        'CGPA': data.CGPA,
        'Hostel': data.Hostel,
        'HistoryOfBacklogs': data.HistoryOfBacklogs
    }])
    # Extract columns naturally based on dict order which matches train features order
    return df

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", model_loaded=model is not None)

@app.get("/options", response_model=OptionsResponse)
async def get_options():
    if le_stream is None or routing_engine is None:
        raise HTTPException(status_code=503, detail="Artifacts not loaded.")
    return OptionsResponse(
        streams=list(le_stream.classes_),
        skills=routing_engine.get_skill_list(),
        jobs=routing_engine.get_job_list()
    )

def get_placement_level(probability: float):
    # Map probability of placement to levels
    # Since we want to reuse the JS/CSS, we map:
    # High chance -> good (represented as "LOW" risk so it shows green in UI)
    # Medium chance -> medium
    # Low chance -> bad (represented as "HIGH" risk so it shows red in UI)
    if probability >= 0.7:
        return "LOW", "Very High Confidence"  # LOW risk of being unplaced -> High chance
    elif probability >= 0.5:
        return "LOW", "High Confidence"
    elif probability >= 0.3:
        return "MEDIUM", "Moderate Confidence"
    else:
        return "HIGH", "High Confidence" # HIGH risk of being unplaced -> Low chance

@app.post("/predict", response_model=PredictionResponse)
async def predict_placement(data: StudentData):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    df = prepare_input(data)
    df_processed = preprocessor.transform(df)
    prediction = int(model.predict(df_processed)[0])
    probability = float(model.predict_proba(df_processed)[0][1])
    risk_level, confidence = get_placement_level(probability)
    
    rec_job = None
    miss_skills = []
    graph_base64 = None
    
    if routing_engine and data.skills:
        # Get machine recommendation
        job, ms = routing_engine.recommend(data.skills)
        if job:
            rec_job = job
            miss_skills = ms
            
        # Graph logic: prioritize desired role if it exists, but show both in gap analysis if possible.
        # Since the UI only has one img slot, we can generate a combined view or stick to desired.
        # The user specifically mentioned suggested vs desired job graph.
        target_graph = data.desired_role if data.desired_role else rec_job
        if target_graph:
            graph_base64 = routing_engine.get_subgraph_figure_base64(target_graph, data.skills)
            
    return PredictionResponse(
        prediction=prediction,
        probability_percentage=round(probability * 100, 2),
        risk_level=risk_level,
        confidence=confidence,
        recommended_job=rec_job,
        missing_skills=miss_skills,
        graph_data=graph_base64
    )

def interpret_feature(feature_name: str, impact: float) -> str:
    clean_name = feature_name.replace('_', ' ').title()
    abs_impact = abs(impact)
    intensity = "significantly" if abs_impact > 0.3 else ("moderately" if abs_impact > 0.1 else "slightly")
    
    if impact > 0:
        return f"{clean_name} {intensity} improves placement chances"
    else:
        return f"{clean_name} {intensity} reduces placement chances"

@app.post("/explain", response_model=ExplainResponse)
async def explain_pred(data: StudentData):
    if model is None or explainer is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    df = prepare_input(data)
    df_processed = preprocessor.transform(df)
    shap_values = explainer.shap_values(df_processed)
    
    if isinstance(shap_values, list):
        vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
    else:
        vals = shap_values[0]
        if len(vals.shape) == 2:
            vals = vals[0]
            
    if hasattr(explainer.expected_value, '__iter__'):
        base_value = float(explainer.expected_value[1]) if len(explainer.expected_value) > 1 else float(explainer.expected_value[0])
    else:
        base_value = float(explainer.expected_value)
        
    feature_impact = list(zip(df.columns, vals.tolist()))
    feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)
    
    top_factors = []
    for feature, impact in feature_impact[:5]:
        top_factors.append(FactorImpact(
            feature=feature,
            impact=round(float(impact), 4),
            direction="Improves Chances" if impact > 0 else "Reduces Chances",
            interpretation=interpret_feature(feature, impact)
        ))
        
    prediction_value = base_value + sum(vals)
    return ExplainResponse(
        top_contributing_factors=top_factors,
        base_value=round(base_value, 4),
        prediction_value=round(float(prediction_value), 4)
    )

# Basic what-if replacing Medical conditions with CGPA/Internships
@app.post("/whatif")
async def whatif_analysis(data: StudentData):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    def _predict(d: dict) -> float:
        df = prepare_input(StudentData(**d))
        df_processed = preprocessor.transform(df)
        return float(model.predict_proba(df_processed)[0][1]) * 100

    orig_dict = data.model_dump()
    orig_risk = _predict(orig_dict)
    orig_level, _ = get_placement_level(orig_risk / 100)
    
    scenarios = []
    sid = 1
    
    if routing_engine and data.skills and data.desired_role:
        rec_job, _ = routing_engine.recommend(data.skills)
        if rec_job:
            transition = routing_engine.get_career_transition_path(rec_job, data.desired_role)
            if transition and transition.get("skills_to_learn"):
                skills_list = ", ".join(transition["skills_to_learn"][:3])
                scenarios.append({
                    "scenario_id": sid, "title": "Learn Missing Skills", "description": "What if you learned skills for your desired role?",
                    "change_summary": f"Learn: {skills_list}",
                    "original_risk": orig_risk, "modified_risk": orig_risk,
                    "risk_delta": 0,
                    "risk_reduction_percent": 0,
                    "icon": "🧠", "factor_changed": "Skills",
                    "original_value": "Current Skills", "suggested_value": skills_list
                })
                sid+=1

    # Add Internships
    if data.Internships < 3:
        mod = orig_dict.copy()
        mod['Internships'] += 1
        mod_risk = _predict(mod)
        scenarios.append({
            "scenario_id": sid, "title": "Extra Internship", "description": "What if you did one more internship?",
            "change_summary": f"Internships: {data.Internships} → {mod['Internships']}",
            "original_risk": orig_risk, "modified_risk": mod_risk,
            "risk_delta": orig_risk - mod_risk,
            "risk_reduction_percent": ((mod_risk - orig_risk)/orig_risk*100) if orig_risk>0 else 0,
            "icon": "💼", "factor_changed": "Internships",
            "original_value": str(data.Internships), "suggested_value": str(mod['Internships'])
        })
        sid+=1
        
    # Clear Backlogs
    if data.HistoryOfBacklogs == 1:
        mod = orig_dict.copy()
        mod['HistoryOfBacklogs'] = 0
        mod_risk = _predict(mod)
        scenarios.append({
            "scenario_id": sid, "title": "Clear Backlogs", "description": "What if you had no history of backlogs?",
            "change_summary": "Backlogs: Yes → No",
            "original_risk": orig_risk, "modified_risk": mod_risk,
            "risk_delta": orig_risk - mod_risk,
            "risk_reduction_percent": ((mod_risk - orig_risk)/orig_risk*100) if orig_risk>0 else 0,
            "icon": "✅", "factor_changed": "HistoryOfBacklogs",
            "original_value": "Yes", "suggested_value": "No"
        })
        sid+=1
        
    # Change Stream (Example: If not CS, what if CS?)
    if data.Stream != "Computer Science" and "Computer Science" in le_stream.classes_:
        mod = orig_dict.copy()
        mod['Stream'] = "Computer Science"
        mod_risk = _predict(mod)
        if (orig_risk - mod_risk) < 0: # Only suggest if it improves chance
            scenarios.append({
                "scenario_id": sid, "title": "Switch to CS", "description": "What if you switched your stream to Computer Science?",
                "change_summary": f"Stream: {data.Stream} → Computer Science",
                "original_risk": orig_risk, "modified_risk": mod_risk,
                "risk_delta": orig_risk - mod_risk,
                "risk_reduction_percent": ((mod_risk - orig_risk)/orig_risk*100) if orig_risk>0 else 0,
                "icon": "💻", "factor_changed": "Stream",
                "original_value": data.Stream, "suggested_value": "Computer Science"
            })
            sid+=1
            
    # Stay in Hostel
    if data.Hostel == 0:
        mod = orig_dict.copy()
        mod['Hostel'] = 1
        mod_risk = _predict(mod)
        if (orig_risk - mod_risk) < 0: # Only suggest if it improves chance
            scenarios.append({
                "scenario_id": sid, "title": "Stay in Hostel", "description": "What if you stayed in a hostel? (More peer interactions)",
                "change_summary": "Hostel: No → Yes",
                "original_risk": orig_risk, "modified_risk": mod_risk,
                "risk_delta": orig_risk - mod_risk,
                "risk_reduction_percent": ((mod_risk - orig_risk)/orig_risk*100) if orig_risk>0 else 0,
                "icon": "🏢", "factor_changed": "Hostel",
                "original_value": "No", "suggested_value": "Yes"
            })
            sid+=1

    import math
    for s in scenarios:
        # Invert delta so positive = 'improvement' for the JS UI
        s['risk_delta'] = s['modified_risk'] - s['original_risk']
        
    scenarios.sort(key=lambda x: x['risk_delta'], reverse=True)
    
    # Calculate TRUE COMBINED risk by applying all positive changes together
    best_case = orig_dict.copy()
    if data.Internships < 3: best_case['Internships'] += 1
    if data.HistoryOfBacklogs == 1: best_case['HistoryOfBacklogs'] = 0
    
    # Optional stream switch if it was a scenario that improved things
    if data.Stream != "Computer Science" and "Computer Science" in le_stream.classes_:
        # re-check if it's better
        if _predict(best_case.copy()) < _predict({**best_case, 'Stream': 'Computer Science'}):
             best_case['Stream'] = 'Computer Science'

    combined_risk = _predict(best_case)
    combined_level, _ = get_placement_level(combined_risk / 100)

    return {
        "original_risk": orig_risk,
        "original_risk_level": orig_level,
        "scenarios": scenarios,
        "best_scenario": scenarios[0] if scenarios else None,
        "combined_risk": combined_risk, 
        "combined_risk_level": combined_level
    }

@app.post("/chat/start")
async def chat_s(req: dict):
    if not CHAT_AVAILABLE:
        raise HTTPException(status_code=503)
    sid, greet = start_chat_session(req.get('patient_data'), req.get('prediction'), req.get('explanation'), req.get('whatif'))
    return {"session_id": sid, "message": greet}

@app.post("/chat/message")
async def chat_m(req: dict):
    if not CHAT_AVAILABLE:
        raise HTTPException(status_code=503)
    resp = get_chat_response(req.get('session_id'), req.get('message'))
    return {"response": resp}

# --- Static File Serving for Frontend ---
# 1. Mount styles.css and script.js so they can be loaded by index.html automatically
app.mount("/static", StaticFiles(directory="."), name="static")

# 2. Serve index.html at root
@app.get("/")
async def read_index():
    return FileResponse("index.html")

# 3. Handle favicon if needed
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("favicon.ico")

if __name__ == "__main__":
    import uvicorn
    # Important for HF: Host 0.0.0.0 and Port 7860
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
