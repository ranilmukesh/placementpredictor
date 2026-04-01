"""
PlacementPredictor+ LLM Chat Module
AI-powered career chat using Agno SDK + Nvidia LLM
Context Injection pattern: student data is injected into the system prompt.
"""

import os
import uuid

# Load .env file written by start_placement_predictor.bat
# override=True ensures .env always wins (fixes Windows subprocess env inheritance issues)
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(dotenv_path=env_path, override=True)
    print(f"[LLM] Loaded .env from: {env_path}")
except ImportError:
    print("[LLM] python-dotenv not installed, relying on env var directly")

# Debug: check if key is present
_key = os.environ.get("NVIDIA_API_KEY", "")
if _key:
    print(f"[LLM] NVIDIA_API_KEY found: {_key[:12]}...{_key[-4:]}")
else:
    print("[LLM] WARNING: NVIDIA_API_KEY is NOT set!")

from agno.agent import Agent
from agno.models.nvidia import Nvidia
from agno.db.sqlite import SqliteDb

# Ensure tmp directory exists for chat DB
os.makedirs("tmp", exist_ok=True)

# SQLite for multi-turn chat persistence (exactly as shown in Agno docs)
_chat_db = SqliteDb(db_file="tmp/placement_chat.db")

# In-memory registry: session_id -> Agent instance
_sessions: dict = {}


def _binary_to_yes_no(value) -> str:
    """Map binary-encoded values to dataset semantics: 0 = No, 1 = Yes."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes"}:
            return "Yes"
        if normalized in {"0", "false", "no", ""}:
            return "No"
    try:
        return "Yes" if int(value) == 1 else "No"
    except Exception:
        return "No"


def build_system_context(
    student_data: dict,
    prediction: dict,
    explanation: dict,
    whatif: dict,
) -> str:
    """Build structured context string from student analysis data."""
    
    # Guard against None values
    student_data = student_data or {}
    prediction = prediction or {}
    explanation = explanation or {}
    whatif = whatif or {}

    # ── Student Profile ──
    student_lines = [
        f"Gender: {student_data.get('Gender', 'N/A')}",
        f"Age: {student_data.get('Age', 'N/A')} years",
        f"Stream: {student_data.get('Stream', 'N/A')}",
        f"Internships: {student_data.get('Internships', 'N/A')}",
        f"CGPA: {student_data.get('CGPA', 'N/A')}",
        f"Hostel: {_binary_to_yes_no(student_data.get('Hostel'))}",
        f"History of Backlogs: {_binary_to_yes_no(student_data.get('HistoryOfBacklogs'))}",
        f"Skills: {', '.join(student_data.get('skills', []))}",
        f"Desired Role: {student_data.get('desired_role', 'Not Specified')}"
    ]
    student_block = "\n".join(f"  - {l}" for l in student_lines)

    # ── Prediction ──
    pred_block = (
        f"  - Placement Chance: {prediction.get('probability_percentage', '?')}%\n"
        f"  - Risk Level: {prediction.get('risk_level', '?')}\n"
        f"  - Confidence: {prediction.get('confidence', '?')}\n"
        f"  - Recommended Role: {prediction.get('recommended_job', 'N/A')}"
    )

    # ── SHAP Factors ──
    factors = explanation.get("top_contributing_factors", [])
    factor_lines = []
    for f in factors:
        factor_lines.append(
            f"  - {f.get('feature', '?')} | {f.get('direction', '?')} | "
            f"{f.get('interpretation', '')}"
        )
    shap_block = "\n".join(factor_lines) if factor_lines else "  (None available)"

    # ── What-If Scenarios ──
    scenarios = whatif.get("scenarios", [])
    scenario_lines = []
    for s in scenarios:
        scenario_lines.append(
            f"  - {s.get('title', '?')}: "
            f"{s.get('original_risk', '?')}% -> {s.get('modified_risk', '?')}% "
            f"(delta: {s.get('risk_delta', 0):+.1f}%) | {s.get('description', '')}"
        )
    whatif_block = "\n".join(scenario_lines) if scenario_lines else "  (None generated)"

    combined = whatif.get("combined_risk")
    combined_line = ""
    if combined is not None:
        combined_line = (
            f"\n  BEST COMBINED OUTCOME (all changes): "
            f"{combined}% ({whatif.get('combined_risk_level', '?')})"
        )

    return (
        "=== STUDENT CAREER ASSESSMENT DATA ===\n\n"
        f"STUDENT PROFILE:\n{student_block}\n\n"
        f"PLACEMENT CHANCE PREDICTION:\n{pred_block}\n\n"
        f"TOP CONTRIBUTING FACTORS (SHAP):\n{shap_block}\n\n"
        f"WHAT-IF SCENARIOS:\n{whatif_block}{combined_line}\n\n"
        "======================================"
    )


def start_chat_session(
    patient_data: dict,
    prediction: dict,
    explanation: dict,
    whatif: dict,
) -> tuple:
    """
    Start a new chat session with student context injected.
    Returns (session_id, greeting_message_text).
    """
    session_id = f"pp-{uuid.uuid4().hex[:8]}"
    system_context = build_system_context(patient_data, prediction, explanation, whatif)

    # Create agent following exact Agno SDK pattern from the docs
    agent = Agent(
        model=Nvidia(
                max_tokens=16384,
                temperature=0.1,  # Slightly higher for better reasoning
                top_p=0.95,
                id="minimaxai/minimax-m2.5"
            ),
        # description is added to the START of system message
        description=(
            "You are Placement AI, a professional and encouraging career assistant "
            "built into the PlacementPredictor+ dashboard. "
            "A student has just completed their assessment. "
            "Their full data and results are below.\n\n"
            + system_context
        ),
        # instructions are wrapped in <instructions> tags
        instructions=[
            "Speak in an encouraging, practical, and professional tone.",
            "Reference the student's specific data, target jobs, and skills when answering.",
            "Explain career terms and gaps mapped in plain language.",
            "When discussing What-If scenarios, give practical academic or internship tips.",
            "If placement chances are LOW, be reassuring but give robust constructive criticism.",
            "Always remind them you are an AI, not an official hiring manager.",
            "Keep responses concise (2-3 paragraphs) unless asked for more.",
            "Never fabricate data — only reference the assessment data above.",
            "If the student asks to compare their target job with the recommended job, focus on skill matching between the two."
        ],
        # expected_output is appended to END of system message
        expected_output=(
            "Clear, encouraging, personalized career guidance based on "
            "the student's specific assessment data, with practical interview and upskilling next steps."
        ),
        # Persistence (exactly as shown in Agno docs)
        db=_chat_db,
        add_history_to_context=True,
        num_history_runs=5,
        add_datetime_to_context=True,
    )

    _sessions[session_id] = agent

    # Auto-generate greeting
    greeting_prompt = (
        "The student has just seen their placement chance results. "
        "Introduce yourself in 1 sentence, then give a brief encouraging "
        "summary of their key findings (chance %, top 2 factors or their recommended job). "
        "End by asking what career or academic topic they'd like to discuss. Keep it concise."
    )
    response = agent.run(greeting_prompt, session_id=session_id)
    
    # Error checking
    if "Connection error" in response.content or "404" in response.content or "Unknown model error" in response.content or str(getattr(response, "status", "")).lower() == "error":
        raise ConnectionError(f"LLM API Error: {response.content}")
        
    return session_id, response.content


def get_chat_response(session_id: str, user_message: str) -> str:
    """Get a response in an existing session. Returns response text."""
    agent = _sessions.get(session_id)
    if agent is None:
        raise ValueError(f"Session '{session_id}' not found or expired.")

    response = agent.run(user_message, session_id=session_id)
    
    if "Connection error" in response.content or "404" in response.content or "Unknown model error" in response.content or str(getattr(response, "status", "")).lower() == "error":
        raise ConnectionError(f"LLM API Error: {response.content}")
        
    return response.content