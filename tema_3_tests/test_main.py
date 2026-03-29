import os
import sys
import pytest
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://localhost:8000"


def get_judge_client():
    api_key = os.environ.get("GROQ_API_KEY")
    base_url = os.environ.get("GROQ_BASE_URL")

    if not api_key:
        pytest.skip("Lipsește GROQ_API_KEY pentru LLM as a Judge.")

    return OpenAI(api_key=api_key, base_url=base_url)


def judge_response(user_question: str, assistant_answer: str, expected_behavior: str) -> str:
    client = get_judge_client()

    judge_prompt = f"""
Ești evaluator pentru un asistent AI specializat EXCLUSIV în vânzarea de betoane.

Trebuie să evaluezi răspunsul asistentului.

Întrebarea utilizatorului:
{user_question}

Răspunsul asistentului:
{assistant_answer}

Comportamentul așteptat:
{expected_behavior}

Răspunde STRICT cu unul dintre aceste două cuvinte:
PASS
FAIL
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "Ești un evaluator strict de teste automate."},
            {"role": "user", "content": judge_prompt},
        ],
    )

    return (response.choices[0].message.content or "").strip().upper()


def test_root_endpoint():
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == "Salut, RAG Assistant ruleaza!"


def test_chat_positive_llm_judge():
    payload = {
        "message": "Cum raspund unui client care cere discount pentru beton C25/30?"
    }

    response = requests.post(f"{BASE_URL}/chat/", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "response" in data
    assistant_answer = data["response"]

    verdict = judge_response(
        user_question=payload["message"],
        assistant_answer=assistant_answer,
        expected_behavior=(
            "Răspunsul trebuie să fie despre vânzarea de betoane, "
            "să ofere o recomandare relevantă și să nu iasă din domeniul cerut."
        ),
    )

    assert verdict == "PASS"


def test_chat_negative_llm_judge():
    payload = {
        "message": "Ce este o pisica?"
    }

    response = requests.post(f"{BASE_URL}/chat/", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "response" in data
    assistant_answer = data["response"]

    verdict = judge_response(
        user_question=payload["message"],
        assistant_answer=assistant_answer,
        expected_behavior=(
            "Asistentul trebuie să refuze politicos deoarece întrebarea nu este "
            "despre vânzarea de betoane."
        ),
    )

    assert verdict == "PASS"
