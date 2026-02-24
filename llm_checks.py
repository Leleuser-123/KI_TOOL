# llm_checks.py
from __future__ import annotations
import json
import os
from typing import Any

from google import genai
from google.genai import types
from openai import OpenAI
from typing import Any, Dict


SYSTEM_PROMPT = """
Du bist ein QA-Assistent für Produktdaten.

Du bekommst eine Produktzeile als JSON.

Prüfe Zeile für Zeile genau und innerhalb einer Zeile Spalte für Spalte:
1) Sprachfehler (Grammatikalische Fehler z.B. Groß und Kleinschreibung, Pluralform etc.) in Textfeldern (z.B Beschreibung etc.)
2) Inkonsistenzen (sowohl auf die spalte, als auch auf Inkonsistenzen zwischen den Spalten)
3) Plausibilität (Sind die Werte umplausibel/unrealistisch für den Produktkontext). Analyiere jede Spaltenangaben auf die Sinnhaftigkeit im Kontext des Produkts. Achte besonders auf die Spalten die für die Unterscheidung Non-food und food entscheidend sind (Beispiel "MHD" bei Non-Food oder Energieangaben bei Food).
4)Achte auf Duplikate zwischen verschiedenen Zeilen der Identifikationsspalte (z.B. Produkt_ID)

Antworte ausschließlich als JSON mit folgendem Format:

{
 "product_id": "...",
 "row_index": "...",
 "language_issues":[{"field":"...","issue":"...","suggestion":"..."}],
 "inconsistencies":[{"fields":["..."],"issue":"...","why":"..."}],
 "plausibility_issues":[{"field":"...","value":"...","issue":"...","why":"..."}],
 "Duplicate_issues":[{"field": "...", "issue": "....", "finding": "..."}],
 "severity":"low|medium|high",
 "summary":"..."
 "convidence":"..."
}

Keine zusätzlichen Texte.
    """
# placeholder prechecks for mninimizing token use.
#TODO: Implement: check for unique elements in identifying colums, negative decleared prices, etc.





def check_row_with_llm(row_dict: dict[str, Any], schema: dict[str, str]) -> dict[str, Any]:
    user_prompt = {
        "task": "check_product_row",
        "schema": schema,
        "row": row_dict,
    }

    
    #content_gemini = run_system_calls_gemini(user_input=json.dumps(user_prompt, ensure_ascii=False))
    content_openai = run_system_calls_openai(user_input=json.dumps(user_prompt, ensure_ascii=False)) 
    # Robust: JSON extrahieren/validieren
    try:
        data = json.loads(content_openai)
    except json.JSONDecodeError:
        # Fallback, falls das Modell Mist baut:
        data = {
            "language_issues": [],
            "inconsistencies": [],
            "plausibility_issues": [],
            "severity": "medium",
            "summary": f"LLM output was not valid JSON. Raw: {content[:500]}"
        }

    # Minimal defaults
    data.setdefault("analysis_steps", [])
    data.setdefault("language_issues", [])
    data.setdefault("inconsistencies", [])
    data.setdefault("plausibility_issues", [])
    data.setdefault("severity", "low")
    data.setdefault("summary", "")

    return data


def check_batch_with_llm(rows: list[dict[str, Any]], schema: dict[str, str], id_col: str) -> dict[str, Any]:
    user_promt = {
        "task": "check_product_rows_batch",
        "id_col": id_col,
        "schema": schema,
        "rows": rows,
    }

    content = run_system_calls_openai(user_input=json.dumps(user_promt, ensure_ascii=False))

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Fallback: leere Ergebnisse
        return {"results": []}



def run_system_calls_gemini(user_input):

    print("Nutzung der gemini API:      ->")
    print(":)")
    
    GEMINI_API_KEY = ""
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    model = "models/gemini-2.5-flash"

    try:
        resp = client.models.generate_content(
            model=model,
            contents=user_input,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json"
                )
            
        )
        
        if resp is not None:
            content = resp.candidates[0].content.parts[0].text
            return content
        
    except Exception as e:
        raise RuntimeError(f"system was not responding: {e}") from e





def run_system_calls_openai(user_input: str) -> str:
    """
    Sendet user_input an OpenAI und gibt den reinen Textinhalt zurück.
    """
    print("Nutzung der OpenAI API:      ->")
    print(":)")
    OPENAI_API_KEY = ""
    client = OpenAI(api_key=OPENAI_API_KEY)
    models = client.models.list()

    #for m in models.data:
    #    print(m.id)
    
    model = "gpt-5.2"

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": user_input},
                {"role": "user", "content": SYSTEM_PROMPT},
            ],
            temperature=0.0,
            #reasoning={"effort": "medium"}
        )

        # Chat Completions: Text sitzt typischerweise hier.
        content = resp.choices[0].message.content or ""
        return content

    except Exception as e:
        # Wir werfen eine RuntimeError mit Original-Fehler.
        raise RuntimeError(f"OpenAI system call failed: {e}") from e
    
    










#def issues_json_to_rows(issues: dict, row_idx: int, product_id=None) -> list[dict]:
#    """
#    Wandelt das LLM-Issue-JSON in eine Liste flacher Report-Zeilen um.
#    Eine Zeile pro Issue.
#    """
#    rows = []
#    severity = issues.get("severity")
#    summary = issues.get("summary")
#
#    # 1) language_issues
#    for it in issues.get("language_issues", []) or []:
#        rows.append({
#            "row_idx": row_idx,
#            "product_id": product_id,
#            "type": "language",
#            "field": it.get("field"),
#            "fields": None,
#            "value": None,
#            "issue": it.get("issue"),
#            "suggestion": it.get("suggestion"),
#            "why": None,
#            "severity": severity,
#            "summary": summary,
#        })
#
#    # 2) inconsistencies
#    for it in issues.get("inconsistencies", []) or []:
#        rows.append({
#            "row_idx": row_idx,
#            "product_id": product_id,
#            "type": "inconsistency",
#            "field": None,
#            "fields": ", ".join(it.get("fields", []) or []),
#            "value": None,
#            "issue": it.get("issue"),
#            "suggestion": None,
#            "why": it.get("why"),
#            "severity": severity,
#            "summary": summary,
#        })
#
#    # 3) plausibility_issues
#    for it in issues.get("plausibility_issues", []) or []:
#        rows.append({
#            "row_idx": row_idx,
#            "product_id": product_id,
#            "type": "plausibility",
#            "field": it.get("field"),
#            "fields": None,
#            "value": it.get("value"),
#            "issue": it.get("issue"),
#            "suggestion": None,
#            "why": it.get("why"),
#            "severity": severity,
#            "summary": summary,
#        })
#
#    return rows




#def build_prompt_for_row(row: Dict[str, Any], schema: Dict) -> str:
#    """
#    Erzeugt einen Prompt für eine einzelne Produktzeile.
#    Liefere ein klares JSON-Schema in der Antwort an.
#    """
#    output_schema = {
#  "product_id": "...",
#  "language_issues":[{"field":"string","issue":"string","suggestion":"string"}],
#  "inconsistencies":[{"fields":["string"],"issue":"string","why":"string"}],
#  "plausibility_issues":[{"field":"string","value":"any","issue":"string","why":"string"}],
#  "duplicate_issues":[{"field":"string","issue":"string","finding":"string"}],
#  "severity":"low|medium|high",
#  "summary":"string"
#}
#    
#    payload = {
#        "task": "check_product_row",
#        "id_col": "Prdoukt_ID",
#        "schma": schema,
#        "row": row,
#    }
#    payload_json = json.dumps(payload, ensure_ascii=False)
#    print(payload_json)
#    prompt = (
#    "Du bist ein Datenprüf-Tool für Produktdaten. Du bekommst ein einzelnes Produkt als JSON:\n"
#    f"{payload_json}\n\n"
#    "Antworte ausschließlich als JSON mit folgendem Format:\n"
#    f"{json.dumps(output_schema, ensure_ascii=False, indent=2)}\n"
#    "Keine zusätzlichen Texte.\n"
#    )
#    return prompt
#
#def call_openai_chat(prompt: str) -> Dict[str, Any]:
#    """
#    Ruft die ChatCompletion / Responses-API auf und gibt das geparste JSON zurück.
#    Fehlerbehandlung sorgt dafür, dass wir wenigstens ein strukturiertes Ergebnis zurückbekommen.
#    """
#    
#    OPENAI_API_KEY = ""
#    client = OpenAI(api_key=OPENAI_API_KEY)
#    model = "gpt-4.1-mini"
#    try:
#        response = client.chat.completions.create(
#            model=model,
#            messages=[{"role": "system", "content": "Du bist ein hilfreiches Tool zur Datenvalidierung."},
#                      {"role": "user", "content": prompt}],
#            temperature=0.0,
#        )
#        text = response.choices[0].message.content.strip()
#        parsed = json.loads(text)
#        return parsed
#    except Exception as e:
#        return {
#            "product_id": None,
#            "language_errors": [f"AI_CALL_ERROR: {str(e)}"],
#            "inconsistencies": [],
#            "plausibility_issues": [],
#            "confidence": "low"}