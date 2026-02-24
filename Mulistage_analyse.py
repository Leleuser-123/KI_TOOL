import json
from typing import Any, Dict, List
from collections import defaultdict
from openai import OpenAI
import pandas as pd
import numpy as np
from pathlib import Path

MODEL = "gpt-4.1-mini"
client = OpenAI(api_key="")


# ==========================================================
# ------------------ LLM CORE CALL -------------------------
# ==========================================================

def call_llm(system_prompt: str, user_prompt: str) -> dict:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = resp.choices[0].message.content or ""
        json_content = json.loads(content)
        return json_content

    except Exception as e:
        print("LLM ERROR:", e)
        return {}


# ==========================================================
# ------------------ STAGE 0 -------------------------------
# Deterministic Hard Checks
# ==========================================================

def stage_0_hard_checks(rows: List[Dict[str, Any]], id_col: str):
    issues = defaultdict(lambda: {"duplicate_issues": [], "hard_issues": []})

    # Duplicate detection
    seen = defaultdict(list)
    for idx, row in enumerate(rows):
        seen[row.get(id_col)].append(idx)

    for product_id, indices in seen.items():
        if len(indices) > 1:
            for i in indices:
                issues[i]["duplicate_issues"].append({
                    "field": id_col,
                    "issue": "Duplicate ID",
                    "finding": f"Product ID {product_id} appears multiple times"
                })

    # Simple numeric checks
    for idx, row in enumerate(rows):
        if "Preis_EUR" in row and isinstance(row["Preis_EUR"], (int, float)):
            if row["Preis_EUR"] < 0:
                issues[idx]["hard_issues"].append({
                    "field": "price",
                    "issue": "Negative price"
                })

    return issues


# ==========================================================
# ------------------ STAGE 1 -------------------------------
# Language Check
# ==========================================================

def stage_1_language(row: Dict[str, Any], row_index: int, schema):
    system_prompt = """
Du bist ein deterministischer QA-Assistent für strukturierte Produktdaten.

Du analysierst ausschließlich die bereitgestellten JSON-Daten auf Grammatik und Rechtschreibung, sowie Duplikate.
Du erfindest keine fehlenden Informationen.
Du korrigierst keine Inhalte, sondern meldest nur Probleme.
Du arbeitest zeilenweise und feldweise.

Du gibst ausschließlich gültiges JSON zurück.
Kein Markdown.
Keine Erklärtexte außerhalb des JSON.
"""

    user_prompt = f"""
    TASK:
    Prüfe die folgenden Produktzeilen systematisch.

    PRÜFKRITERIEN:

    1) Sprachfehler
       - Grammatik
       - Groß-/Kleinschreibung
       - falsche Pluralformen
       - fehlerhafte Formulierungen in Textfeldern (z.B. Beschreibung) 

    2) Duplikate
       - doppelte Werte in der Identifikationsspalte: "SKU"

    Falls Informationen fehlen, kennzeichne dies als Issue.

    ANTWORT-SCHEMA (STRICT JSON):

    {{
     "product_id": "string",
     "row_index": "integer",
     "language_issues":[{{"field":"string","issue":"string","suggestion":"string"}}],
     "duplicate_issues":[{{"field":"string","issue":"string","finding":"string"}}],
     "severity":"low|medium|high",
     "summary":"string",
     "confidence":0-100
    }}

    DATENSCHEMA:
    {json.dumps(schema, ensure_ascii=False)}

    PRODUKTZEILEN:
    {json.dumps(row, ensure_ascii=False)}

    Gib ausschließlich gültiges JSON zurück.
    """

    return call_llm(system_prompt, user_prompt)


# ==========================================================
# ------------------ STAGE 2 -------------------------------
# Inconsistency Check
# ==========================================================

def stage_2_inconsistency(row: Dict[str, Any], row_index: int, schema):
    system_prompt = """
Du bist ein deterministischer QA-Assistent für strukturierte Produktdaten.

Du analysierst ausschließlich die bereitgestellten JSON-Daten auf Inkonsistenzen. Dabei sollen KEINE Rechtschreibfehler mehr berücksichtig werden.
Du erfindest keine fehlenden Informationen.
Du korrigierst keine Inhalte, sondern meldest nur Probleme.
Du arbeitest zeilenweise und feldweise.

Du gibst ausschließlich gültiges JSON zurück.
Kein Markdown.
Keine Erklärtexte außerhalb des JSON.
"""

    user_prompt = f"""
    TASK:
    Prüfe die folgenden Produktzeilen systematisch.

    PRÜFKRITERIEN:
    1) Inkonsistenzen
       - innerhalb einer Spalte
       - zwischen mehreren Spalten

    Falls Informationen fehlen, kennzeichne dies als Issue.

    ANTWORT-SCHEMA (STRICT JSON):

    {{
     "product_id": "string",
     "row_index": "integer",
     "inconsistencies":[{{"fields":["string"],"issue":"string","why":"string"}}],
     "severity":"low|medium|high",
     "summary":"string",
     "confidence":0-100
    }}

    DATENSCHEMA:
    {json.dumps(schema, ensure_ascii=False)}

    PRODUKTZEILEN:
    {json.dumps(row, ensure_ascii=False)}

    Gib ausschließlich gültiges JSON zurück.
    """

    return call_llm(system_prompt, user_prompt)


# ==========================================================
# ------------------ STAGE 3 -------------------------------
# Plausibility Check
# ==========================================================

def stage_3_plausibility(row: Dict[str, Any], row_index: int, schema):
    system_prompt = """
Du bist ein deterministischer QA-Assistent für strukturierte Produktdaten.

Du analysierst ausschließlich die bereitgestellten JSON-Daten auf Plausibilitäten zwischen verschiedenen Spalten.
Du erfindest keine fehlenden Informationen.
Du korrigierst keine Inhalte, sondern meldest nur Probleme.
Du arbeitest zeilenweise und feldweise.

Du gibst ausschließlich gültiges JSON zurück.
Kein Markdown.
Keine Erklärtexte außerhalb des JSON.
"""

    user_prompt = f"""
    TASK:
    Prüfe die folgenden Produktzeilen systematisch.

    PRÜFKRITERIEN:

    1) Plausibilität
       - unrealistische oder unlogische Werte im Produktkontext
       - besondere Prüfung bei Unterscheidung Food vs Non-Food
         (z.B. MHD bei Non-Food oder fehlende Energieangaben bei Food)

    Falls Informationen fehlen, kennzeichne dies als Issue.

    ANTWORT-SCHEMA (STRICT JSON):

    {{
     "product_id": "string",
     "row_index": "integer",
     "inconsistencies":[{{"fields":["string"],"issue":"string","why":"string"}}],
     "severity":"low|medium|high",
     "summary":"string",
     "confidence":0-100
    }}

    DATENSCHEMA:
    {json.dumps(schema, ensure_ascii=False)}

    PRODUKTZEILEN:
    {json.dumps(row, ensure_ascii=False)}

    Gib ausschließlich gültiges JSON zurück.
    """

    return call_llm(system_prompt, user_prompt)


# ==========================================================
# ------------------ STAGE 4 -------------------------------
# Aggregation (Python-based)
# ==========================================================

def aggregate_results(row, row_index, hard, lang, incons, plaus):
    
    language_issues = lang.get("language_issues", [])
    duplicate_issues_llm = lang.get("duplicate_issues", [])
    inconsistencies = incons.get("inconsistencies", [])
    plausibility_issues = plaus.get("plausibility_issues", [])
    hard_duplicate = hard.get("duplicate_issues", [])
    hard_issues = hard.get("hard_issues", [])

    all_duplicates = hard_duplicate + duplicate_issues_llm

    total_issues = (
        len(language_issues) +
        len(inconsistencies) +
        len(plausibility_issues) +
        len(all_duplicates) +
        len(hard_issues)
    )

    result = {
        "product_id": row.get("row_data", {}).get("Produkt_ID"),
        "row_index": row_index,
        "language_issues": language_issues,
        "inconsistencies": inconsistencies,
        "plausibility_issues": plausibility_issues,
        "duplicate_issues": all_duplicates,
        "hard_issues": hard_issues,
        "severity": "low",
        "summary": "",
        "confidence": 100,
        "total_issue": total_issues
    }

    if total_issues == 0:
        result["severity"] = "low"
        result["summary"] = "No issues detected."
        result["confidence"] = 100

    elif total_issues < 3:
        result["severity"] = "medium"
        result["summary"] = "Minor issues detected."
        result["confidence"] = 80

    else:
        result["severity"] = "high"
        result["summary"] = "Multiple significant issues detected."
        result["confidence"] = 60

    return result


# ==========================================================
# ------------------ MASTER PIPELINE -----------------------
# ==========================================================

def multistage_analysis(rows: List[Dict[str, Any]], id_col: str, schema: dict):
    
    
    hard_issues = stage_0_hard_checks(rows, id_col)
    final_results = []

    for idx, row in enumerate(rows):
        lang = stage_1_language(row, idx, schema)
        incons = stage_2_inconsistency(row, idx, schema)
        plaus = stage_3_plausibility(row, idx, schema)

        aggregated = aggregate_results(
            row,
            idx,
            hard_issues.get(idx, {}),
            lang,
            incons,
            plaus
        )

        final_results.append(aggregated)

    return pd.DataFrame(final_results) 

def load_excel(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.replace([np.nan, np.inf, -np.inf], None)
    
    return df 

def generate_batch(df, size=1):
    batch = []
    for idx, row in df.iterrows():
        batch.append({
            "row_index": idx + 2,
            "row_data": row.to_dict()
        })
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
        
def generate_schema(df):
    schema = {}
    for c in df.columns:
        schema[c] = str(df[c].dtype)
    return schema

def main():
    input_path = "./data_2/kaufhaus_produktdaten_test_100.xlsx"
    output_path = "./data_2/multistage_errorreport.xlsx"
    
    df = load_excel(input_path)
    schema = generate_schema(df)
    aggregate_rows = []
    for i, batch in enumerate(generate_batch(df)):
        if i >= 1:
            break
        aggregate_rows = batch
        
    final_results = multistage_analysis(aggregate_rows, "Produkt_ID", schema)
    
    final_results.to_excel(output_path, index=False)
    print(f"Wrote report: {Path(output_path).resolve()}")
    
     
    

if __name__ == "__main__":
    main()
    