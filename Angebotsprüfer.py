import os
from pathlib import Path
import json
from typing import Any, Dict, List
import numpy as np

import pandas as pd
from openai import OpenAI


#SYSTEM_PROMPT = """ 
#Du bist ein Datenprüf-Tool für Angebote/Rechnungen.
#
#Analysiere die folgende Angebots-/Rechnungszeilen und finde Fehler/Risiken:\n"
#- Rechenfehler (z.B. Positionen, MwSt, Summe),
#- Inkonsistenzen/Widersprüche,
#- Plausibilität (unrealistische Werte),
#- Zahlungsbedingungen (Vorkasse, extrem kurz),
#- Rechtliche Risiken (Haftung ausgeschlossen, unklare Provision etc.),
#Antworte exakt im angegebenen JSON-Schema:
#
#{
#  "doc_id": "string",
#  "row_index": "integer",
#  "language_issues": [{"field":"string","issue":"string","suggestion":"string"}],
#  "inconsistencies": [{"fields":["string"],"issue":"string","why":"string"}],
#  "plausibility_issues": [{"field":"string","value":"string","issue":"string","why":"string"}],
#  "duplicate_issues": [{"field":"string","issue":"string","finding":"string"}],
#  "severity": "low|medium|high",
#  "summary": "string",
#  "confidence": "low|medium|high"
#}
#
#Gib NUR gültiges JSON zurück. Kein Markdown. Keine Erklärtexte.
#Wenn dir Informationen fehlen: gib trotzdem JSON zurück und markiere das als Issue.\n
#"""

SYSTEM_PROMPT = """
Du bist ein deterministisches Datenprüf- und Risikoanalyse-Tool.

Du analysierst strukturierte Datenzeilen und identifizierst:
- Fehler
- Inkonsistenzen
- Risiken
- Unplausible Werte

Du erfindest keine fehlenden Informationen.
Du interpretierst nur die bereitgestellten Daten.
Du korrigierst keine Inhalte, sondern meldest nur Probleme.

Antworte ausschließlich im angeforderten JSON-Format.
Gib niemals Markdown oder erklärenden Fließtext zurück.
"""

def check_batch_with_llm(rows: list[dict[str, Any]], schema: dict[str, str]) -> dict[str, Any]:
    user_prompt = f"""
    TASK:
    Analysiere die folgenden Angebots-/Rechnungszeilen.

    PRÜFE AUF:
    - Rechenfehler (z.B. Positionen, MwSt, Summe)
    - Inkonsistenzen/Widersprüche
    - Plausibilität (unrealistische Werte)
    - Zahlungsbedingungen (z.B. Vorkasse, extrem kurze Fristen)
    - Rechtliche Risiken (z.B. Haftung ausgeschlossen, unklare Provision)

    Wenn Informationen fehlen, markiere dies als Issue.

    ANTWORT-SCHEMA (STRICT JSON):

    {{
      "doc_id": "string",
      "row_index": "integer",
      "language_issues": [{{"field":"string","issue":"string","suggestion":"string"}}],
      "inconsistencies": [{{"fields":["string"],"issue":"string","why":"string"}}],
      "plausibility_issues": [{{"field":"string","value":"string","issue":"string","why":"string"}}],
      "duplicate_issues": [{{"field":"string","issue":"string","finding":"string"}}],
      "severity": "low|medium|high",
      "summary": "string",
      "confidence": "low|medium|high"
    }}

    DATENSCHEMA:
    {json.dumps(schema, ensure_ascii=False)}

    ZEILEN:
    {json.dumps(rows, ensure_ascii=False)}

    Gib ausschließlich gültiges JSON zurück.
    """


    content = run_system_calls_openai(user_input=json.dumps(user_prompt, ensure_ascii=False))

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Fallback: leere Ergebnisse
        return {"results": []}
    
def run_system_calls_openai(user_input: str) -> str:
    """
    Sendet user_input an OpenAI und gibt den reinen Textinhalt zurück.
    """
    print("Nutzung der OpenAI API:      ->")
    print(":)")
    OPENAI_API_KEY = ""  # <-- NICHT hardcoden; besser als CLI-Arg oder env. (aber du wolltest ohne env)

    client = OpenAI(api_key=OPENAI_API_KEY)
    models = client.models.list()

    #for m in models.data:
    #    print(m.id)
    
    model = "gpt-4.1-mini"

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
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



def load_excel(path: str) -> pd.DataFrame:
    
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.replace([np.nan, np.inf, -np.inf], None)
    
    required_cols = ["doc_id", "text"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in excel: {col}. Found: {list(df.columns)}")
    
    return df

def chunk_rows(df, size=2):
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
        
def build_error_report_batches(df: pd.DataFrame, id_col: str | None = None) -> pd.DataFrame:
    report_rows = []
    max_batches = 1

    # leichte Metainfo fürs LLM: Spalten + simple Typen
    schema = {}
    for c in df.columns:
        schema[c] = str(df[c].dtype)

    # id_col auflösen
    resolved_id_col = id_col if (id_col and id_col in df.columns) else ("doc_id" if "doc_id" in df.columns else df.columns[0])

    # 2) Batches
    for i, batch in enumerate(chunk_rows(df)):
        if i >= max_batches:
            break
        print("heir kommt der batch:",   batch)
        
        ## 1) schnelle Regelchecks (billig, deterministisch)
        #rule_issues = precheck_row_minimal(row_dict)

        # 2) LLM-Checks (Sprache, Inkonsistenz, Plausibilität)
        llm_result = check_batch_with_llm(batch, schema=schema)
        print(":.....   .....:", llm_result)
        # Ergebniszeile im Reportresolved_id_col
        
        for res in llm_result:
            report_rows.append({
                "doc_id": res["doc_id"],
                "row_index": res["row_index"],
                "severity": res["severity"],
                "confidence": res["confidence"],
                "summary": res["summary"],
                "language_issues": json.dumps(res["language_issues"], ensure_ascii=False),
                "inconsistencies": json.dumps(res["inconsistencies"], ensure_ascii=False),
                "plausibility_issues": json.dumps(res["plausibility_issues"], ensure_ascii=False),
                "duplicate_issues": json.dumps(res["duplicate_issues"], ensure_ascii=False),
            })

    return pd.DataFrame(report_rows)

def main():
    
    input_path = "./data_2/offers.xlsx"
    output_path = "./data_2/fehlerbericht_offers_openai.xlsx"

    df = load_excel(input_path)
    report_df = build_error_report_batches(df, id_col=None)
    
    report_df.to_excel(output_path, index=False)
    print(f"Wrote report: {Path(output_path).resolve()}")
    

if __name__ == "__main__":
    main()


    
