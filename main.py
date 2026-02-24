# main.py
from pathlib import Path
import pandas as pd
import sys
import numpy as np
import time


from llm_checks import check_row_with_llm, check_batch_with_llm



def load_xlsx(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    # Optional: Spaltennamen normalisieren (Trim)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.replace([np.nan, np.inf, -np.inf], None)
    #print(df)
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
    resolved_id_col = id_col if (id_col and id_col in df.columns) else ("Produkt_ID" if "Produkt_ID" in df.columns else df.columns[0])

    # 2) Batches
    for i, batch in enumerate(chunk_rows(df)):
        if i >= max_batches:
            break
        
        ## 1) schnelle Regelchecks (billig, deterministisch)
        #rule_issues = precheck_row_minimal(row_dict)

        # 2) LLM-Checks (Sprache, Inkonsistenz, Plausibilität)
        llm_result = check_batch_with_llm(batch, schema=schema, id_col=resolved_id_col)
        print(llm_result)

        # Ergebniszeile im Report
        for res in llm_result:
            report_rows.append({
                "batch_index": i,
                "row_index": res["row_index"], 
                "product_id": res["product_id"],
                "llm_language_issues": res["language_issues"],
                "llm_inconsistencies": res["inconsistencies"],
                "llm_plausibility_issues": res["plausibility_issues"],
                "llm_duplicates": res["Duplicate_issues"],
                "llm_summary": res["summary"],
                "llm_severity": res["severity"],
            })

    return pd.DataFrame(report_rows)



def build_error_report(df: pd.DataFrame, id_col: str | None = None) -> pd.DataFrame:
    report_rows = []

    # leichte Metainfo fürs LLM: Spalten + simple Typen
    schema = {}
    
    for c in df.columns:
        schema[c] = str(df[c].dtype)
    
    rows_to_check = [41,42,43]
    
    for pos in rows_to_check:
        row = df.iloc[pos]
        idx = df.index[pos]  
        
        row_dict = row.to_dict()
        
        ## 1) schnelle Regelchecks (billig, deterministisch)
        #rule_issues = precheck_row_minimal(row_dict)

        # 2) LLM-Checks (Sprache, Inkonsistenz, Plausibilität)
        llm_result = check_row_with_llm(row_dict=row_dict, schema=schema, )

        # Ergebniszeile im Report
        product_id = row_dict.get(id_col) if id_col else row_dict.get("Produkt_ID")
        print("Produkt_ID: ", product_id)

        report_rows.append({
            "row_index": idx,
            "product_id": product_id,                # Liste
            "llm_language_issues": llm_result["language_issues"],
            "llm_inconsistencies": llm_result["inconsistencies"],
            "llm_plausibility_issues": llm_result["plausibility_issues"],
            "llm_analysis_step": llm_result["analysis_steps"],
            "llm_summary": llm_result["summary"],
            "llm_severity": llm_result["severity"],
        })

    return pd.DataFrame(report_rows)


def main():
    
    input_path = "./data_2/kaufhaus_produktdaten_test_100.xlsx"
    output_path = "./data_2/fehlerbericht_openai.xlsx"

    df = load_xlsx(input_path)
    report_df = build_error_report_batches(df, id_col=None)
    
    report_df.to_excel(output_path, index=False)
    print(f"Wrote report: {Path(output_path).resolve()}")
    

if __name__ == "__main__":
    main()
    
    
    



#def build_error_report_batches_stackof(df: pd.DataFrame, id_col: str | None = None, batch_delay = 0.35) -> pd.DataFrame:
#    report_rows = []
#    max_batches = 1
#
#    # leichte Metainfo fürs LLM: Spalten + simple Typen
#    schema = {}
#    for c in df.columns:
#        schema[c] = str(df[c].dtype)
#
#    # id_col auflösen
#    resolved_id_col = id_col if (id_col and id_col in df.columns) else ("Produkt_ID" if "Produkt_ID" in df.columns else df.columns[0])
#
#    # 2) Batches
#    for i, batch in enumerate(chunk_rows(df)):
#        if i >= max_batches:
#            break
#        
#        ## 1) schnelle Regelchecks (billig, deterministisch)
#        #rule_issues = precheck_row_minimal(row_dict)
#
#        # 2) LLM-Checks (Sprache, Inkonsistenz, Plausibilität)
#        prompt = build_prompt_for_row(batch, schema)
#        ai_result = call_openai_chat(prompt)
#        print(ai_result)
#        for resp in ai_result:
#            report_rows.append({
#            id_col: resolved_id_col,
#            "language_errors": resp["language_errors"],
#            "inconsistencies": resp["inconsistencies"],
#            "plausibility_issues": resp["plausibility_issues"],
#            "confidence": resp["confidence"]
#        })
#        time.sleep(batch_delay)
#        report_df = pd.DataFrame(report_rows)
#        return report_df