import os
import sys
import json
import pandas as pd

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pbm_audit.generator import run_generation
from pbm_audit.parser import AIContractParser
from pbm_audit.engine import PBMAuditEngine

def main():
    print("=" * 60)
    print("      Antigravity PBM Audit & AI Platform Orchestrator")
    print("=" * 60)

    # 1. Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    config_dir = os.path.join(base_dir, "config")
    audit_results_dir = os.path.join(data_dir, "audit_results")

    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(audit_results_dir, exist_ok=True)

    # 2. Run Data Generator if CSVs don't exist
    claims_file = os.path.join(data_dir, "pbm_claims.csv")
    if not os.path.exists(claims_file):
        print("\n[Step 1/3] Generating synthetic claims and reference drug databases...")
        run_generation(data_dir)
    else:
        print("\n[Step 1/3] Synthetic claims files already present, skipping generation.")

    # 3. Run AI Contract Parser if config doesn't exist
    config_file = os.path.join(config_dir, "benefit_design.json")
    if not os.path.exists(config_file):
        print("\n[Step 2/3] Extracting benefit rules from contract SPD...")
        contract_file = os.path.join(data_dir, "contract_summary.txt")
        with open(contract_file, "r") as f:
            contract_text = f.read()
        
        parser = AIContractParser()
        parsed_config = parser.parse_contract(contract_text)
        
        with open(config_file, "w") as f:
            json.dump(parsed_config, f, indent=2)
        print(f"Benefit design configuration saved to: {config_file}")
    else:
        print("\n[Step 2/3] Benefit design config already present, skipping parsing.")

    # 4. Run Audit Engine if results don't exist
    results_summary_file = os.path.join(audit_results_dir, "audit_summary.json")
    if not os.path.exists(results_summary_file):
        print("\n[Step 3/3] Initializing Audit Engine and running calculations...")
        
        drugs_file = os.path.join(data_dir, "ndc_reference.csv")
        members_file = os.path.join(data_dir, "member_eligibility.csv")
        
        with open(config_file, "r") as f:
            config = json.load(f)
            
        engine = PBMAuditEngine()
        claims_df, drugs_df, members_df = engine.load_data(claims_file, drugs_file, members_file)
        
        # Execute audits
        flagged_df = engine.run_all_audits(claims_df, drugs_df, config)
        
        # Handle hybrid Spark / Pandas data types
        if engine.use_spark:
            flagged_pdf = flagged_df.toPandas()
            total_claims_count = int(claims_df.count())
            total_flagged_count = int(flagged_df.count())
        else:
            flagged_pdf = flagged_df
            total_claims_count = len(claims_df)
            total_flagged_count = len(flagged_df)
            
        # Cache results locally
        flagged_pdf.to_csv(os.path.join(audit_results_dir, "flagged_claims.csv"), index=False)
        
        # Summary & counts
        summary_df = flagged_pdf.groupby("audit_test").agg(
            flagged_claims_count=("claim_id", "count"),
            total_financial_impact=("financial_impact", "sum")
        ).reset_index()
        
        # Rebates
        reconciliation, total_rebate = engine.perform_rebate_reconciliation(claims_df, config)
        pd.DataFrame(reconciliation).to_csv(os.path.join(audit_results_dir, "rebate_reconciliation.csv"), index=False)
        
        summary_dict = {
            "total_claims": total_claims_count,
            "total_flagged_claims": total_flagged_count,
            "total_financial_impact": round(float(flagged_pdf["financial_impact"].sum()), 2),
            "expected_rebate_yield": round(total_rebate, 2),
            "test_counts_and_impacts": {}
        }
        
        for _, row in summary_df.iterrows():
            summary_dict["test_counts_and_impacts"][row["audit_test"]] = {
                "count": int(row["flagged_claims_count"]),
                "impact": float(row["total_financial_impact"])
            }
            
        with open(results_summary_file, "w") as f:
            json.dump(summary_dict, f, indent=2)
            
        print("Audit calculations completed and results cached.")
    else:
        print("\n[Step 3/3] Audit results already cached.")

    print("\n" + "=" * 60)
    print("All backend components initialized and ready!")
    print("To launch the dashboard, execute:")
    print("   streamlit run app/main.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
