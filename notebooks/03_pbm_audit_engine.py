# Databricks notebook source
# MAGIC %md
# MAGIC # 03: PySpark PBM Claims Audit Engine
# MAGIC This notebook runs the PySpark-based Pharmacy Benefit Manager (PBM) Claims Audit engine.
# MAGIC 
# MAGIC It loads the claims database, references, and benefit design rules, runs the 6 Next Generation claims audit tests, and performs the pricing and rebate reconciliation.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Initialize Session and Imports

# COMMAND ----------

import os
import json
import sys
import pandas as pd

# Ensure parent directory is in python path
sys.path.append(os.path.abspath(".."))

from pbm_audit.engine import PBMAuditEngine

# COMMAND ----------

# DBTITLE 1,Start/Retrieve Spark Session
# PBMAuditEngine will automatically get or create standard Spark session
engine = PBMAuditEngine()
print(f"Spark Version: {engine.spark.version}")

# COMMAND ----------

# DBTITLE 2,Load Benefit Design Configuration
config_path = "./config/benefit_design.json"
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Configuration file not found at {config_path}. Please run notebook 02 first.")

with open(config_path, "r") as f:
    config = json.load(f)

print(f"Loaded config for plan: {config.get('plan_name')}")

# COMMAND ----------

# DBTITLE 3,Load Datasets into PySpark
claims_file = "./data/pbm_claims.csv"
drugs_file = "./data/ndc_reference.csv"
members_file = "./data/member_eligibility.csv"

# Load using the engine's loader
claims_df, drugs_df, members_df = engine.load_data(claims_file, drugs_file, members_file)

print(f"Total claims loaded: {claims_df.count()}")
print(f"Total reference drugs loaded: {drugs_df.count()}")
print(f"Total eligible members loaded: {members_df.count()}")

# COMMAND ----------

# DBTITLE 4,Execute Audit Tests
print("Executing PBM Claims Audit Engine...")
audit_results_df = engine.run_all_audits(claims_df, drugs_df, config)

# Cache results for faster summary and queries
audit_results_df.cache()

print(f"Audit run complete. Total flagged claim lines: {audit_results_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Audit Summaries & Findings

# COMMAND ----------

# DBTITLE 1,Summarize Flagged Claims by Audit Category
summary_df = audit_results_df.groupBy("audit_test").agg(
    F.count("claim_id").alias("flagged_claims_count"),
    F.round(F.sum("financial_impact"), 2).alias("total_financial_impact")
).orderBy(F.desc("total_financial_impact"))

print("Summary of Flagged Claims by Audit Test Category:")
summary_df.show(truncate=False)

# COMMAND ----------

# DBTITLE 2,Top Drugs (NDCs) Flagged for Audit Anomalies
top_drugs_df = audit_results_df.groupBy("ndc").agg(
    F.count("claim_id").alias("flags_count"),
    F.round(F.sum("financial_impact"), 2).alias("total_impact")
).orderBy(F.desc("total_impact")).limit(10)

print("Top 10 Flagged NDCs by Financial Recovery Impact:")
top_drugs_df.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: PBM Rebate Guarantees Reconciliation

# COMMAND ----------

# DBTITLE 1,Run Rebate Guarantees Reconciliation
reconciliation, total_rebate = engine.perform_rebate_reconciliation(claims_df, config)

print(f"Total expected rebate yield based on guarantees: ${total_rebate:,.2f}")
print("\nReconciliation Breakdown by Channel and Drug Type:")
recon_pdf = pd.DataFrame(reconciliation)
print(recon_pdf.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4: Write Audit Results (Delta Lake / parquet)

# COMMAND ----------

# DBTITLE 1,Save Flagged Claims to Output Directory
output_results_dir = "./data/audit_results"
# Locally we save as standard CSV or parquet. In Databricks, we save as Delta format.
# Let's save as CSV for local app compatibility and Parquet for enterprise analytics
audit_results_df.write.mode("overwrite").parquet(os.path.join(output_results_dir, "flagged_claims.parquet"))
audit_results_df.write.mode("overwrite").csv(os.path.join(output_results_dir, "flagged_claims.csv"), header=True)

# Also save summaries as JSON for the dashboard to read
summary_dict = {
    "total_claims": claims_df.count(),
    "total_flagged_claims": audit_results_df.count(),
    "total_financial_impact": round(audit_results_df.select(F.sum("financial_impact")).collect()[0][0] or 0.0, 2),
    "expected_rebate_yield": total_rebate,
    "test_counts_and_impacts": {}
}

for row in summary_df.collect():
    summary_dict["test_counts_and_impacts"][row["audit_test"]] = {
        "count": row["flagged_claims_count"],
        "impact": row["total_financial_impact"]
    }

with open(os.path.join(output_results_dir, "audit_summary.json"), "w") as f:
    json.dump(summary_dict, f, indent=2)

print(f"Audit results successfully written to: {os.path.abspath(output_results_dir)}")
