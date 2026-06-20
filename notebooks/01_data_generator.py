# Databricks notebook source
# MAGIC %md
# MAGIC # 01: PBM Audit Data Generator
# MAGIC This notebook generates synthetic Pharmacy Benefit Manager (PBM) claims, reference drug databases, member eligibility files, and mock contract summaries.
# MAGIC 
# MAGIC It simulates claims adjudication anomalies (e.g. invalid NDCs, questionable AWP pricing, DAW penalty bypass, incorrect copays, duplicate claims, refill-too-soon).

# COMMAND ----------

import os
# Try importing generator package, or define inline if run from scratch
try:
    from pbm_audit.generator import run_generation
except ImportError:
    # Append path in Databricks context if needed
    import sys
    sys.path.append(os.path.abspath(".."))
    from pbm_audit.generator import run_generation

# COMMAND ----------

# DBTITLE 1,Define output path (Databricks DBFS or local workspace)
# In Databricks, we usually write to DBFS: "/dbfs/tmp/pbm_audit_data/" or similar.
# Here we write to local `./data` folder, which is compatible with our app and notebooks.
output_dir = "./data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Generating synthetic PBM audit data and saving to {os.path.abspath(output_dir)}...")

# COMMAND ----------

# DBTITLE 2,Run Generation
run_generation(output_dir)

# COMMAND ----------

# DBTITLE 3,Verify files were created
print("Files generated in output directory:")
for f in os.listdir(output_dir):
    file_path = os.path.join(output_dir, f)
    print(f" - {f} ({os.path.getsize(file_path)} bytes)")
