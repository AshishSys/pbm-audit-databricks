# Databricks notebook source
# MAGIC %md
# MAGIC # 02: AI Contract Parser
# MAGIC This notebook demonstrates how to use Generative AI (LLMs) to extract contract rules and parameters from natural language Summary Plan Descriptions (SPDs) and PBM agreements.
# MAGIC 
# MAGIC It outputs a structured benefit design JSON config which drives the PySpark claims auditing engine.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Configure OpenAI API (Optional)
# MAGIC To run using OpenAI, configure your `OPENAI_API_KEY` in your environment or Databricks secrets manager.
# MAGIC 
# MAGIC If no key is configured, the system will fall back to a rule-based regex parser to allow seamless execution.

# COMMAND ----------

import os
import json
import sys

# Ensure parent directory is in python path
sys.path.append(os.path.abspath(".."))

from pbm_audit.parser import AIContractParser

# COMMAND ----------

# DBTITLE 1,Read Plaintext Contract Summary
contract_file = "./data/contract_summary.txt"

if not os.path.exists(contract_file):
    raise FileNotFoundError(
        f"Contract file not found at {contract_file}. "
        "Please run the 01_data_generator notebook first to generate sample files."
    )

with open(contract_file, "r") as f:
    contract_text = f.read()

print("Loaded Contract Document Summary:")
print("-" * 50)
print(contract_text[:800] + "...\n[Truncated]")
print("-" * 50)

# COMMAND ----------

# DBTITLE 2,Initialize AI Contract Parser
# Will automatically load API key from environment variable OPENAI_API_KEY
parser = AIContractParser()

# COMMAND ----------

# DBTITLE 3,Parse Contract into JSON Benefit Rules Config
print("Parsing contract rules...")
benefit_design_rules = parser.parse_contract(contract_text)

# Pretty print parsed configuration
print(json.dumps(benefit_design_rules, indent=2))

# COMMAND ----------

# DBTITLE 4,Save Config for PBM Audit Engine
config_dir = "./config"
if not os.path.exists(config_dir):
    os.makedirs(config_dir)
    
config_output = os.path.join(config_dir, "benefit_design.json")
with open(config_output, "w") as f:
    json.dump(benefit_design_rules, f, indent=2)

print(f"Benefit design config written successfully to: {os.path.abspath(config_output)}")
