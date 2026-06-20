import pytest
import os
import json
import pandas as pd
from pbm_audit.engine import PBMAuditEngine

try:
    from pyspark.sql import SparkSession
    HAS_SPARK_VENV = True
except ImportError:
    HAS_SPARK_VENV = False

@pytest.fixture(scope="module")
def audit_engine():
    """Fixture to initialize hybrid engine (detects environment)."""
    return PBMAuditEngine()

@pytest.fixture(scope="module")
def sample_config():
    return {
        "plan_name": "Test Plan",
        "audit_year": 2026,
        "copay_structure": {
            "retail_retail": {"generic": 10.0, "brand": 35.0},
            "mail_order": {"generic": 20.0, "brand": 70.0},
            "specialty": {"generic": 50.0, "brand": 150.0}
        },
        "daw_penalty_policy": {
            "enabled": True,
            "daw_codes_subject_to_penalty": [1, 2],
            "penalty_type": "cost_difference"
        },
        "refill_too_soon": {
            "threshold_percentage": 0.75
        },
        "duplicate_claims": {
            "window_days": 0
        },
        "rebate_guarantees": {
            "retail_generic": 10.0,
            "retail_brand": 150.0,
            "mail_generic": 20.0,
            "mail_brand": 250.0,
            "specialty_generic": 400.0,
            "specialty_brand": 1000.0
        }
    }

def helper_create_df(engine, data, columns, spark_session=None):
    """Helper to create either a Spark DataFrame or Pandas DataFrame based on engine mode."""
    if engine.use_spark:
        return engine.spark.createDataFrame(data, columns)
    else:
        return pd.DataFrame(data, columns=columns)

def helper_collect(engine, df):
    """Helper to collect results as standard python dicts for assertion."""
    if engine.use_spark:
        return [row.asDict() for row in df.collect()]
    else:
        return df.to_dict(orient="records")

def test_audit_invalid_ndcs(audit_engine):
    claims_data = [
        ("C1", "M1", "12345-6789-01", "2026-01-01", 30, 30, 0, 10.0, 10.0, 0.0, "retail", "generic"), # Valid
        ("C2", "M2", "99999-9999-99", "2026-01-01", 30, 30, 0, 15.0, 10.0, 5.0, "retail", "generic")  # Invalid NDC
    ]
    columns = ["claim_id", "member_id", "ndc", "fill_date", "quantity", "days_supply", "daw_code", "awp_billed", "copay_paid", "pbm_paid", "channel", "drug_type"]
    claims_df = helper_create_df(audit_engine, claims_data, columns)
    
    drugs_data = [("12345-6789-01", "Drug A", True, 0.5, None, "retail")]
    drugs_cols = ["ndc", "drug_name", "is_generic", "awp_per_unit", "generic_for", "channel"]
    drugs_df = helper_create_df(audit_engine, drugs_data, drugs_cols)
    
    flagged_df = audit_engine.audit_invalid_ndcs(claims_df, drugs_df)
    results = helper_collect(audit_engine, flagged_df)
    
    assert len(results) == 1
    assert results[0]["claim_id"] == "C2"
    assert results[0]["ndc"] == "99999-9999-99"
    assert float(results[0]["financial_impact"]) == 5.0

def test_audit_questionable_awp(audit_engine):
    claims_data = [
        ("C1", "M1", "12345-6789-01", "2026-01-01", 30, 30, 0, 15.0, 10.0, 5.0, "retail", "generic"),  # Valid ($0.50 * 30 = $15.0)
        ("C2", "M2", "12345-6789-01", "2026-01-01", 30, 30, 0, 30.0, 10.0, 20.0, "retail", "generic")  # Questionable ($30 instead of $15)
    ]
    columns = ["claim_id", "member_id", "ndc", "fill_date", "quantity", "days_supply", "daw_code", "awp_billed", "copay_paid", "pbm_paid", "channel", "drug_type"]
    claims_df = helper_create_df(audit_engine, claims_data, columns)
    
    drugs_data = [("12345-6789-01", "Drug A", True, 0.5, None, "retail")]
    drugs_cols = ["ndc", "drug_name", "is_generic", "awp_per_unit", "generic_for", "channel"]
    drugs_df = helper_create_df(audit_engine, drugs_data, drugs_cols)
    
    flagged_df = audit_engine.audit_questionable_awp(claims_df, drugs_df)
    results = helper_collect(audit_engine, flagged_df)
    
    assert len(results) == 1
    assert results[0]["claim_id"] == "C2"
    assert float(results[0]["financial_impact"]) == 15.0  # $30 - $15 expected AWP

def test_audit_duplicate_claims(audit_engine):
    claims_data = [
        ("C1", "M1", "12345-6789-01", "2026-01-01", 30, 30, 0, 15.0, 10.0, 5.0, "retail", "generic"),
        ("C2", "M1", "12345-6789-01", "2026-01-01", 30, 30, 0, 15.0, 10.0, 5.0, "retail", "generic"), # Duplicate
        ("C3", "M1", "12345-6789-01", "2026-01-02", 30, 30, 0, 15.0, 10.0, 5.0, "retail", "generic")  # Different day
    ]
    columns = ["claim_id", "member_id", "ndc", "fill_date", "quantity", "days_supply", "daw_code", "awp_billed", "copay_paid", "pbm_paid", "channel", "drug_type"]
    
    # Handle date parsing for pandas
    if not audit_engine.use_spark:
        for i in range(len(claims_data)):
            row = list(claims_data[i])
            row[3] = pd.to_datetime(row[3]).date()
            claims_data[i] = tuple(row)
            
    claims_df = helper_create_df(audit_engine, claims_data, columns)
    
    flagged_df = audit_engine.audit_duplicate_claims(claims_df)
    results = helper_collect(audit_engine, flagged_df)
    
    assert len(results) == 1
    assert results[0]["claim_id"] == "C2"
    assert float(results[0]["financial_impact"]) == 5.0
