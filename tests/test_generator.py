import os
import tempfile
import pandas as pd
from pbm_audit.generator import (
    generate_reference_drugs,
    generate_members,
    generate_claims,
    run_generation
)

def test_generate_reference_drugs():
    df = generate_reference_drugs()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "ndc" in df.columns
    assert "awp_per_unit" in df.columns
    assert "is_generic" in df.columns

def test_generate_members():
    df = generate_members(num_members=20)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 20
    assert "member_id" in df.columns
    assert "status" in df.columns

def test_generate_claims():
    df_drugs = generate_reference_drugs()
    df_members = generate_members(num_members=50)
    df_claims = generate_claims(df_drugs, df_members, num_claims=100)
    
    assert isinstance(df_claims, pd.DataFrame)
    assert not df_claims.empty
    assert "claim_id" in df_claims.columns
    assert "ndc" in df_claims.columns
    assert "awp_billed" in df_claims.columns
    assert "copay_paid" in df_claims.columns

def test_run_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_generation(tmpdir)
        
        assert os.path.exists(os.path.join(tmpdir, "ndc_reference.csv"))
        assert os.path.exists(os.path.join(tmpdir, "member_eligibility.csv"))
        assert os.path.exists(os.path.join(tmpdir, "pbm_claims.csv"))
        assert os.path.exists(os.path.join(tmpdir, "contract_summary.txt"))
