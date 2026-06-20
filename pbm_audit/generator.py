import os
import random
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_reference_drugs():
    """Generates a reference NDC master drug database."""
    drugs = [
        # Generic Drugs (Common)
        {"ndc": "00093-3109-05", "drug_name": "Lisinopril 10mg", "is_generic": True, "awp_per_unit": 0.35, "generic_for": "Prinivil", "channel": "retail"},
        {"ndc": "00093-0058-01", "drug_name": "Atorvastatin 20mg", "is_generic": True, "awp_per_unit": 0.45, "generic_for": "Lipitor", "channel": "retail"},
        {"ndc": "68180-0121-01", "drug_name": "Metformin 500mg", "is_generic": True, "awp_per_unit": 0.15, "generic_for": "Glucophage", "channel": "retail"},
        {"ndc": "65862-0198-99", "drug_name": "Amoxicillin 500mg", "is_generic": True, "awp_per_unit": 0.25, "generic_for": "Amoxil", "channel": "retail"},
        {"ndc": "50111-0333-01", "drug_name": "Fluoxetine 20mg", "is_generic": True, "awp_per_unit": 0.40, "generic_for": "Prozac", "channel": "retail"},
        {"ndc": "60505-0134-00", "drug_name": "Omeprazole 20mg", "is_generic": True, "awp_per_unit": 0.30, "generic_for": "Prilosec", "channel": "retail"},
        
        # Brand Drugs (Common)
        {"ndc": "00006-0952-54", "drug_name": "Prinivil 10mg", "is_generic": False, "awp_per_unit": 3.80, "generic_for": None, "channel": "retail"},
        {"ndc": "00071-0156-23", "drug_name": "Lipitor 20mg", "is_generic": False, "awp_per_unit": 4.50, "generic_for": None, "channel": "retail"},
        {"ndc": "00002-3270-30", "drug_name": "Prozac 20mg", "is_generic": False, "awp_per_unit": 5.20, "generic_for": None, "channel": "retail"},
        {"ndc": "00006-0275-31", "drug_name": "Singulair 10mg", "is_generic": False, "awp_per_unit": 6.10, "generic_for": None, "channel": "retail"},
        
        # Mail Order Drugs (often maintenance)
        {"ndc": "00093-7271-98", "drug_name": "Levothyroxine 50mcg", "is_generic": True, "awp_per_unit": 0.20, "generic_for": "Synthroid", "channel": "mail"},
        {"ndc": "00074-4339-13", "drug_name": "Synthroid 50mcg", "is_generic": False, "awp_per_unit": 2.10, "generic_for": None, "channel": "mail"},
        
        # Specialty Drugs (Expensive)
        {"ndc": "00074-0243-02", "drug_name": "Humira 40mg/0.4ml", "is_generic": False, "awp_per_unit": 1850.00, "generic_for": None, "channel": "specialty"},
        {"ndc": "50242-0040-62", "drug_name": "Enbrel 50mg/ml", "is_generic": False, "awp_per_unit": 1720.00, "generic_for": None, "channel": "specialty"},
        {"ndc": "00085-4345-05", "drug_name": "Keytruda 100mg/4ml", "is_generic": False, "awp_per_unit": 2400.00, "generic_for": None, "channel": "specialty"},
        {"ndc": "60505-0211-12", "drug_name": "Imatinib 100mg", "is_generic": True, "awp_per_unit": 12.50, "generic_for": "Gleevec", "channel": "specialty"},
        {"ndc": "00078-0401-34", "drug_name": "Gleevec 100mg", "is_generic": False, "awp_per_unit": 110.00, "generic_for": None, "channel": "specialty"}
    ]
    return pd.DataFrame(drugs)

def generate_members(num_members=200):
    """Generates synthetic member eligibility dataset."""
    first_names = ["John", "Jane", "Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Henry", "Ivy", "Jack", "Kate", "Leo", "Mary", "Nick", "Olivia", "Paul", "Ruth", "Sam"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"]
    
    members = []
    start_date = datetime(1950, 1, 1)
    end_date = datetime(2015, 12, 31)
    
    for i in range(1, num_members + 1):
        member_id = f"M{100000 + i}"
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        dob = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
        
        # Coverages: Active or Terminated
        status = "Active" if random.random() < 0.9 else "Terminated"
        coverage_end = "" if status == "Active" else (datetime(2026, 1, 1) + timedelta(days=random.randint(1, 150))).strftime("%Y-%m-%d")
        
        members.append({
            "member_id": member_id,
            "first_name": first_name,
            "last_name": last_name,
            "dob": dob.strftime("%Y-%m-%d"),
            "status": status,
            "coverage_start_date": "2024-01-01",
            "coverage_end_date": coverage_end
        })
        
    return pd.DataFrame(members)

def generate_claims(df_drugs, df_members, num_claims=1000, start_year=2026):
    """Generates synthetic PBM claims data, embedding Milliman audit audit-failures."""
    claims = []
    
    # Pre-calculate active members
    active_member_ids = df_members["member_id"].tolist()
    
    # Benefit design mapping for copays
    copays = {
        "retail": {"generic": 10.0, "brand": 35.0},
        "mail": {"generic": 20.0, "brand": 70.0},
        "specialty": {"generic": 50.0, "brand": 150.0}
    }
    
    base_date = datetime(start_year, 1, 1)
    
    for claim_idx in range(1, num_claims + 1):
        claim_id = f"C{2000000 + claim_idx}"
        member_id = random.choice(active_member_ids)
        
        # Pick a drug
        drug = df_drugs.sample(n=1).iloc[0]
        ndc = drug["ndc"]
        is_generic = drug["is_generic"]
        awp_per_unit = drug["awp_per_unit"]
        channel = drug["channel"]
        
        # Setup basic params
        qty = random.choice([30, 60, 90]) if channel != "specialty" else 30
        days_supply = qty  # Standard days supply
        
        fill_date = base_date + timedelta(days=random.randint(0, 360))
        
        # Calculate standard fields
        awp_billed = round(qty * awp_per_unit, 2)
        
        # Determine drug type for copay mapping
        copay_type = "generic" if is_generic else "brand"
        expected_copay = copays[channel][copay_type]
        
        copay_paid = expected_copay
        daw_code = 0
        pbm_paid = max(0.0, round(awp_billed - copay_paid, 2))
        
        # We will embed a few test anomalies intentionally
        anomaly_type = "None"
        
        # Check to randomly insert anomalies
        rand_val = random.random()
        if rand_val < 0.08:  # 8% total anomalies
            anomaly_select = random.choice([
                "invalid_ndc", 
                "questionable_awp", 
                "daw_penalty_bypass", 
                "incorrect_copay", 
                "duplicate", 
                "refill_too_soon"
            ])
            
            if anomaly_select == "invalid_ndc":
                ndc = "99999-9999-99"  # Dummy invalid NDC
                anomaly_type = "invalid_ndc"
                
            elif anomaly_select == "questionable_awp":
                # Inflate the AWP billed by 50%
                awp_billed = round(awp_billed * 1.5, 2)
                pbm_paid = max(0.0, round(awp_billed - copay_paid, 2))
                anomaly_type = "questionable_awp"
                
            elif anomaly_select == "daw_penalty_bypass":
                # Brand drug with generic equivalent, DAW 1 (doctor requested brand) or DAW 2 (member requested brand)
                # But copay remains default brand copay without DAW cost-diff penalty
                brand_drugs = df_drugs[~df_drugs["is_generic"] & df_drugs["generic_for"].isna() & (df_drugs["channel"] != "specialty")]
                if len(brand_drugs) > 0:
                    brand_drug = brand_drugs.sample(1).iloc[0]
                    # Find its generic equivalent if any
                    gen_equiv = df_drugs[df_drugs["generic_for"] == brand_drug["drug_name"]]
                    if len(gen_equiv) > 0:
                        gen_drug = gen_equiv.iloc[0]
                        ndc = brand_drug["ndc"]
                        qty = 30
                        days_supply = 30
                        awp_billed = round(qty * brand_drug["awp_per_unit"], 2)
                        
                        # In regular case, DAW penalty is applied: member pays Brand Copay + (Brand Cost - Generic Cost)
                        # Here, we bypass it (anomaly): PBM pays the rest and charges ONLY brand copay
                        copay_paid = copays[brand_drug["channel"]]["brand"] 
                        daw_code = 1  # Doctor requested brand
                        pbm_paid = max(0.0, round(awp_billed - copay_paid, 2))
                        anomaly_type = "daw_penalty_bypass"
                        
            elif anomaly_select == "incorrect_copay":
                # Paid lower copay, say $0.00
                copay_paid = 0.0
                pbm_paid = awp_billed
                anomaly_type = "incorrect_copay"
                
            elif anomaly_select == "duplicate":
                # We will write this claim, and append another identical claim for the same day
                anomaly_type = "duplicate"
                
            elif anomaly_select == "refill_too_soon":
                # We will handle this by appending a matching claim shortly after
                anomaly_type = "refill_too_soon"
                
        claims.append({
            "claim_id": claim_id,
            "member_id": member_id,
            "ndc": ndc,
            "fill_date": fill_date.strftime("%Y-%m-%d"),
            "quantity": qty,
            "days_supply": days_supply,
            "daw_code": daw_code,
            "awp_billed": awp_billed,
            "copay_paid": copay_paid,
            "pbm_paid": pbm_paid,
            "channel": channel,
            "drug_type": "generic" if is_generic else "brand",
            "anomaly_type": anomaly_type
        })
        
        # Post-processing helper to create matching duplicates and refill-too-soon claims
        if anomaly_type == "duplicate":
            # Add an identical claim on the same day
            claims.append({
                "claim_id": f"{claim_id}D",
                "member_id": member_id,
                "ndc": ndc,
                "fill_date": fill_date.strftime("%Y-%m-%d"),
                "quantity": qty,
                "days_supply": days_supply,
                "daw_code": daw_code,
                "awp_billed": awp_billed,
                "copay_paid": copay_paid,
                "pbm_paid": pbm_paid,
                "channel": channel,
                "drug_type": "generic" if is_generic else "brand",
                "anomaly_type": "duplicate"
            })
        elif anomaly_type == "refill_too_soon":
            # Add a claim 5 days later (previous days_supply was 30 or 60 or 90, so 5 is way too early)
            refill_date = fill_date + timedelta(days=5)
            claims.append({
                "claim_id": f"{claim_id}R",
                "member_id": member_id,
                "ndc": ndc,
                "fill_date": refill_date.strftime("%Y-%m-%d"),
                "quantity": qty,
                "days_supply": days_supply,
                "daw_code": daw_code,
                "awp_billed": awp_billed,
                "copay_paid": copay_paid,
                "pbm_paid": pbm_paid,
                "channel": channel,
                "drug_type": "generic" if is_generic else "brand",
                "anomaly_type": "refill_too_soon"
            })

    return pd.DataFrame(claims)

def generate_sample_contract_text():
    """Generates natural language contract text for the AI Parser to read."""
    text = """
PHARMACY BENEFIT MANAGEMENT SERVICE AGREEMENT - SUMMARY PLAN DESCRIPTION (SPD)
Audit Year: 2026
Plan Sponsor: Standard Employer Health Plan

This document outlines the agreed-upon benefit designs, copayment schedules, pricing discounts, 
and performance metrics for pharmacy benefits.

SECTION 1: MEMBER COPAYMENTS
Members will be responsible for a flat copayment based on the pharmacy channel and drug tier:
1. Retail Network Pharmacy (Up to 30 Days Supply):
   - Generic Drugs: Member Copay of $10.00 per prescription.
   - Brand Name Drugs: Member Copay of $35.00 per prescription.
2. Mail Order Pharmacy (Up to 90 Days Supply):
   - Generic Drugs: Member Copay of $20.00 per prescription.
   - Brand Name Drugs: Member Copay of $70.00 per prescription.
3. Specialty Pharmacy Program (Up to 30 Days Supply):
   - Generic Drugs: Member Copay of $50.00 per prescription.
   - Brand Name Drugs: Member Copay of $150.00 per prescription.

SECTION 2: DISPENSE AS WRITTEN (DAW) RULES
If a member or physician selects a Brand Name Drug when a Generic equivalent is available 
(specifically DAW Codes 1 and 2), the member will be responsible for the Standard Brand Copay 
PLUS the difference in cost between the Brand AWP and the Generic AWP. This penalty shall 
apply automatically unless explicitly waived by the Plan Administrator.

SECTION 3: REFILL TIMING PATTERNS
Plan benefits require that a prescription be at least 75% utilized before a refill is authorized 
and paid. This is calculated using the Days Supply of the immediately preceding claim. Claims filled 
prior to this threshold will be flagged as Refill-too-soon and are subject to audit and recovery.

SECTION 4: DUPLICATE CLAIMS
Claims dispensed for the same member, same drug (NDC), on the same day are prohibited and 
represent administrative processing duplicates. The PBM is responsible for recovering 100% of 
such duplicate overpayments.

SECTION 5: PBM REBATE GUARANTEES
The PBM guarantees the following minimum manufacturer rebate payments back to the Plan Sponsor:
- Retail Network Generic claims: $15.00 per claim
- Retail Network Brand claims: $220.00 per claim
- Mail Order Generic claims: $35.00 per claim
- Mail Order Brand claims: $310.00 per claim
- Specialty Generic claims: $600.00 per claim
- Specialty Brand claims: $1500.00 per claim
    """
    return text

def run_generation(output_dir):
    """Orchestrates creation and saving of all datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating reference drug database...")
    df_drugs = generate_reference_drugs()
    df_drugs.to_csv(os.path.join(output_dir, "ndc_reference.csv"), index=False)
    
    print("Generating member eligibility database...")
    df_members = generate_members()
    df_members.to_csv(os.path.join(output_dir, "member_eligibility.csv"), index=False)
    
    print("Generating claims database...")
    df_claims = generate_claims(df_drugs, df_members, num_claims=1200)
    df_claims.to_csv(os.path.join(output_dir, "pbm_claims.csv"), index=False)
    
    print("Generating mock contract text summary...")
    contract_text = generate_sample_contract_text()
    with open(os.path.join(output_dir, "contract_summary.txt"), "w") as f:
        f.write(contract_text)
        
    print(f"All synthetic datasets generated successfully in: {output_dir}")

if __name__ == "__main__":
    run_generation("./data")
