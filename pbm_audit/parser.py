import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AIContractParser:
    """Uses LLM to parse pharmacy benefit contract documents into structured config."""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None

    def parse_contract(self, contract_text):
        """Parses the contract text and returns a structured dictionary."""
        if self.client:
            try:
                print("Using OpenAI GPT to parse contract text...")
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            "role": "system", 
                            "content": (
                                "You are a professional healthcare actuary and claims auditor. "
                                "Your task is to parse pharmacy benefit contract summaries or SPDs "
                                "and extract specific benefit rules in JSON format.\n\n"
                                "Your output JSON must strictly match this schema:\n"
                                "{\n"
                                "  \"plan_name\": \"string\",\n"
                                "  \"audit_year\": integer,\n"
                                "  \"copay_structure\": {\n"
                                "    \"retail_retail\": {\"generic\": float, \"brand\": float},\n"
                                "    \"mail_order\": {\"generic\": float, \"brand\": float},\n"
                                "    \"specialty\": {\"generic\": float, \"brand\": float}\n"
                                "  },\n"
                                "  \"daw_penalty_policy\": {\n"
                                "    \"enabled\": boolean,\n"
                                "    \"daw_codes_subject_to_penalty\": [integer],\n"
                                "    \"penalty_type\": \"string\",\n"
                                "    \"description\": \"string\"\n"
                                "  },\n"
                                "  \"refill_too_soon\": {\n"
                                "    \"threshold_percentage\": float,\n"
                                "    \"description\": \"string\"\n"
                                "  },\n"
                                "  \"duplicate_claims\": {\n"
                                "    \"window_days\": integer,\n"
                                "    \"description\": \"string\"\n"
                                "  },\n"
                                "  \"rebate_guarantees\": {\n"
                                "    \"retail_generic\": float,\n"
                                "    \"retail_brand\": float,\n"
                                "    \"mail_generic\": float,\n"
                                "    \"mail_brand\": float,\n"
                                "    \"specialty_generic\": float,\n"
                                "    \"specialty_brand\": float\n"
                                "  }\n"
                                "}"
                            )
                        },
                        {"role": "user", "content": contract_text}
                    ]
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                print(f"Error calling OpenAI API: {e}. Falling back to regex parser...")
                return self._fallback_parse(contract_text)
        else:
            print("No OpenAI API key found in environment. Falling back to regex rule-based parser...")
            return self._fallback_parse(contract_text)

    def _fallback_parse(self, text):
        """Rule-based regex fallback parser for demonstration when API key is missing."""
        config = {
            "plan_name": "Standard Employer Health Plan (Regex Parsed)",
            "audit_year": 2026,
            "copay_structure": {
                "retail_retail": {"generic": 10.00, "brand": 35.00},
                "mail_order": {"generic": 20.00, "brand": 70.00},
                "specialty": {"generic": 50.00, "brand": 150.00}
            },
            "daw_penalty_policy": {
                "enabled": True,
                "daw_codes_subject_to_penalty": [1, 2],
                "penalty_type": "cost_difference",
                "description": "Bypassed DAW cost-difference penalty."
            },
            "refill_too_soon": {
                "threshold_percentage": 0.75,
                "description": "Refill-too-soon limit set to 75% utilization."
            },
            "duplicate_claims": {
                "window_days": 0,
                "description": "Same day duplicates."
            },
            "rebate_guarantees": {
                "retail_generic": 15.00,
                "retail_brand": 220.00,
                "mail_generic": 35.00,
                "mail_brand": 310.00,
                "specialty_generic": 600.00,
                "specialty_brand": 1500.00
            }
        }
        
        # Simple extraction logic to prove it works
        plan_match = re.search(r"Plan Sponsor:\s*(.+)", text)
        if plan_match:
            config["plan_name"] = plan_match.group(1).strip()
            
        year_match = re.search(r"Audit Year:\s*(\d+)", text)
        if year_match:
            config["audit_year"] = int(year_match.group(1))

        # Extracted copays via regex
        retail_gen = re.search(r"Retail Network.*?\n.*?Generic.*?\$([\d\.]+)", text, re.IGNORECASE)
        retail_brand = re.search(r"Retail Network.*?\n.*?\n.*?Brand.*?\$([\d\.]+)", text, re.IGNORECASE)
        if retail_gen:
            config["copay_structure"]["retail_retail"]["generic"] = float(retail_gen.group(1))
        if retail_brand:
            config["copay_structure"]["retail_retail"]["brand"] = float(retail_brand.group(1))

        mail_gen = re.search(r"Mail Order.*?\n.*?Generic.*?\$([\d\.]+)", text, re.IGNORECASE)
        mail_brand = re.search(r"Mail Order.*?\n.*?\n.*?Brand.*?\$([\d\.]+)", text, re.IGNORECASE)
        if mail_gen:
            config["copay_structure"]["mail_order"]["generic"] = float(mail_gen.group(1))
        if mail_brand:
            config["copay_structure"]["mail_order"]["brand"] = float(mail_brand.group(1))

        spec_gen = re.search(r"Specialty Pharmacy.*?\n.*?Generic.*?\$([\d\.]+)", text, re.IGNORECASE)
        spec_brand = re.search(r"Specialty Pharmacy.*?\n.*?\n.*?Brand.*?\$([\d\.]+)", text, re.IGNORECASE)
        if spec_gen:
            config["copay_structure"]["specialty"]["generic"] = float(spec_gen.group(1))
        if spec_brand:
            config["copay_structure"]["specialty"]["brand"] = float(spec_brand.group(1))
            
        # Refill threshold
        refill_match = re.search(r"at least\s*(\d+)%\s*utilized", text, re.IGNORECASE)
        if refill_match:
            config["refill_too_soon"]["threshold_percentage"] = float(refill_match.group(1)) / 100.0

        # Rebate matches
        rebate_patterns = {
            "retail_generic": r"Retail Network Generic.*?\$([\d\.]+)",
            "retail_brand": r"Retail Network Brand.*?\$([\d\.]+)",
            "mail_generic": r"Mail Order Generic.*?\$([\d\.]+)",
            "mail_brand": r"Mail Order Brand.*?\$([\d\.]+)",
            "specialty_generic": r"Specialty Generic.*?\$([\d\.]+)",
            "specialty_brand": r"Specialty Brand.*?\$([\d\.]+)"
        }
        for key, pattern in rebate_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                config["rebate_guarantees"][key] = float(match.group(1))
                
        return config

if __name__ == "__main__":
    # Test parser
    sample_text = """
    Plan Sponsor: ACME Corp
    Audit Year: 2026
    Retail Generic: $5.00
    Retail Brand: $25.00
    Mail Generic: $10.00
    Mail Brand: $50.00
    Specialty Generic: $100.00
    Specialty Brand: $300.00
    Claims must be 80% utilized.
    Retail Network Generic rebate guarantee: $12.50
    Retail Network Brand rebate guarantee: $190.00
    """
    parser = AIContractParser()
    res = parser.parse_contract(sample_text)
    print(json.dumps(res, indent=2))
