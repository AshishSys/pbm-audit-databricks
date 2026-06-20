import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class AIAuditorAssistant:
    """AI Claims Auditor Chatbot providing natural language insights into PBM audit findings."""
    
    def __init__(self, audit_summary_dict, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.summary_data = audit_summary_dict
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None

    def ask(self, user_question):
        """Answers questions based on the consolidated audit summary statistics."""
        if self.client:
            try:
                system_prompt = (
                    "You are 'Antigravity AI Auditor', a specialized AI assistant that helps health plan sponsors "
                    "understand and analyze Pharmacy Benefit Manager (PBM) claims audits.\n\n"
                    "Below is the official summary data of the PBM audit findings:\n"
                    f"{self.summary_data}\n\n"
                    "Use this summary data to answer the user's questions clearly, accurately, and professionally. "
                    "Highlight potential recovery savings (ROI) and explain the audit categories if asked. "
                    "If the user asks for detailed information that is not in this summary, politely suggest they "
                    "look at the tables or export detailed reports in the dashboard."
                )
                
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_question}
                    ]
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error executing AI query: {e}. Fallback: Here is the summary data: {self.summary_data}"
        else:
            # Fallback response engine using keyword matching
            q = user_question.lower()
            total_savings = self.summary_data.get('total_financial_impact', 0)
            tests_summary = self.summary_data.get('test_counts_and_impacts', {})
            
            if "saving" in q or "recover" in q or "money" in q or "financial" in q:
                return (
                    f"Based on the audit findings, there is a total potential recovery of **${total_savings:,.2f}** "
                    f"due to PBM adjudication errors. The largest source of recovery is "
                    f"**{self._get_max_impact_test(tests_summary)}**."
                )
            elif "invalid ndc" in q:
                ndc_data = tests_summary.get("Invalid NDC", {"count": 0, "impact": 0.0})
                return f"We identified **{ndc_data['count']}** claims with Invalid NDCs, resulting in an impact of **${ndc_data['impact']:,.2f}**."
            elif "copay" in q:
                copay_data = tests_summary.get("Incorrect Copay", {"count": 0, "impact": 0.0})
                return f"We found **{copay_data['count']}** claims with Incorrect Copays, resulting in an impact of **${copay_data['impact']:,.2f}**."
            elif "daw" in q:
                daw_data = tests_summary.get("DAW Penalty Bypass", {"count": 0, "impact": 0.0})
                return f"We found **{daw_data['count']}** claims bypassing DAW penalties, with an impact of **${daw_data['impact']:,.2f}**."
            elif "duplicate" in q:
                dup_data = tests_summary.get("Duplicate Claim", {"count": 0, "impact": 0.0})
                return f"Duplicate claims accounted for **{dup_data['count']}** flags and **${dup_data['impact']:,.2f}** in recovery potential."
            elif "refill" in q or "soon" in q:
                rts_data = tests_summary.get("Refill Too Soon", {"count": 0, "impact": 0.0})
                return f"Refill-too-soon claims accounted for **{rts_data['count']}** flags and **${rts_data['impact']:,.2f}** in recovery potential."
            else:
                return (
                    "Hello! I am the Antigravity AI Auditor assistant. Here is a quick snapshot of the audit findings:\n"
                    f"- **Total Flagged Claims**: {self.summary_data.get('total_flagged_claims', 0)} out of {self.summary_data.get('total_claims', 0)} analyzed.\n"
                    f"- **Total Potential Savings**: ${total_savings:,.2f}\n"
                    f"- **Expected Rebates Yield**: ${self.summary_data.get('expected_rebate_yield', 0):,.2f}\n\n"
                    "Feel free to configure your `OPENAI_API_KEY` in a `.env` file to unlock deep conversational auditing!"
                )

    def _get_max_impact_test(self, tests_summary):
        max_test = "None"
        max_val = -1
        for test_name, data in tests_summary.items():
            if data["impact"] > max_val:
                max_val = data["impact"]
                max_test = test_name
        return f"{max_test} (${max_val:,.2f})"
