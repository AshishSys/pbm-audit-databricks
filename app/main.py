import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import local packages
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pbm_audit.generator import run_generation, generate_sample_contract_text
from pbm_audit.parser import AIContractParser
from pbm_audit.engine import PBMAuditEngine
from pbm_audit.assistant import AIAuditorAssistant

# Page Config
st.set_page_config(
    page_title="Antigravity PBM Audit & AI Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Inline fallback styles if stylesheet missing
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        .main-header {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            padding: 2rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        }
        .kpi-card {
            background-color: #f8fafc;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 5px solid #0284c7;
            box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1);
            transition: all 0.2s ease-in-out;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        }
        .kpi-title {
            font-size: 0.875rem;
            color: #64748b;
            text-transform: uppercase;
            font-weight: 600;
        }
        .kpi-value {
            font-size: 1.875rem;
            color: #0f172a;
            font-weight: 700;
            margin-top: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True)

local_css(os.path.join(os.path.dirname(__file__), "styles.css"))

# Paths configuration
DATA_DIR = "./data"
CONFIG_DIR = "./config"
AUDIT_RESULTS_DIR = os.path.join(DATA_DIR, "audit_results")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(AUDIT_RESULTS_DIR, exist_ok=True)

# Helper to check if data is generated
def is_data_available():
    return os.path.exists(os.path.join(DATA_DIR, "pbm_claims.csv"))

# Helper to check if audit results are available
def is_audit_results_available():
    return os.path.exists(os.path.join(AUDIT_RESULTS_DIR, "audit_summary.json"))

# Function to run audit locally using our PBMAuditEngine
def run_pbm_audit():
    with st.spinner("Running PySpark PBM Audit Engine..."):
        claims_file = os.path.join(DATA_DIR, "pbm_claims.csv")
        drugs_file = os.path.join(DATA_DIR, "ndc_reference.csv")
        members_file = os.path.join(DATA_DIR, "member_eligibility.csv")
        config_file = os.path.join(CONFIG_DIR, "benefit_design.json")
        
        # In case files aren't generated yet, generate them
        if not os.path.exists(claims_file):
            run_generation(DATA_DIR)
            
        # Start PySpark Engine
        engine = PBMAuditEngine()
        claims_df, drugs_df, members_df = engine.load_data(claims_file, drugs_file, members_file)
        
        # Load config
        with open(config_file, "r") as f:
            config = json.load(f)
            
        # Run audit
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
            
        # Save results locally for Streamlit
        flagged_pdf.to_csv(os.path.join(AUDIT_RESULTS_DIR, "flagged_claims.csv"), index=False)
        
        # Summary & counts
        summary_df = flagged_pdf.groupby("audit_test").agg(
            flagged_claims_count=("claim_id", "count"),
            total_financial_impact=("financial_impact", "sum")
        ).reset_index()
        
        # Rebates
        reconciliation, total_rebate = engine.perform_rebate_reconciliation(claims_df, config)
        pd.DataFrame(reconciliation).to_csv(os.path.join(AUDIT_RESULTS_DIR, "rebate_reconciliation.csv"), index=False)
        
        total_impact = float(flagged_pdf["financial_impact"].sum())
        
        # Save JSON summary
        summary_dict = {
            "total_claims": total_claims_count,
            "total_flagged_claims": total_flagged_count,
            "total_financial_impact": round(total_impact, 2),
            "expected_rebate_yield": round(total_rebate, 2),
            "test_counts_and_impacts": {}
        }
        
        for _, row in summary_df.iterrows():
            summary_dict["test_counts_and_impacts"][row["audit_test"]] = {
                "count": int(row["flagged_claims_count"]),
                "impact": float(row["total_financial_impact"])
            }
            
        with open(os.path.join(AUDIT_RESULTS_DIR, "audit_summary.json"), "w") as f:
            json.dump(summary_dict, f, indent=2)
            
        st.success("PySpark Audit Engine run completed successfully! Data updated.")

# Page Header
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-weight:700; font-size:2.5rem; letter-spacing:-0.05rem;">🔬 PBM Claims Audit & AI Assistant</h1>
    <p style="margin:0.5rem 0 0 0; font-weight:300; opacity:0.85; font-size:1.1rem;">
        Enterprise PBM compliance and financial recovery platform powered by AI and Databricks
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=400&q=80", use_column_width=True, caption="Next Generation Claims Auditing Best Practices")
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ System Controls")

if st.sidebar.button("🔄 Generate Synthetic Claims", help="Re-generate mock claims & drug reference databases"):
    with st.spinner("Generating data..."):
        run_generation(DATA_DIR)
        st.sidebar.success("Synthetic claims data generated!")
        st.rerun()

# Check if config exists, if not write default
if not os.path.exists(os.path.join(CONFIG_DIR, "benefit_design.json")):
    # Trigger generation which creates config falls
    run_generation(DATA_DIR)

if is_data_available():
    if st.sidebar.button("🚀 Run PySpark Audit Engine", help="Execute 6 Next Generation audit tests on claim files"):
        run_pbm_audit()
        st.rerun()
else:
    st.sidebar.warning("⚠️ No claims data generated yet. Click 'Generate Synthetic Claims' above.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔑 API Configurations")
api_key_input = st.sidebar.text_input("OpenAI API Key (Optional)", type="password", help="Enables conversational AI auditor chat and document parser")

if api_key_input:
    os.environ["OPENAI_API_KEY"] = api_key_input

# Main Tabs
tab_exec, tab_tests, tab_parser, tab_chat = st.tabs([
    "📊 Executive Summary", 
    "🕵️ Audit Tests Detail", 
    "📄 AI Contract Parser", 
    "💬 AI Auditor Chat"
])

# ================= TAB: Executive Summary =================
with tab_exec:
    if not is_audit_results_available():
        st.info("👋 Welcome! Click 'Run PySpark Audit Engine' in the sidebar to perform the initial audit calculations.")
    else:
        # Load summary
        with open(os.path.join(AUDIT_RESULTS_DIR, "audit_summary.json"), "r") as f:
            summary = json.load(f)
            
        total_claims = summary["total_claims"]
        flagged_claims = summary["total_flagged_claims"]
        recovery_impact = summary["total_financial_impact"]
        rebates_yield = summary["expected_rebate_yield"]
        
        # Premium Metric Cards
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        with m_col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Claims Audited</div>
                <div class="kpi-value">{total_claims:,}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: #e11d48;">
                <div class="kpi-title">Flagged Audit Exceptions</div>
                <div class="kpi-value" style="color: #e11d48;">{flagged_claims:,}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col3:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: #16a34a;">
                <div class="kpi-title">Potential Recoverable Funds</div>
                <div class="kpi-value" style="color: #16a34a;">${recovery_impact:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col4:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: #8b5cf6;">
                <div class="kpi-title">Guaranteed Rebates Yield</div>
                <div class="kpi-value" style="color: #8b5cf6;">${rebates_yield:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Charts section
        c_col1, c_col2 = st.columns([1, 1])
        
        test_counts = summary["test_counts_and_impacts"]
        chart_data = []
        for test, val in test_counts.items():
            chart_data.append({"Audit Test": test, "Count": val["count"], "Financial Impact ($)": val["impact"]})
            
        df_chart = pd.DataFrame(chart_data)
        
        with c_col1:
            st.markdown("### 📊 Exception Costs by Audit Test")
            if not df_chart.empty:
                fig_bar = px.bar(
                    df_chart, 
                    x="Financial Impact ($)", 
                    y="Audit Test", 
                    orientation='h',
                    color="Financial Impact ($)",
                    color_continuous_scale=px.colors.sequential.Sunsetdark,
                    text_auto='.2s'
                )
                fig_bar.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.write("No findings recorded.")
                
        with c_col2:
            st.markdown("### 🍕 Share of Flags by Category")
            if not df_chart.empty:
                fig_donut = px.pie(
                    df_chart, 
                    names="Audit Test", 
                    values="Count", 
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_donut.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_donut, use_container_width=True)
            else:
                st.write("No findings recorded.")

        # Detailed Table Summary
        st.markdown("### 📋 Audit Test Category Breakdown")
        st.table(df_chart.style.format({
            "Count": "{:,}",
            "Financial Impact ($)": "${:,.2f}"
        }))

# ================= TAB: Audit Tests Detail =================
with tab_tests:
    if not is_audit_results_available():
        st.info("Please run the PySpark Audit Engine from the sidebar to view detailed exceptions.")
    else:
        # Load claims CSV
        flagged_claims_df = pd.read_csv(os.path.join(AUDIT_RESULTS_DIR, "flagged_claims.csv"))
        
        st.markdown("### 🕵️ Audit Test Result Drilldown")
        
        test_options = ["All Exceptions"] + list(flagged_claims_df["audit_test"].unique())
        selected_test = st.selectbox("Select Audit Test to Inspect:", test_options)
        
        if selected_test == "All Exceptions":
            filtered_df = flagged_claims_df
        else:
            filtered_df = flagged_claims_df[flagged_claims_df["audit_test"] == selected_test]
            
        st.markdown(f"**Found {len(filtered_df):,} flags** matching criteria:")
        
        # Display data
        st.dataframe(
            filtered_df[["claim_id", "member_id", "ndc", "fill_date", "awp_billed", "copay_paid", "pbm_paid", "financial_impact", "audit_notes"]],
            column_config={
                "claim_id": "Claim ID",
                "member_id": "Member ID",
                "ndc": "NDC Code",
                "fill_date": "Fill Date",
                "awp_billed": st.column_config.NumberColumn("Billed AWP", format="$%.2f"),
                "copay_paid": st.column_config.NumberColumn("Copay Paid", format="$%.2f"),
                "pbm_paid": st.column_config.NumberColumn("PBM Paid", format="$%.2f"),
                "financial_impact": st.column_config.NumberColumn("Financial Overpayment", format="$%.2f"),
                "audit_notes": "Audit Notes"
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Rebates Reconciliation Table
        st.markdown("### 💵 PBM Rebate Guarantee Reconciliation Detail")
        if os.path.exists(os.path.join(AUDIT_RESULTS_DIR, "rebate_reconciliation.csv")):
            rebates_df = pd.read_csv(os.path.join(AUDIT_RESULTS_DIR, "rebate_reconciliation.csv"))
            st.dataframe(
                rebates_df,
                column_config={
                    "rebate_tier": "Rebate Channel/Drug Tier",
                    "claim_count": "Claim Count",
                    "guaranteed_rate_per_claim": st.column_config.NumberColumn("Guaranteed Rate / Claim", format="$%.2f"),
                    "expected_rebate_yield": st.column_config.NumberColumn("Expected Rebate Yield", format="$%.2f"),
                    "total_awp_spend": st.column_config.NumberColumn("Total Billed AWP Spend", format="$%.2f"),
                    "total_pbm_spend": st.column_config.NumberColumn("Total PBM Paid Spend", format="$%.2f"),
                },
                use_container_width=True,
                hide_index=True
            )

# ================= TAB: AI Contract Parser =================
with tab_parser:
    st.markdown("### 📄 LLM Contract Parsing & Parameter Extraction")
    st.markdown(
        "Upload a PBM Contract summary or Summary Plan Description (SPD), "
        "and let the AI parse benefit rules (copays, DAW rules, rebates) automatically."
    )
    
    # Contract textarea
    contract_input = st.text_area(
        "PBM Summary Plan Description / Agreement Contract Text:", 
        value=generate_sample_contract_text(),
        height=350
    )
    
    col_parser_1, col_parser_2 = st.columns([1, 1])
    
    with col_parser_1:
        if st.button("🧠 Parse Contract with AI", use_container_width=True):
            parser = AIContractParser(api_key=api_key_input)
            with st.spinner("Analyzing text with AI Contract Parser..."):
                parsed_json = parser.parse_contract(contract_input)
                
                # Write to config file
                with open(os.path.join(CONFIG_DIR, "benefit_design.json"), "w") as f:
                    json.dump(parsed_json, f, indent=2)
                    
                st.success("Successfully parsed contract text! JSON configuration updated below.")
                st.json(parsed_json)
                
    with col_parser_2:
        st.markdown("#### ⚙️ Current Active Benefit Design Rules Config")
        config_path = os.path.join(CONFIG_DIR, "benefit_design.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                active_config = json.load(f)
            st.json(active_config)
        else:
            st.info("No active config file found. Run parser above or check system directories.")

# ================= TAB: AI Auditor Chat =================
with tab_chat:
    st.markdown("### 💬 Conversational AI Auditor")
    st.markdown(
        "Interact with our specialized AI Auditor Assistant to ask questions about the claims audit findings, "
        "reconcile guarantees, or identify cost-saving highlights."
    )
    
    if not is_audit_results_available():
        st.warning("Please run the PySpark Audit Engine from the sidebar to populate audit data before chatting.")
    else:
        # Load summary for context
        with open(os.path.join(AUDIT_RESULTS_DIR, "audit_summary.json"), "r") as f:
            summary = json.load(f)
            
        # Instantiate assistant
        assistant = AIAuditorAssistant(summary_dict=summary, api_key=api_key_input)
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I am your AI Claims Auditor. Ask me anything about the audit results, recovery ROI, or performance anomalies."}
            ]
            
        # Display chat messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        # User input
        if prompt := st.chat_input("Ask a question (e.g., 'What is the total potential savings?' or 'Explain the DAW bypass findings'):"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            # Get response
            with st.chat_message("assistant"):
                with st.spinner("AI Auditor thinking..."):
                    response = assistant.ask(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
