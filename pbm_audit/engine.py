import os
import json
import warnings
import pandas as pd

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    HAS_PYSPARK = True
except ImportError:
    HAS_PYSPARK = False

class PBMAuditEngine:
    """Hybrid Audit Engine supporting PySpark (for Databricks) and Pandas fallback (for local Java-less run)."""
    
    def __init__(self, spark_session=None):
        self.use_spark = False
        self.spark = None
        
        if HAS_PYSPARK:
            try:
                # Attempt to get or create Spark session
                self.spark = spark_session or SparkSession.builder \
                    .appName("PBMAuditEngine") \
                    .config("spark.sql.shuffle.partitions", "2") \
                    .getOrCreate()
                # Test if Spark context works (forces gateway initialization check)
                self.spark.sparkContext.version
                self.use_spark = True
                print("PBMAuditEngine: Spark Session successfully initialized. Using PySpark mode.")
            except Exception as e:
                warnings.warn(
                    f"Spark initialization failed (likely due to missing Java Runtime): {e}. "
                    "Falling back to pure Pandas mode."
                )
        else:
            print("PBMAuditEngine: pyspark not installed. Using Pandas mode.")
            
    def load_data(self, claims_path, drugs_path, members_path):
        """Loads claims, reference drugs, and member databases based on execution mode."""
        if self.use_spark:
            claims_df = self.spark.read.csv(claims_path, header=True, inferSchema=True)
            drugs_df = self.spark.read.csv(drugs_path, header=True, inferSchema=True)
            members_df = self.spark.read.csv(members_path, header=True, inferSchema=True)
            
            # Parse date columns in Spark
            claims_df = claims_df.withColumn("fill_date", F.to_date(F.col("fill_date"), "yyyy-MM-dd"))
            members_df = members_df.withColumn("coverage_start_date", F.to_date(F.col("coverage_start_date"), "yyyy-MM-dd"))
            members_df = members_df.withColumn("coverage_end_date", F.to_date(F.col("coverage_end_date"), "yyyy-MM-dd"))
            return claims_df, drugs_df, members_df
        else:
            claims_df = pd.read_csv(claims_path)
            drugs_df = pd.read_csv(drugs_path)
            members_df = pd.read_csv(members_path)
            
            # Parse date columns in Pandas
            claims_df["fill_date"] = pd.to_datetime(claims_df["fill_date"]).dt.date
            members_df["coverage_start_date"] = pd.to_datetime(members_df["coverage_start_date"]).dt.date
            members_df["coverage_end_date"] = pd.to_datetime(members_df["coverage_end_date"]).dt.date
            return claims_df, drugs_df, members_df

    # ================= PUBLIC ROUTING METHODS =================

    def audit_invalid_ndcs(self, claims_df, drugs_df):
        """Milliman Test 1: Invalid NDCs (routed dynamically)."""
        if self.use_spark:
            return self._spark_audit_invalid_ndcs(claims_df, drugs_df)
        else:
            return self._pandas_audit_invalid_ndcs(claims_df, drugs_df)

    def audit_questionable_awp(self, claims_df, drugs_df):
        """Milliman Test 2: Questionable AWP (routed dynamically)."""
        if self.use_spark:
            return self._spark_audit_questionable_awp(claims_df, drugs_df)
        else:
            return self._pandas_audit_questionable_awp(claims_df, drugs_df)

    def audit_daw_penalties(self, claims_df, drugs_df, config):
        """Milliman Test 3: DAW Penalty Bypass (routed dynamically)."""
        if self.use_spark:
            return self._spark_audit_daw_penalties(claims_df, drugs_df, config)
        else:
            return self._pandas_audit_daw_penalties(claims_df, drugs_df, config)

    def audit_incorrect_copays(self, claims_df, config):
        """Milliman Test 4: Incorrect Copays (routed dynamically)."""
        if self.use_spark:
            return self._spark_audit_incorrect_copays(claims_df, config)
        else:
            return self._pandas_audit_incorrect_copays(claims_df, config)

    def audit_duplicate_claims(self, claims_df):
        """Milliman Test 5: Duplicate Claims (routed dynamically)."""
        if self.use_spark:
            return self._spark_audit_duplicate_claims(claims_df)
        else:
            return self._pandas_audit_duplicate_claims(claims_df)

    def audit_refill_too_soon(self, claims_df, config):
        """Milliman Test 6: Refill-Too-Soon (routed dynamically)."""
        if self.use_spark:
            return self._spark_audit_refill_too_soon(claims_df, config)
        else:
            return self._pandas_audit_refill_too_soon(claims_df, config)

    # ================= PYSPARK IMPLEMENTATIONS =================

    def _spark_audit_invalid_ndcs(self, claims_df, drugs_df):
        joined = claims_df.alias("c").join(
            drugs_df.alias("d"),
            F.col("c.ndc") == F.col("d.ndc"),
            "left"
        )
        flagged = joined.filter(F.col("d.ndc").isNull()) \
            .withColumn("audit_test", F.lit("Invalid NDC")) \
            .withColumn("financial_impact", F.col("c.pbm_paid")) \
            .withColumn("audit_notes", F.concat(F.lit("Submitted NDC "), F.col("c.ndc"), F.lit(" not found in drug database.")))
        return flagged.select("c.claim_id", "c.member_id", "c.ndc", "c.fill_date", "c.awp_billed", "c.copay_paid", "c.pbm_paid", "audit_test", "financial_impact", "audit_notes")

    def _spark_audit_questionable_awp(self, claims_df, drugs_df):
        joined = claims_df.alias("c").join(drugs_df.alias("d"), F.col("c.ndc") == F.col("d.ndc"), "inner")
        joined = joined.withColumn("expected_awp", F.round(F.col("c.quantity") * F.col("d.awp_per_unit"), 2))
        joined = joined.withColumn("awp_diff", F.round(F.col("c.awp_billed") - F.col("expected_awp"), 2))
        joined = joined.withColumn("pct_diff", F.abs(F.col("awp_diff") / F.col("expected_awp")))
        
        flagged = joined.filter(F.col("pct_diff") > 0.01) \
            .withColumn("audit_test", F.lit("Questionable AWP")) \
            .withColumn("financial_impact", F.col("awp_diff")) \
            .withColumn("audit_notes", F.concat(F.lit("Billed AWP $"), F.col("c.awp_billed"), F.lit(" is higher than reference expected AWP $"), F.col("expected_awp")))
        return flagged.select("c.claim_id", "c.member_id", "c.ndc", "c.fill_date", "c.awp_billed", "c.copay_paid", "c.pbm_paid", "audit_test", "financial_impact", "audit_notes")

    def _spark_audit_daw_penalties(self, claims_df, drugs_df, config):
        daw_codes = config["daw_penalty_policy"]["daw_codes_subject_to_penalty"]
        copays = config["copay_structure"]
        brand_claims = claims_df.filter((F.col("daw_code").isin(daw_codes)) & (F.col("drug_type") == "brand"))
        generics_df = drugs_df.filter(F.col("is_generic") == True)
        
        joined = brand_claims.alias("c") \
            .join(drugs_df.alias("brand_d"), F.col("c.ndc") == F.col("brand_d.ndc"), "inner") \
            .join(generics_df.alias("gen_d"), F.col("brand_d.generic_for") == F.col("gen_d.generic_for"), "left")
            
        joined = joined.filter(F.col("gen_d.ndc").isNotNull())
        joined = joined.withColumn("brand_cost", F.col("c.quantity") * F.col("brand_d.awp_per_unit"))
        joined = joined.withColumn("generic_cost", F.col("c.quantity") * F.col("gen_d.awp_per_unit"))
        joined = joined.withColumn("cost_difference", F.round(F.col("brand_cost") - F.col("generic_cost"), 2))
        
        resolved_expected_brand_copay = F.when(F.col("c.channel") == "retail", copays["retail_retail"]["brand"]) \
            .when(F.col("c.channel") == "mail", copays["mail_order"]["brand"]) \
            .otherwise(copays["specialty"]["brand"])
            
        joined = joined.withColumn("expected_copay_with_penalty", resolved_expected_brand_copay + F.col("cost_difference"))
        joined = joined.withColumn("bypassed_penalty", F.round(F.col("expected_copay_with_penalty") - F.col("c.copay_paid"), 2))
        
        flagged = joined.filter(F.col("bypassed_penalty") > 1.00) \
            .withColumn("audit_test", F.lit("DAW Penalty Bypass")) \
            .withColumn("financial_impact", F.col("bypassed_penalty")) \
            .withColumn("audit_notes", F.concat(F.lit("DAW penalty of $"), F.col("bypassed_penalty"), F.lit(" not charged to member for brand drug with generic available.")))
        return flagged.select("c.claim_id", "c.member_id", "c.ndc", "c.fill_date", "c.awp_billed", "c.copay_paid", "c.pbm_paid", "audit_test", "financial_impact", "audit_notes")

    def _spark_audit_incorrect_copays(self, claims_df, config):
        copays = config["copay_structure"]
        expected_copay_col = F.when(
            (F.col("channel") == "retail") & (F.col("drug_type") == "generic"), F.lit(copays["retail_retail"]["generic"])
        ).when(
            (F.col("channel") == "retail") & (F.col("drug_type") == "brand"), F.lit(copays["retail_retail"]["brand"])
        ).when(
            (F.col("channel") == "mail") & (F.col("drug_type") == "generic"), F.lit(copays["mail_order"]["generic"])
        ).when(
            (F.col("channel") == "mail") & (F.col("drug_type") == "brand"), F.lit(copays["mail_order"]["brand"])
        ).when(
            (F.col("channel") == "specialty") & (F.col("drug_type") == "generic"), F.lit(copays["specialty"]["generic"])
        ).when(
            (F.col("channel") == "specialty") & (F.col("drug_type") == "brand"), F.lit(copays["specialty"]["brand"])
        ).otherwise(F.lit(0.0))
        
        claims_with_expected = claims_df.withColumn("expected_copay", expected_copay_col)
        claims_with_expected = claims_with_expected.withColumn("copay_diff", F.round(F.col("expected_copay") - F.col("copay_paid"), 2))
        
        flagged = claims_with_expected.filter((F.col("copay_diff") != 0.0) & (F.col("daw_code") == 0)) \
            .withColumn("audit_test", F.lit("Incorrect Copay")) \
            .withColumn("financial_impact", F.col("copay_diff")) \
            .withColumn("audit_notes", F.concat(F.lit("Member paid $"), F.col("copay_paid"), F.lit(" instead of expected copay $"), F.col("expected_copay")))
        return flagged.select("claim_id", "member_id", "ndc", "fill_date", "awp_billed", "copay_paid", "pbm_paid", "audit_test", "financial_impact", "audit_notes")

    def _spark_audit_duplicate_claims(self, claims_df):
        window_spec = Window.partitionBy("member_id", "ndc", "fill_date").orderBy("claim_id")
        claims_with_rn = claims_df.withColumn("rn", F.row_number().over(window_spec))
        
        flagged = claims_with_rn.filter(F.col("rn") > 1) \
            .withColumn("audit_test", F.lit("Duplicate Claim")) \
            .withColumn("financial_impact", F.col("pbm_paid")) \
            .withColumn("audit_notes", F.lit("Multiple claims processed for same member, drug, and fill date."))
        return flagged.select("claim_id", "member_id", "ndc", "fill_date", "awp_billed", "copay_paid", "pbm_paid", "audit_test", "financial_impact", "audit_notes")

    def _spark_audit_refill_too_soon(self, claims_df, config):
        threshold = config["refill_too_soon"]["threshold_percentage"]
        window_spec = Window.partitionBy("member_id", "ndc").orderBy("fill_date", "claim_id")
        
        processed = claims_df \
            .withColumn("prev_fill_date", F.lag("fill_date", 1).over(window_spec)) \
            .withColumn("prev_days_supply", F.lag("days_supply", 1).over(window_spec))
            
        processed = processed.withColumn("days_elapsed", F.datediff(F.col("fill_date"), F.col("prev_fill_date")))
        processed = processed.withColumn("min_required_days", F.round(F.col("prev_days_supply") * threshold, 0))
        
        flagged = processed.filter((F.col("prev_fill_date").isNotNull()) & (F.col("days_elapsed") < F.col("min_required_days"))) \
            .withColumn("audit_test", F.lit("Refill Too Soon")) \
            .withColumn("financial_impact", F.col("pbm_paid")) \
            .withColumn("audit_notes", F.concat(
                F.lit("Refill occurred after "), F.col("days_elapsed"), 
                F.lit(" days, but required at least "), F.col("min_required_days"), 
                F.lit(" days ("), F.lit(int(threshold * 100)), F.lit("% of previous "), 
                F.col("prev_days_supply"), F.lit("-day supply).")
            ))
        return flagged.select("claim_id", "member_id", "ndc", "fill_date", "awp_billed", "copay_paid", "pbm_paid", "audit_test", "financial_impact", "audit_notes")

    # ================= PANDAS IMPLEMENTATIONS (FALLBACK) =================

    def _pandas_audit_invalid_ndcs(self, claims_df, drugs_df):
        merged = claims_df.merge(drugs_df, on="ndc", how="left")
        flagged = merged[merged["drug_name"].isna()].copy()
        
        flagged["audit_test"] = "Invalid NDC"
        flagged["financial_impact"] = flagged["pbm_paid"].astype(float)
        flagged["audit_notes"] = flagged["ndc"].apply(lambda ndc: f"Submitted NDC {ndc} not found in drug database.")
        
        return flagged[["claim_id", "member_id", "ndc", "fill_date", "awp_billed", "copay_paid", "pbm_paid", "audit_test", "financial_impact", "audit_notes"]]

    def _pandas_audit_questionable_awp(self, claims_df, drugs_df):
        merged = claims_df.merge(drugs_df, on="ndc", how="inner")
        
        # Enforce float conversion to avoid Arrow type errors in Pandas 3
        qty = merged["quantity"].astype(float)
        ref_awp = merged["awp_per_unit"].astype(float)
        billed_awp = merged["awp_billed"].astype(float)
        
        merged["expected_awp"] = round(qty * ref_awp, 2)
        merged["awp_diff"] = round(billed_awp - merged["expected_awp"], 2)
        merged["pct_diff"] = (merged["awp_diff"] / merged["expected_awp"]).abs()
        
        flagged = merged[merged["pct_diff"] > 0.01].copy()
        flagged["audit_test"] = "Questionable AWP"
        flagged["financial_impact"] = flagged["awp_diff"].astype(float)
        flagged["audit_notes"] = flagged.apply(
            lambda r: f"Billed AWP ${r['awp_billed']:.2f} is higher than reference expected AWP ${r['expected_awp']:.2f}", axis=1
        )
        return flagged[["claim_id", "member_id", "ndc", "fill_date", "awp_billed", "copay_paid", "pbm_paid", "audit_test", "financial_impact", "audit_notes"]]

    def _pandas_audit_daw_penalties(self, claims_df, drugs_df, config):
        daw_codes = config["daw_penalty_policy"]["daw_codes_subject_to_penalty"]
        copays = config["copay_structure"]
        
        brand_claims = claims_df[(claims_df["daw_code"].isin(daw_codes)) & (claims_df["drug_type"] == "brand")].copy()
        
        # Merge brand details
        merged = brand_claims.merge(drugs_df[["ndc", "drug_name", "generic_for", "awp_per_unit"]], on="ndc", how="inner")
        
        # Get generics reference
        generics = drugs_df[drugs_df["is_generic"] == True][["generic_for", "awp_per_unit", "ndc"]].rename(
            columns={"awp_per_unit": "gen_awp_per_unit", "ndc": "gen_ndc"}
        )
        
        # Join on generic_for drug name
        merged = merged.merge(generics, on="generic_for", how="left")
        merged = merged[merged["gen_ndc"].notna()].copy()
        
        brand_cost = merged["quantity"].astype(float) * merged["awp_per_unit"].astype(float)
        gen_cost = merged["quantity"].astype(float) * merged["gen_awp_per_unit"].astype(float)
        merged["cost_difference"] = round(brand_cost - gen_cost, 2).astype(float)
        
        def get_expected_brand_copay(channel):
            if channel == "retail":
                return float(copays["retail_retail"]["brand"])
            elif channel == "mail":
                return float(copays["mail_order"]["brand"])
            else:
                return float(copays["specialty"]["brand"])
                
        merged["base_brand_copay"] = merged["channel"].apply(get_expected_brand_copay).astype(float)
        merged["expected_copay_with_penalty"] = merged["base_brand_copay"] + merged["cost_difference"]
        merged["bypassed_penalty"] = round(merged["expected_copay_with_penalty"] - merged["copay_paid"].astype(float), 2)
        
        flagged = merged[merged["bypassed_penalty"] > 1.00].copy()
        flagged["audit_test"] = "DAW Penalty Bypass"
        flagged["financial_impact"] = flagged["bypassed_penalty"].astype(float)
        flagged["audit_notes"] = flagged.apply(
            lambda r: f"DAW penalty of ${r['bypassed_penalty']:.2f} not charged to member for brand drug with generic available.", axis=1
        )
        return flagged[["claim_id", "member_id", "ndc", "fill_date", "awp_billed", "copay_paid", "pbm_paid", "audit_test", "financial_impact", "audit_notes"]]

    def _pandas_audit_incorrect_copays(self, claims_df, config):
        copays = config["copay_structure"]
        
        def resolve_expected(row):
            ch = row["channel"]
            dt = row["drug_type"]
            if ch == "retail":
                return float(copays["retail_retail"]["generic"]) if dt == "generic" else float(copays["retail_retail"]["brand"])
            elif ch == "mail":
                return float(copays["mail_order"]["generic"]) if dt == "generic" else float(copays["mail_order"]["brand"])
            elif ch == "specialty":
                return float(copays["specialty"]["generic"]) if dt == "generic" else float(copays["specialty"]["brand"])
            return 0.0

        claims_with_expected = claims_df.copy()
        claims_with_expected["expected_copay"] = claims_with_expected.apply(resolve_expected, axis=1).astype(float)
        claims_with_expected["copay_diff"] = round(claims_with_expected["expected_copay"] - claims_with_expected["copay_paid"].astype(float), 2)
        
        flagged = claims_with_expected[(claims_with_expected["copay_diff"] != 0.0) & (claims_with_expected["daw_code"] == 0)].copy()
        flagged["audit_test"] = "Incorrect Copay"
        flagged["financial_impact"] = flagged["copay_diff"].astype(float)
        flagged["audit_notes"] = flagged.apply(
            lambda r: f"Member paid ${r['copay_paid']:.2f} instead of expected copay ${r['expected_copay']:.2f}", axis=1
        )
        return flagged[["claim_id", "member_id", "ndc", "fill_date", "awp_billed", "copay_paid", "pbm_paid", "audit_test", "financial_impact", "audit_notes"]]

    def _pandas_audit_duplicate_claims(self, claims_df):
        sorted_claims = claims_df.sort_values("claim_id").copy()
        duplicates_mask = sorted_claims.duplicated(subset=["member_id", "ndc", "fill_date"], keep="first")
        
        flagged = sorted_claims[duplicates_mask].copy()
        flagged["audit_test"] = "Duplicate Claim"
        flagged["financial_impact"] = flagged["pbm_paid"].astype(float)
        flagged["audit_notes"] = "Multiple claims processed for same member, drug, and fill date."
        
        return flagged[["claim_id", "member_id", "ndc", "fill_date", "awp_billed", "copay_paid", "pbm_paid", "audit_test", "financial_impact", "audit_notes"]]

    def _pandas_audit_refill_too_soon(self, claims_df, config):
        threshold = config["refill_too_soon"]["threshold_percentage"]
        sorted_claims = claims_df.sort_values(["member_id", "ndc", "fill_date", "claim_id"]).copy()
        
        # Calculate shifted fields
        sorted_claims["prev_fill_date"] = sorted_claims.groupby(["member_id", "ndc"])["fill_date"].shift(1)
        sorted_claims["prev_days_supply"] = sorted_claims.groupby(["member_id", "ndc"])["days_supply"].shift(1)
        
        # Drop rows where there is no previous claim
        processed = sorted_claims[sorted_claims["prev_fill_date"].notna()].copy()
        
        # Calculate difference in days
        processed["days_elapsed"] = (processed["fill_date"] - processed["prev_fill_date"]).apply(lambda x: x.days).astype(float)
        processed["min_required_days"] = round(processed["prev_days_supply"].astype(float) * threshold, 0).astype(float)
        
        flagged = processed[processed["days_elapsed"] < processed["min_required_days"]].copy()
        flagged["audit_test"] = "Refill Too Soon"
        flagged["financial_impact"] = flagged["pbm_paid"].astype(float)
        flagged["audit_notes"] = flagged.apply(
            lambda r: f"Refill occurred after {r['days_elapsed']:.0f} days, but required at least {r['min_required_days']:.0f} days ({int(threshold*100)}% of previous {r['prev_days_supply']:.0f}-day supply).", axis=1
        )
        return flagged[["claim_id", "member_id", "ndc", "fill_date", "awp_billed", "copay_paid", "pbm_paid", "audit_test", "financial_impact", "audit_notes"]]

    # ================= GENERAL ENTRYPOINTS =================

    def run_all_audits(self, claims_df, drugs_df, config):
        """Runs all 6 audit tests, consolidates results into a single output (Spark or Pandas)."""
        if self.use_spark:
            df_invalid_ndc = self._spark_audit_invalid_ndcs(claims_df, drugs_df)
            valid_claims_df = claims_df.join(drugs_df, "ndc", "inner").select(claims_df["*"])
            
            df_quest_awp = self._spark_audit_questionable_awp(valid_claims_df, drugs_df)
            df_daw = self._spark_audit_daw_penalties(valid_claims_df, drugs_df, config)
            df_copay = self._spark_audit_incorrect_copays(valid_claims_df, config)
            df_dups = self._spark_audit_duplicate_claims(valid_claims_df)
            df_rts = self._spark_audit_refill_too_soon(valid_claims_df, config)
            
            consolidated_df = df_invalid_ndc \
                .union(df_quest_awp) \
                .union(df_daw) \
                .union(df_copay) \
                .union(df_dups) \
                .union(df_rts)
            return consolidated_df
        else:
            df_invalid_ndc = self._pandas_audit_invalid_ndcs(claims_df, drugs_df)
            
            # Filter claims with valid NDCs for tests 2-6
            valid_claims_df = claims_df[claims_df["ndc"].isin(drugs_df["ndc"])].copy()
            
            df_quest_awp = self._pandas_audit_questionable_awp(valid_claims_df, drugs_df)
            df_daw = self._pandas_audit_daw_penalties(valid_claims_df, drugs_df, config)
            df_copay = self._pandas_audit_incorrect_copays(valid_claims_df, config)
            df_dups = self._pandas_audit_duplicate_claims(valid_claims_df)
            df_rts = self._pandas_audit_refill_too_soon(valid_claims_df, config)
            
            consolidated_df = pd.concat([df_invalid_ndc, df_quest_awp, df_daw, df_copay, df_dups, df_rts], ignore_index=True)
            return consolidated_df

    def perform_rebate_reconciliation(self, claims_df, config):
        """Performs PBM Rebate Reconciliation (Spark or Pandas)."""
        rebates = config["rebate_guarantees"]
        
        if self.use_spark:
            # PySpark implementation
            classified = claims_df.withColumn("channel_tier", F.concat(
                F.when(F.col("channel") == "retail", F.lit("retail_")).otherwise(F.lit("mail_")),
                F.col("drug_type")
            ))
            classified = classified.withColumn("channel_tier", F.when(
                F.col("channel") == "specialty", 
                F.concat(F.lit("specialty_"), F.col("drug_type"))
            ).otherwise(F.col("channel_tier")))
            
            summary = classified.groupBy("channel_tier").agg(
                F.count("claim_id").alias("claim_count"),
                F.sum("awp_billed").alias("total_awp_spend"),
                F.sum("pbm_paid").alias("total_pbm_spend")
            ).collect()
            
            reconciliation_results = []
            total_expected_rebate = 0.0
            
            for row in summary:
                tier = row["channel_tier"]
                count = row["claim_count"]
                guaranteed_rate = rebates.get(tier, 0.0)
                expected_rebate = count * guaranteed_rate
                total_expected_rebate += expected_rebate
                
                reconciliation_results.append({
                    "rebate_tier": tier,
                    "claim_count": count,
                    "guaranteed_rate_per_claim": guaranteed_rate,
                    "expected_rebate_yield": round(expected_rebate, 2),
                    "total_awp_spend": round(row["total_awp_spend"] or 0.0, 2),
                    "total_pbm_spend": round(row["total_pbm_spend"] or 0.0, 2)
                })
            return reconciliation_results, round(total_expected_rebate, 2)
        else:
            # Pandas implementation
            def get_tier(row):
                ch = row["channel"]
                dt = row["drug_type"]
                if ch == "specialty":
                    return f"specialty_{dt}"
                elif ch == "retail":
                    return f"retail_{dt}"
                else:
                    return f"mail_{dt}"
                    
            claims_df = claims_df.copy()
            claims_df["channel_tier"] = claims_df.apply(get_tier, axis=1)
            
            summary = claims_df.groupby("channel_tier").agg(
                claim_count=("claim_id", "count"),
                total_awp_spend=("awp_billed", "sum"),
                total_pbm_spend=("pbm_paid", "sum")
            ).reset_index()
            
            reconciliation_results = []
            total_expected_rebate = 0.0
            
            for _, row in summary.iterrows():
                tier = row["channel_tier"]
                count = row["claim_count"]
                guaranteed_rate = rebates.get(tier, 0.0)
                expected_rebate = count * guaranteed_rate
                total_expected_rebate += expected_rebate
                
                reconciliation_results.append({
                    "rebate_tier": tier,
                    "claim_count": int(count),
                    "guaranteed_rate_per_claim": guaranteed_rate,
                    "expected_rebate_yield": round(expected_rebate, 2),
                    "total_awp_spend": round(float(row["total_awp_spend"]), 2),
                    "total_pbm_spend": round(float(row["total_pbm_spend"]), 2)
                })
            return reconciliation_results, round(total_expected_rebate, 2)
