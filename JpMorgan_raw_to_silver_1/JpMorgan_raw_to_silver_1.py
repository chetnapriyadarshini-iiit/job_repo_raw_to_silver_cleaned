import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql import functions as F, types as T
from pyspark.sql.functions import col, trim, upper, when, current_date
from pyspark.sql.types import IntegerType, LongType

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args[JOB_NAME], args)

#production level , 1st data quality and preprocessing flag check and then implementing the checked flag for changing 
#ex if only flaged of outlier , then outlier removing fuction will execute . 

# =========================
# Preprocessing Functions
# =========================


UNIQUE_KEYS = ["employee_id", "employee_name"]
CATEGORICALS = ["department", "state"]
NUMERICALS   = ["salary", "age", "bonus"]

BUCKET_BASE = "majorproject02-jpmorgan-bronze/"
BRONZE_PARQUET = "fromdf-jpmorganoffice_parquet" #Spark df in parquet format
GLUE_DB = "majorproject02-jpmorgan_db"

silvermain_glue_path1 = "s3://majorproject02-jpmorgan-silver/partitioned_by_department/"
silvermain_glue_path2 = "s3://majorproject02-jpmorgan-silver/partitioned_by_state_department/"
silver_spark_path = "s3://majorproject02-jpmorgan-silver/cleaned_data_stored_as_spark_df/"

def sink_with_catalog(path: str, table_name: str, partition_keys: list, ctx_name: str):
    """
    Create a Glue catalog sink that:
      - writes Parquet to S3
      - updates/creates the Glue Catalog table (schema + partitions)
    partition_keys: list of partition columns ([] for non-partitioned KPIs)
    """
    sink = glueContext.getSink(
        path=path,
        connection_type="s3",
        updateBehavior="UPDATE_IN_DATABASE",      # merge schema/partitions if table exists
        partitionKeys=partition_keys,
        compression="snappy",
        enableUpdateCatalog=True,                 # <-- critical: tells Glue to maintain the Catalog table
        transformation_ctx=ctx_name
    )
    sink.setCatalogInfo(catalogDatabase=GLUE_DB, catalogTableName=table_name)
    sink.setFormat("glueparquet")                 # optimized writer for Parquet + Catalog
    return sink


# =========================
# FLAG DISPLAY FUNCTION
# =========================
def display_flags(cfg):
    print("🚩 CONFIGURED PREPROCESSING FLAGS:")
    print("=" * 50)
    
    flags = {
        "Structural": [
            ("snake_case", "Column name standardization"),
            ("trim_nulls", "String trimming & null handling"),
            ("dedup", "Duplicate removal")
        ],
        "Data Quality": [
            ("clean_numerics", "Numeric type conversion"),
            ("dq_rules", "Data quality rules & flagging"),
            ("detect_outliers", "Outlier detection"),
            ("remove_outliers", "Outlier removal")
        ],
        "Business Logic": [
            ("standardize_cats", "Categorical standardization"),
            ("data_enrichment", "Derived columns creation"),
            ("audit", "Audit fields addition")
        ]
    }
    
    for category, flag_list in flags.items():
        print(f"\n{category}:")
        for flag, description in flag_list:
            status = "✅ ENABLED" if cfg.get(flag) else "❌ DISABLED"
            print(f"  {flag:20} {status:15} - {description}")
    
    print("\n📋 CONFIGURATION PARAMETERS:")
    for key, value in cfg.items():
        if key not in [f[0] for f_list in flags.values() for f in f_list]:
            if key != "analysis_results":  # Skip large analysis results for display
                print(f"  {key:20} = {value}")
    
    print("=" * 50)
    
# =========================
# PREPROCESSING FUNCTIONS
# =========================

# --- 1) String hygiene: trim + empty->null + control chars
def fx_trim_and_nulls(df):
    print("🔧 Executing: String trimming & null conversion")
    for c in df.columns:
        # Trim and convert empty strings to null
        df = df.withColumn(c, F.when(F.trim(F.col(c)) == "", None).otherwise(F.trim(F.col(c))))
        # Remove control characters
        df = df.withColumn(c, F.regexp_replace(F.col(c), r"[\x00-\x1F\x7F]", ""))
    return df
    
# --- 2) Numeric cleaning: strip symbols -> cast with validation
def fx_clean_numerics(df, numeric_cols=NUMERICALS):
    print("🔧 Executing: Numeric cleaning & type conversion")
    
    # Register try_cast function if not available
    try:
        F.try_cast
    except AttributeError:
        def try_cast(col, to_type):
            return F.when(col.isNull(), F.lit(None)).otherwise(
                F.when(F.col(col).cast(to_type).isNotNull(), F.col(col).cast(to_type))
            )
    
    def clean(col):
        return F.regexp_replace(F.col(col), r"[^0-9\-\.]", "")
    
    for c in numeric_cols:
        if c in df.columns:
            # Clean and cast, set to null if conversion fails
            df = df.withColumn(c, 
                F.when(F.col(c).isNull(), F.lit(None))
                 .otherwise(F.when(F.col(c).cast(T.StringType()) == "", F.lit(None))
                 .otherwise(clean(c).cast(T.DoubleType())))
            )
    return df
    
# --- 3) Categorical standardization
def fx_standardize_cats(df, state_map=None, dept_map=None):
    print("🔧 Executing: Categorical standardization")
    
    if "state" in df.columns and state_map:
        # Create mapping expression
        mapping_list = []
        for k, v in state_map.items():
            mapping_list.extend([F.lit(k), F.lit(v)])
        mapping_expr = F.create_map(mapping_list)
        
        df = df.withColumn("state", 
            F.upper(F.coalesce(mapping_expr[F.upper(F.col("state"))], F.upper(F.col("state"))))
        )
    
    if "department" in df.columns and dept_map:
        # Create mapping expression
        mapping_list = []
        for k, v in dept_map.items():
            mapping_list.extend([F.lit(k), F.lit(v)])
        mapping_expr = F.create_map(mapping_list)
        
        df = df.withColumn("department", 
            F.coalesce(mapping_expr[F.col("department")], F.col("department"))
        )
    
    # Standardize all categoricals to uppercase
    for cat_col in CATEGORICALS:
        if cat_col in df.columns:
            df = df.withColumn(cat_col, F.upper(F.trim(F.col(cat_col))))
    
    return df
    
# --- 4) Domain/range checks with detailed DQ flags
def fx_dq_rules(df):
    print("🔧 Executing: Data quality rules & flagging")
    
    df = df.withColumn("dq_age_ok", 
        (F.col("age").isNull()) | ((F.col("age") >= 18) & (F.col("age") <= 70))
    ).withColumn("dq_salary_ok",
        (F.col("salary").isNull()) | ((F.col("salary") >= 1000) & (F.col("salary") <= 500000))
    ).withColumn("dq_bonus_ok",
        (F.col("bonus").isNull()) | ((F.col("bonus") >= 0) & (F.col("bonus") <= 200000))
    ).withColumn("dq_employee_id_ok",
        F.col("employee_id").isNotNull() & (F.col("employee_id") > 0)
    ).withColumn("dq_name_ok",
        F.col("employee_name").isNotNull() & (F.length(F.trim(F.col("employee_name"))) > 1)
    )
    
    # Overall DQ status
    dq_columns = [c for c in df.columns if c.startswith("dq_") and c.endswith("_ok")]
    dq_expr = F.greatest(*[F.col(c) for c in dq_columns])
    df = df.withColumn("dq_status", 
        F.when(dq_expr, F.lit("PASS")).otherwise(F.lit("FAIL"))
    ).withColumn("dq_score",
        F.round((sum(F.col(c).cast("int") for c in dq_columns) / F.lit(len(dq_columns))) * 100, 2)
    )
    
    return df
    
# --- 5) Outlier detection using IQR method
def fx_detect_outliers(df, numeric_cols=NUMERICALS):
    print("🔧 Executing: Outlier detection")
    
    for col_name in numeric_cols:
        if col_name in df.columns:
            # Calculate quartiles
            quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.05)
            if quantiles and len(quantiles) == 2:
                Q1, Q3 = quantiles
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df = df.withColumn(f"outlier_{col_name}",
                    ~F.col(col_name).between(lower_bound, upper_bound)
                )
    
    return df
    
# --- 6) Remove outliers (only if flag is True)
def fx_remove_outliers(df, numeric_cols=NUMERICALS):
    print("🔧 Executing: Outlier removal")
    
    outlier_cols = [f"outlier_{col}" for col in numeric_cols if f"outlier_{col}" in df.columns]
    if outlier_cols:
        # Remove rows that are outliers in any numeric column
        outlier_condition = ~F.greatest(*[F.col(col) for col in outlier_cols])
        initial_count = df.count()
        df = df.filter(outlier_condition)
        removed_count = initial_count - df.count()
        print(f"📊 Removed {removed_count} outlier records")
    
    return df
    
# --- 7) Deduplication
def fx_dedup(df, keys=UNIQUE_KEYS, ts_col=None):
    print("🔧 Executing: Deduplication")
    
    if ts_col and ts_col in df.columns:
        window = Window.partitionBy(*keys).orderBy(F.col(ts_col).desc())
        df = df.withColumn("__rn", F.row_number().over(window))\
               .filter(F.col("__rn") == 1)\
               .drop("__rn")
    else:
        initial_count = df.count()
        df = df.dropDuplicates(keys)
        final_count = df.count()
        print(f"📊 Removed {initial_count - final_count} duplicate records")
    
    return df
    
# --- 8) Audit fields
def fx_audit(df, source_system="jpmorgan_raw", batch_id=None):
    print("🔧 Executing: Adding audit fields")
    
    return (df
        .withColumn("source_system", F.lit(source_system))
        .withColumn("batch_id", F.lit(batch_id))
        .withColumn("ingestion_ts", F.current_timestamp())
        .withColumn("etl_version", F.lit("2.0"))
    )
    
# --- 9) Column normalization to snake_case
def fx_snake_case(df):
    print("🔧 Executing: Column name standardization")
    
    def to_snake_case(name):
        name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
        name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
        return name.lower().replace(" ", "_").replace("-", "_")
    
    mapping = {c: to_snake_case(c) for c in df.columns}
    for old, new in mapping.items():
        if old != new:
            df = df.withColumnRenamed(old, new)
    
    return df
    
# --- 10) Data enrichment (derived columns)
def fx_data_enrichment(df):
    print("🔧 Executing: Data enrichment")
    
    df = df.withColumn("total_compensation", 
        F.coalesce(F.col("salary"), F.lit(0)) + F.coalesce(F.col("bonus"), F.lit(0))
    ).withColumn("bonus_percentage",
        F.when(F.col("salary") > 0, (F.col("bonus") / F.col("salary")) * 100)
         .otherwise(F.lit(0))
    ).withColumn("age_group",
        F.when(F.col("age") < 25, "Under 25")
         .when(F.col("age") < 35, "25-34")
         .when(F.col("age") < 45, "35-44")
         .when(F.col("age") < 55, "45-54")
         .otherwise("55+")
    ).withColumn("compensation_tier",
        F.when(F.col("total_compensation") < 50000, "Entry")
         .when(F.col("total_compensation") < 100000, "Mid")
         .when(F.col("total_compensation") < 200000, "Senior")
         .otherwise("Executive")
    )
    
    return df
    
# =========================
# PIPELINE ORCHESTRATOR
# =========================
def run_pipeline(df, cfg):
    """
    Enhanced pipeline with comprehensive preprocessing flags
    """
    
    # Display configuration
    display_flags(cfg)
    
    print("\n🎯 STARTING PIPELINE EXECUTION")
    print("=" * 50)
    
    initial_count = df.count()
    print(f"📊 Initial record count: {initial_count}")
    
    # Track execution steps
    execution_steps = []
    
    # Pipeline execution
    if cfg.get("snake_case"):       
        df = fx_snake_case(df)
        execution_steps.append("✅ Column standardization")
    
    if cfg.get("trim_nulls"):       
        df = fx_trim_and_nulls(df)
        execution_steps.append("✅ String hygiene")
    
    if cfg.get("clean_numerics"):   
        df = fx_clean_numerics(df, NUMERICALS)
        execution_steps.append("✅ Numeric cleaning")
    
    if cfg.get("standardize_cats"): 
        df = fx_standardize_cats(df, cfg.get("state_map"), cfg.get("dept_map"))
        execution_steps.append("✅ Categorical standardization")
    
    if cfg.get("detect_outliers"):  
        df = fx_detect_outliers(df, NUMERICALS)
        execution_steps.append("✅ Outlier detection")
    
    if cfg.get("remove_outliers"):  
        df = fx_remove_outliers(df, NUMERICALS)
        execution_steps.append("✅ Outlier removal")
    
    if cfg.get("dq_rules"):         
        df = fx_dq_rules(df)
        execution_steps.append("✅ Data quality rules")
    
    if cfg.get("dedup"):            
        df = fx_dedup(df, UNIQUE_KEYS, cfg.get("ts_col"))
        execution_steps.append("✅ Deduplication")
    
    if cfg.get("data_enrichment"):  
        df = fx_data_enrichment(df)
        execution_steps.append("✅ Data enrichment")
    
    if cfg.get("audit"):            
        df = fx_audit(df, cfg.get("source_system", "raw"), cfg.get("batch_id"))
        execution_steps.append("✅ Audit fields")
    
    # Final statistics
    final_count = df.count()
    print("\n📊 PIPELINE EXECUTION SUMMARY:")
    print("=" * 50)
    for step in execution_steps:
        print(step)
    
    print(f"\n📈 RECORD COUNT: {initial_count} → {final_count}")
    print(f"📉 RECORDS PROCESSED: {initial_count - final_count}")
    
    if cfg.get("dq_rules") and "dq_status" in df.columns:
        dq_stats = df.groupBy("dq_status").count().collect()
        print("\n🔍 DATA QUALITY STATUS:")
        for row in dq_stats:
            print(f"  {row['dq_status']}: {row['count']} records")
    
    return df
    
# =========================
# PHASE 01 - FLAGGING & CONFIGURATION GENERATION
# =========================

def analyze_data_and_generate_flags(df):
    """
    Phase 01: Analyze data and generate dynamic configuration flags
    Returns: cfg_live (configuration dictionary)
    """
    print("🔍 PHASE 01: DATA ANALYSIS & FLAG GENERATION")
    print("=" * 60)
    
    # Initialize configuration
    cfg_live = {
        # Structural flags (defaults)
        "snake_case": True,
        "trim_nulls": True,
        "dedup": True,
        
        # Data quality flags
        "clean_numerics": True,
        "dq_rules": True,
        "detect_outliers": True,
        "remove_outliers": False,
        
        # Business logic flags
        "standardize_cats": True,
        "data_enrichment": True,
        "audit": True,
        
        # Parameters (will be populated dynamically)
        "ts_col": None,
        "state_map": {},
        "dept_map": {},
        "source_system": "jpmorgan_raw",
        "batch_id": f"batch-{F.current_timestamp().cast('string')}",
        
        # Analysis results (for reporting)
        "analysis_results": {}
    }
    
    analysis_results = {}
    
    # 1. Analyze null values
    print("📊 Analyzing null values...")
    null_analysis = {}
    for column in df.columns:
        null_count = df.filter(F.col(column).isNull()).count()
        total_count = df.count()
        null_percentage = (null_count / total_count) * 100 if total_count > 0 else 0
        null_analysis[column] = {
            "null_count": null_count,
            "null_percentage": round(null_percentage, 2)
        }
    
    analysis_results["null_analysis"] = null_analysis
    
    # Enable null trimming if any nulls found
    has_nulls = any(stats["null_count"] > 0 for stats in null_analysis.values())
    cfg_live["trim_nulls"] = has_nulls
    print(f"   - Nulls detected: {has_nulls} → trim_nulls: {has_nulls}")
    
    # 2. Analyze duplicates
    print("📊 Analyzing duplicates...")
    duplicate_count = df.count() - df.distinct().count()
    analysis_results["duplicate_analysis"] = {
        "duplicate_count": duplicate_count,
        "has_duplicates": duplicate_count > 0
    }
    
    cfg_live["dedup"] = duplicate_count > 0
    print(f"   - Duplicates: {duplicate_count} → dedup: {duplicate_count > 0}")
    
    # 3. Analyze numeric columns for cleaning needs
    print("📊 Analyzing numeric columns...")
    numeric_analysis = {}
    for col in NUMERICALS:
        if col in df.columns:
            # Check if contains non-numeric characters
            sample = df.select(col).filter(F.col(col).isNotNull()).limit(10).collect()
            has_special_chars = any(
                str(row[col]).strip() != "" and 
                not str(row[col]).replace('.', '').replace('-', '').replace('$', '').replace(',', '').isdigit()
                for row in sample if row[col] is not None
            )
            numeric_analysis[col] = {
                "has_special_chars": has_special_chars,
                "needs_cleaning": has_special_chars
            }
    
    analysis_results["numeric_analysis"] = numeric_analysis
    needs_numeric_cleaning = any(stats["needs_cleaning"] for stats in numeric_analysis.values())
    cfg_live["clean_numerics"] = needs_numeric_cleaning
    print(f"   - Needs numeric cleaning: {needs_numeric_cleaning} → clean_numerics: {needs_numeric_cleaning}")
    
    # 4. Analyze categorical columns for standardization
    print("📊 Analyzing categorical columns...")
    categorical_analysis = {}
    
    for col in CATEGORICALS:
        if col in df.columns:
            # Get value counts for analysis
            value_counts = df.filter(F.col(col).isNotNull()) \
                           .groupBy(col) \
                           .count() \
                           .orderBy(F.col("count").desc()) \
                           .limit(10) \
                           .collect()
            
            values = [row[col] for row in value_counts if row[col] is not None]
            has_variations = len(values) > 0 and any(
                len(str(val).strip()) != len(str(val)) or 
                str(val) != str(val).upper() 
                for val in values
            )
            
            categorical_analysis[col] = {
                "top_values": values,
                "has_variations": has_variations,
                "unique_count": df.select(col).distinct().count()
            }
    
    analysis_results["categorical_analysis"] = categorical_analysis
    needs_cat_standardization = any(stats["has_variations"] for stats in categorical_analysis.values())
    cfg_live["standardize_cats"] = needs_cat_standardization
    print(f"   - Needs categorical standardization: {needs_cat_standardization} → standardize_cats: {needs_cat_standardization}")
    
    # 5. Generate state and department mappings dynamically
    print("📊 Generating categorical mappings...")
    
    # State mappings
    if "state" in df.columns:
        state_values = df.filter(F.col("state").isNotNull()) \
                        .select("state") \
                        .distinct() \
                        .collect()
        
        state_map = {}
        for row in state_values:
            if row["state"]:
                state_val = str(row["state"]).strip().upper()
                # Create mapping for common variations
                if len(state_val) == 2:
                    state_map[state_val] = state_val
                elif state_val in ["CALIFORNIA", "CAL"]:
                    state_map[state_val] = "CA"
                elif state_val in ["NEW YORK", "NYC"]:
                    state_map[state_val] = "NY"
                elif state_val in ["TEXAS", "TXS"]:
                    state_map[state_val] = "TX"
                else:
                    state_map[state_val] = state_val[:2].upper() if len(state_val) >= 2 else state_val
        
        cfg_live["state_map"] = state_map
    
    # Department mappings
    if "department" in df.columns:
        dept_values = df.filter(F.col("department").isNotNull()) \
                       .select("department") \
                       .distinct() \
                       .collect()
        
        dept_map = {}
        for row in dept_values:
            if row["department"]:
                dept_val = str(row["department"]).strip().upper()
                # Standardize common department names
                if "HR" in dept_val or "HUMAN" in dept_val:
                    dept_map[dept_val] = "HUMAN RESOURCES"
                elif "IT" in dept_val or "TECH" in dept_val:
                    dept_map[dept_val] = "TECHNOLOGY"
                elif "FIN" in dept_val or "ACCOUNT" in dept_val:
                    dept_map[dept_val] = "FINANCE"
                elif "SALE" in dept_val:
                    dept_map[dept_val] = "SALES"
                elif "MARKET" in dept_val or "MKT" in dept_val:
                    dept_map[dept_val] = "MARKETING"
                else:
                    dept_map[dept_val] = dept_val
        
        cfg_live["dept_map"] = dept_map
    
    # 6. Store analysis results
    cfg_live["analysis_results"] = analysis_results
    
    # Display generated configuration
    print("\n🎯 GENERATED CONFIGURATION FLAGS:")
    print("=" * 50)
    display_flags(cfg_live)
    
    # Display analysis summary
    print("\n📋 DATA ANALYSIS SUMMARY:")
    print("=" * 50)
    print(f"Total Records: {df.count()}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Nulls Found: {has_nulls}")
    print(f"Duplicates Found: {duplicate_count}")
    print(f"Numeric Cleaning Needed: {needs_numeric_cleaning}")
    print(f"Categorical Standardization Needed: {needs_cat_standardization}")
    
    return cfg_live
# =========================
# PHASE 02 - MAIN PIPELINE EXECUTION
# =========================

def execute_main_pipeline(df, cfg_live):
    """
    Phase 02: Execute main pipeline with generated configuration
    """
    print("\n🚀 PHASE 02: MAIN PIPELINE EXECUTION")
    print("=" * 60)
    
    # Use the run_pipeline function
    df_processed = run_pipeline(df, cfg_live)
    
    return df_processed

# =========================
# COMPLETE WORKFLOW EXECUTION
# =========================

def execute_complete_workflow(df):
    """
    Complete two-phase workflow execution
    """
    print("🎬 STARTING COMPLETE ETL WORKFLOW")
    print("=" * 60)
    
    # Phase 01: Generate configuration flags
    cfg_live = analyze_data_and_generate_flags(df)
    
    # Save cfg_live as global variable for inspection
    globals()['cfg_live'] = cfg_live
    
    print("\n💾 CONFIGURATION SAVED AS: cfg_live")
    print("   Available for inspection and modification before Phase 02")
    
    # Optional: Add a pause point for manual configuration review
    print("\n⏸️  PHASE 01 COMPLETE!")
    print("   Review the generated configuration above.")
    print("   Modify cfg_live if needed, then proceed to Phase 02.")
    print("   To proceed, call: execute_main_pipeline(df, cfg_live)")
    
    return cfg_live

# Phase 01: Generate configuration
# Generate configuration flags based on data analysis
cfg_live = execute_complete_workflow(df)
# Review and modify configuration if needed
#cfg_live["remove_outliers"] = True  # Enable outlier removal

# Phase 02: Execute with modified configuration  
df_silver = execute_main_pipeline(df, cfg_live)

# =========================
# STEP 4: WRITE TO SILVER LAYER - IMPROVED
# =========================
print("🚀 STARTING SILVER LAYER WRITE OPERATIONS...")
print("=" * 50)

try:
    print("📤 Writing Spark DataFrame to Silver...")
    
    df_silver.write \
        .mode("overwrite") \
        .option("compression", "snappy") \
        .partitionBy("department") \
        .parquet(silver_spark_path)
    
    # Verify Spark write
    df_verify_spark = spark.read.parquet(silver_spark_path)
    spark_record_count = df_verify_spark.count()
    
    print(f"✅ Spark DF SUCCESS → {silver_spark_path}")
    print(f"   📊 Records written: {spark_record_count}")
    print(f"   🗂️  Partitions: department")
    
except Exception as spark_error:
    print(f"❌ Spark DF Write Failed: {spark_error}")
    # Continue with Glue write even if Spark fails
 
job.commit()