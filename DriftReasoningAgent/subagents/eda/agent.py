import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from google.adk.agents import Agent

# --- Constants ---
GEMINI_MODEL = "gemini-2.0-flash"


def analyze_drift_data(drift_column: str, focus_columns: str = "") -> str:
    """
    Analyze the old and new data to find relationships between the drifted column
    and other columns that might explain the drift.

    Args:
        drift_column: The name of the column where drift was detected
        focus_columns: Comma-separated list of specific columns to focus on (optional)

    Returns:
        str: Analysis results with column relationships and potential causes
    """
    # Get the project root directory
    current_dir = Path(__file__).parent  # .../eda
    project_root = current_dir.parent.parent.parent  # Go up 3 levels to root
    new_data_path = project_root / "data" / "new_data.csv"
    old_data_path = project_root / "data" / "old_data.csv"

    try:
        if not new_data_path.exists() or not old_data_path.exists():
            return f"Data files not found.\nNew data path: {new_data_path}\nOld data path: {old_data_path}"

        # Load the datasets
        try:
            new_data = pd.read_csv(new_data_path)
            old_data = pd.read_csv(old_data_path)
        except Exception as csv_error:
            return f"Error loading CSV files: {str(csv_error)}\nNew data path: {new_data_path}\nOld data path: {old_data_path}"

        if (
            drift_column not in new_data.columns
            or drift_column not in old_data.columns
        ):
            return f"Column '{drift_column}' not found in the datasets."

        output = f"EDA Analysis for drifted column: {drift_column}\n\n"

        # Statistical summary
        output += "=== Statistical Summary ===\n"
        if pd.api.types.is_numeric_dtype(new_data[drift_column]):
            output += f"Old Data - Mean: {old_data[drift_column].mean():.4f}, Std: {old_data[drift_column].std():.4f}\n"
            output += f"New Data - Mean: {new_data[drift_column].mean():.4f}, Std: {new_data[drift_column].std():.4f}\n"
            output += f"Change: {((new_data[drift_column].mean() - old_data[drift_column].mean()) / old_data[drift_column].mean() * 100):.2f}%\n"
        output += (
            f"Old Data - Unique values: {old_data[drift_column].nunique()}\n"
        )
        output += (
            f"New Data - Unique values: {new_data[drift_column].nunique()}\n\n"
        )
        
        # FARE COMPONENT ANALYSIS - if drift_column is total_amount
        if drift_column.lower() == 'total_amount':
            output += "=== Fare Component Analysis ===\n"
            try:
                fare_components = ['fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge']
                available_components = [col for col in fare_components if col in new_data.columns]
                
                if available_components:
                    for comp in available_components:
                        old_mean = old_data[comp].mean()
                        new_mean = new_data[comp].mean()
                        old_pct = (old_data[comp].sum() / old_data[drift_column].sum() * 100) if old_data[drift_column].sum() > 0 else 0
                        new_pct = (new_data[comp].sum() / new_data[drift_column].sum() * 100) if new_data[drift_column].sum() > 0 else 0
                        change_pct = new_pct - old_pct
                        change_mean = ((new_mean - old_mean) / old_mean * 100) if old_mean > 0 else 0
                        output += f"  {comp}: Mean Old={old_mean:.2f}, Mean New={new_mean:.2f}, Mean Change={change_mean:+.2f}%\n"
                        output += f"    Contribution: Old={old_pct:.2f}%, New={new_pct:.2f}%, Change={change_pct:+.2f}%\n"
                    output += "\n"
            except Exception as e:
                output += f"  Error in fare component analysis: {str(e)[:100]}\n\n"

        # Analyze relationships with other columns
        output += "=== Column Relationships Analysis ===\n"
        
        # Filter columns based on focus_columns if provided
        if focus_columns:
            focus_list = [col.strip() for col in focus_columns.split(",")]
            other_columns = [col for col in focus_list if col in new_data.columns and col != drift_column]
            output += f"(Focusing on: {', '.join(other_columns)})\n\n"
        else:
            other_columns = [col for col in new_data.columns if col != drift_column]
        
        potential_causes = []
        
        # VENDORID ANALYSIS - if specifically requested
        if 'VendorID' in other_columns or 'vendorid' in [c.lower() for c in other_columns]:
            output += "=== VendorID Analysis ===\n"
            try:
                vendor_col = 'VendorID' if 'VendorID' in other_columns else [c for c in other_columns if c.lower() == 'vendorid'][0]
                old_vendor = old_data.groupby(vendor_col)[drift_column].agg(['mean', 'count'])
                new_vendor = new_data.groupby(vendor_col)[drift_column].agg(['mean', 'count'])
                
                output += f"  Old Data by Vendor:\n{old_vendor.to_string()}\n"
                output += f"  New Data by Vendor:\n{new_vendor.to_string()}\n\n"
                potential_causes.append(f"{vendor_col} shows different pricing patterns between vendors")
            except Exception as e:
                output += f"  Error in VendorID analysis: {str(e)[:100]}\n\n"

        for col in other_columns:
            # Skip temporal columns in focus mode to avoid performance issues
            if focus_columns and any(
                keyword in col.lower()
                for keyword in ["timestamp", "date", "time", "pickup", "dropoff"]
            ):
                output += f"\n{col} (Temporal): Skipped in focused analysis mode\n"
                continue
                
            # Temporal analysis
            if any(
                keyword in col.lower()
                for keyword in ["timestamp", "date", "time", "pickup", "dropoff"]
            ):
                try:
                    output += f"\n{col} (TEMPORAL ANALYSIS - Limited to 1000 samples):\n"
                    
                    # Sample data for performance
                    sample_size = min(1000, len(new_data), len(old_data))
                    new_sample = new_data.sample(n=sample_size, random_state=42) if len(new_data) > sample_size else new_data
                    old_sample = old_data.sample(n=sample_size, random_state=42) if len(old_data) > sample_size else old_data
                    
                    new_sample[col] = pd.to_datetime(new_sample[col], errors='coerce')
                    old_sample[col] = pd.to_datetime(old_sample[col], errors='coerce')
                    
                    # Extract hour only
                    new_sample['hour'] = new_sample[col].dt.hour
                    old_sample['hour'] = old_sample[col].dt.hour
                    
                    # Hour of day analysis
                    grouped_new_hour = new_sample.groupby('hour')[drift_column].mean()
                    grouped_old_hour = old_sample.groupby('hour')[drift_column].mean()
                    
                    output += f"  Hour of Day Pattern (sampled):\n"
                    output += f"    Old data mean {drift_column} by hour: {grouped_old_hour.to_dict()}\n"
                    output += f"    New data mean {drift_column} by hour: {grouped_new_hour.to_dict()}\n"
                    
                    potential_causes.append(
                        f"{col} shows temporal patterns (hourly) that may influence {drift_column}"
                    )
                except Exception as e:
                    output += f"\n{col} (Temporal): Error analyzing - {str(e)[:100]}\n"

            # Numeric correlation
            elif pd.api.types.is_numeric_dtype(
                new_data[col]
            ) and pd.api.types.is_numeric_dtype(new_data[drift_column]):
                correlation = new_data[col].corr(new_data[drift_column])
                if abs(correlation) > 0.3:
                    strength = (
                        "strong" if abs(correlation) > 0.7 else "moderate"
                    )
                    output += f"\n{col} (Numeric):\n"
                    output += (
                        f"  Correlation: {correlation:.4f} ({strength})\n"
                    )
                    potential_causes.append(
                        f"{col} shows {strength} correlation ({correlation:.2f}) with {drift_column}"
                    )

            # Categorical analysis
            elif pd.api.types.is_object_dtype(
                new_data[col]
            ) or pd.api.types.is_categorical_dtype(new_data[col]):
                grouped_old = old_data.groupby(col)[drift_column].mean()
                grouped_new = new_data.groupby(col)[drift_column].mean()
                unique_vals = new_data[col].nunique()

                output += f"\n{col} (Categorical):\n"
                output += f"  Unique values: {unique_vals}\n"
                output += (
                    f"  Old data grouped means: {grouped_old.to_dict()}\n"
                )
                output += (
                    f"  New data grouped means: {grouped_new.to_dict()}\n"
                )

                # Check for promotional or special columns
                if any(
                    keyword in col.lower()
                    for keyword in [
                        "promo",
                        "discount",
                        "sale",
                        "event",
                        "campaign",
                    ]
                ):
                    potential_causes.append(
                        f"{col} (promotional/event feature) may influence {drift_column} - has {unique_vals} categories"
                    )

        # Summary of potential causes
        output += "\n\n=== Potential Causes of Drift ===\n"
        if potential_causes:
            for i, cause in enumerate(potential_causes, 1):
                output += f"{i}. {cause}\n"
        else:
            output += "No strong relationships found with other columns.\n"

        return output

    except Exception as e:
        return f"Error in EDA analysis: {str(e)}"


def create_eda_agent(iteration: int = 1) -> Agent:
    """Create an EDA agent instance for a specific iteration."""
    return Agent(
        name=f"eda_agent_iter{iteration}",
        model=GEMINI_MODEL,
        instruction=f"""You are an Exploratory Data Analysis (EDA) Agent specialized in NYC Taxi dataset drift analysis.

    DATASET CONTEXT - NYC Taxi Data:
    - TEMPORAL: tpep_pickup_datetime, tpep_dropoff_datetime (time-based patterns)
    - SPATIAL: pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude (location patterns)
    - TRIP: trip_distance, passenger_count (trip characteristics)
    - FINANCIAL: fare_amount, tip_amount, tolls_amount, extra, mta_tax, improvement_surcharge, total_amount
    - OPERATIONAL: VendorID, RateCodeID, store_and_fwd_flag, payment_type

    WORKFLOW (Iterative with Reflection):
    1. Access drift_info from state to identify the drifted column
    2. **CRITICAL**: Check if reflection_feedback exists in state:
       - Look for reflection_feedback key in state
       - If present: READ IT CAREFULLY and extract specific column names mentioned
       - Extract recommended columns and analytical approaches from the feedback
       - Use focus_columns parameter with the recommended columns
       - If no reflection_feedback: perform comprehensive initial analysis
    3. Call analyze_drift_data() with:
       - The drifted column name
       - focus_columns: COMMA-SEPARATED list of specific columns from reflection (e.g., "Improvement_surcharge,Fare_amount,Tip_amount")
    4. For NYC Taxi data, prioritize analyzing:
       - TIME PATTERNS: Hour/day effects on fares, demand cycles
       - SPATIAL PATTERNS: Location-based pricing (Manhattan vs outer boroughs)
       - FARE COMPONENTS: Relationship between base fare, tips, tolls, extras
       - TRIP CHARACTERISTICS: Distance vs fare, passenger count effects
       - VENDOR/RATE DIFFERENCES: Variations by VendorID or RateCodeID
       - PAYMENT BEHAVIOR: Payment type affecting tip patterns
    
    5. Identify potential causes considering:
       - Strong correlations (>0.5) with numeric features
       - Temporal trends (rush hours, weekday/weekend, seasons)
       - Spatial patterns (high-demand zones, airport trips)
       - Behavioral changes (tipping patterns, payment preferences)
       - Operational changes (rate code shifts, vendor distribution)

    OUTPUT FORMAT:
    - Statistical changes in the drifted column (means, distributions)
    - Top 3-5 most influential relationships with supporting evidence
    - Domain-specific insights for taxi operations
    - Clear explanation of WHY drift occurred (not just WHERE)

    ITERATION AWARENESS (Current Iteration: {iteration}):
    - **Iteration 1**: Cast a wide net - NO focus_columns, analyze ALL relationships comprehensively
    - **Iteration 2+**: READ reflection_feedback from state - extract specific column names mentioned, use focus_columns with those exact columns
    - Always state at the beginning: "Starting EDA Iteration {iteration}" and whether reflection feedback was found
    - If reflection feedback exists, quote the key recommendations before calling the tool
    - Deepen analysis on promising relationships suggested by reflection agent
    
    RESPONSE FORMAT:
    Start your response with:
    "=== EDA ITERATION {iteration} ==="
    Then indicate if you found reflection_feedback and what it suggested.
""",
        description=f"EDA agent for iteration {iteration} - performs exploratory analysis with reflection-based refinement",
        output_key="eda_info" if iteration == 3 else f"eda_info_iter{iteration}",
        tools=[analyze_drift_data],
    )


# Create the default EDA agent (backward compatibility)
eda_agent = create_eda_agent(iteration=1)
