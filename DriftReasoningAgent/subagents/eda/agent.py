import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from google.adk.agents import Agent

# --- Constants ---
GEMINI_MODEL = "gemini-2.0-flash"


def analyze_drift_data(drift_column: str) -> str:
    """
    Analyze the old and new data to find relationships between the drifted column
    and other columns that might explain the drift.

    Args:
        drift_column: The name of the column where drift was detected

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
        new_data = pd.read_csv(new_data_path)
        old_data = pd.read_csv(old_data_path)

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
        output += (
            f"Old Data - Unique values: {old_data[drift_column].nunique()}\n"
        )
        output += (
            f"New Data - Unique values: {new_data[drift_column].nunique()}\n\n"
        )

        # Analyze relationships with other columns
        output += "=== Column Relationships ===\n"
        other_columns = [
            col for col in new_data.columns if col != drift_column
        ]
        potential_causes = []

        for col in other_columns:
            # Temporal analysis
            if any(
                keyword in col.lower()
                for keyword in ["timestamp", "date", "time"]
            ):
                try:
                    new_data_temp = new_data.copy()
                    new_data_temp[col] = pd.to_datetime(new_data_temp[col])
                    grouped = new_data_temp.groupby(col)[drift_column].agg(
                        ["mean", "count"]
                    )
                    output += f"\n{col} (Temporal):\n"
                    output += f"  Time-based patterns detected with {len(grouped)} unique time points\n"
                    potential_causes.append(
                        f"{col} shows temporal patterns that may influence {drift_column}"
                    )
                except:
                    pass

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


# Create the EDA agent
eda_agent = Agent(
    name="eda_agent",
    model=GEMINI_MODEL,
    instruction="""You are an Exploratory Data Analysis (EDA) Agent.

    Your task is to analyze the drift detected in a column by exploring relationships
    with other columns in the dataset.

    you can find the column on which the drift was detected by accessing the drift_info from the state.
    you can also use the meta_info from the state to help you with your analysis to understand what each column represents.
    
    You will:
    1. Access drift_info from the state to find which column has drift
    2. Call analyze_drift_data() tool with the drifted column name
    3. Analyze relationships between the drifted column and other columns:
       - Temporal patterns (timestamp, date columns)
       - Numeric correlations
       - Categorical groupings (especially promotional, discount, event columns)
    4. Identify potential causes of drift based on:
       - Strong correlations with other numeric features
       - Changes in categorical features (e.g., promotions affecting sales)
       - Temporal trends
    
    Provide a clear summary of:
    - Statistical changes in the drifted column
    - Columns that show strong relationships with the drifted column and can be reasonably explain the drift

    
    This information will help understand WHY the drift occurred, not just WHERE it occurred and if there were any other columns that can play a role in the drift occurence.
    """,
    description="Performs exploratory data analysis to find relationships and potential causes of detected drift.",
    output_key="eda_info",
    tools=[analyze_drift_data],
)
