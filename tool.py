import pandas as pd
from typing import Dict, Any
from itertools import combinations

def csvs_to_dataframes(uploaded_files):
    """
    Reads multiple uploaded CSV files from Streamlit and returns a dictionary of pandas DataFrames.

    Args:
        uploaded_files (list of UploadedFile): List of Streamlit uploaded file objects.

    Returns:
        dict: Dictionary where keys are file names and values are DataFrames containing the CSV data.
    """
    dataframes = {}
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name.rsplit('.', 1)[0]  # Get file name without extension
        try:
            df = pd.read_csv(uploaded_file)
            if df.empty or df.shape[1] == 0:
                raise ValueError("CSV file is empty or has no columns.")
            dataframes[file_name] = df
        except Exception as e:
            print(f"Error reading {uploaded_file.name}: {e}")
            dataframes[file_name] = None
    return dataframes

def analyze_primary_key_candidates(
    dataframes: Dict[str, pd.DataFrame]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Analyzes each DataFrame to check which columns could be primary key candidates.

    Args:
        dataframes (dict): Dictionary of DataFrames.

    Returns:
        dict: Dictionary with DataFrame names as keys and column analysis as values.
    """
    analysis: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for df_name, df in dataframes.items():
        columns_info: Dict[str, Dict[str, Any]] = {}
        for col in df.columns:
            col_data = df[col]
            info = {
                "has_nulls": col_data.isnull().any(),
                "is_unique": col_data.is_unique,
                "is_constant": col_data.nunique() == 1,
                "has_duplicates": not col_data.is_unique,
                "dtype": str(col_data.dtype),
                "can_be_primary_key": (
                    not col_data.isnull().any() and col_data.is_unique
                )
            }
            columns_info[col] = info
        analysis[df_name] = columns_info
    return analysis

def find_composite_keys(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    For each DataFrame, finds a composite key (set of columns) that uniquely identifies rows
    if no single-column primary key exists.

    Args:
        dataframes (dict): Dictionary of DataFrames.

    Returns:
        dict: Dictionary with DataFrame names as keys and composite key columns (list) as values.
    """
    composite_keys = {}
    for df_name, df in dataframes.items():
        # Skip if DataFrame is None or empty
        if df is None or df.empty:
            composite_keys[df_name] = None
            continue

        # Find columns that could be single-column primary keys
        single_keys = [
            col for col in df.columns
            if not df[col].isnull().any() and df[col].is_unique
        ]
        if single_keys:
            composite_keys[df_name] = single_keys[0]  # Use the first single-column key
            continue

        # Try combinations of columns for composite key
        cols = [col for col in df.columns if not df[col].isnull().any()]
        found = False
        for r in range(2, len(cols) + 1):
            for combo in combinations(cols, r):
                if df[list(combo)].drop_duplicates().shape[0] == df.shape[0]:
                    composite_keys[df_name] = list(combo)
                    found = True
                    break
            if found:
                break
        if not found:
            composite_keys[df_name] = None  # No composite key found
    return composite_keys


def find_dataframe_relations(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Finds possible relations between DataFrames based on shared columns and value overlaps.
    Useful for suggesting join keys.

    Args:
        dataframes (dict): Dictionary of DataFrames.

    Returns:
        dict: Dictionary with tuple of DataFrame names as keys and relation info as values.
    """
    relations = {}
    df_names = list(dataframes.keys())
    for i in range(len(df_names)):
        for j in range(i + 1, len(df_names)):
            df1_name, df2_name = df_names[i], df_names[j]
            df1, df2 = dataframes[df1_name], dataframes[df2_name]
            if df1 is None or df2 is None:
                continue
            shared_cols = set(df1.columns).intersection(set(df2.columns))
            relation_info = []
            for col in shared_cols:
                # Check if there is a significant overlap in values
                overlap = set(df1[col].dropna()).intersection(set(df2[col].dropna()))
                overlap_ratio = (
                    len(overlap) / min(df1[col].nunique(), df2[col].nunique())
                    if min(df1[col].nunique(), df2[col].nunique()) > 0 else 0
                )
                if overlap_ratio > 0.5:  # Arbitrary threshold for meaningful overlap
                    relation_info.append({
                        "column": col,
                        "overlap_ratio": overlap_ratio,
                        "df1_unique": df1[col].is_unique,
                        "df2_unique": df2[col].is_unique
                    })
            if relation_info:
                relations[(df1_name, df2_name)] = relation_info
    return relations

def create_erd_diagram(
    dataframes: Dict[str, pd.DataFrame],
    primary_keys: Dict[str, Any],
    relations: Dict[str, Any]
) -> str:
    """
    Generates a simple ERD diagram in Graphviz DOT format based on DataFrames, primary keys, and relations.

    Args:
        dataframes (dict): Dictionary of DataFrames.
        primary_keys (dict): Dictionary of primary key columns for each DataFrame.
        relations (dict): Dictionary of relations between DataFrames.

    Returns:
        str: Graphviz DOT string representing the ERD diagram.
    """
    dot = ["digraph ERD {", "  rankdir=LR;"]
    # Add table nodes
    for df_name, df in dataframes.items():
        if df is None:
            continue
        pk = primary_keys.get(df_name)
        columns = []
        for col in df.columns:
            if pk == col or (isinstance(pk, list) and col in pk):
                columns.append(f"<b>{col}</b>")
            else:
                columns.append(col)
        label_sep = '\\l'
        label = f"{df_name}|{label_sep.join(columns)}{label_sep}"
        dot.append(f'  "{df_name}" [shape=record, label="{{{label}}}"];')
    # Add relations (edges)
    for relation_key, rel_info in relations.items():
        # Handle both string keys ("df1-df2") and tuple keys (df1, df2)
        if isinstance(relation_key, str) and '-' in relation_key:
            df1, df2 = relation_key.split('-', 1)
        elif isinstance(relation_key, tuple) and len(relation_key) == 2:
            df1, df2 = relation_key
        else:
            continue
            
        for rel in rel_info:
            col = rel["column"]
            dot.append(f'  "{df1}" -> "{df2}" [label="{col}"];')
    dot.append("}")
    return "\n".join(dot)

