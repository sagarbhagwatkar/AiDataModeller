import pandas as pd
import json
import os
import re
from typing import Dict, Any, List, Optional, Iterable
from itertools import combinations
from pathlib import Path

# All abbreviation / vector store (RAG) functionality removed per request.




def csvs_to_dataframes(uploaded_files) -> Dict[str, pd.DataFrame]:
    """Load multiple uploaded CSV files into DataFrames.

    Args:
        uploaded_files: Iterable of file-like objects with a ``name`` attribute.

    Returns:
        Dictionary mapping base filename (without extension) to DataFrame.
        Files that fail to load map to ``None``.
    """
    dataframes: Dict[str, pd.DataFrame] = {}
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name.rsplit('.', 1)[0]
        try:
            df = pd.read_csv(uploaded_file)
            if df.empty or df.shape[1] == 0:
                raise ValueError("CSV file is empty or has no columns.")
            dataframes[file_name] = df
        except Exception as e:  # noqa: BLE001
            print(f"Error reading {uploaded_file.name}: {e}")
            dataframes[file_name] = None
    return dataframes


def csvs_jsons_to_dataframes(uploaded_files) -> Dict[str, pd.DataFrame]:
    """Load multiple uploaded CSV, JSON (nested), or Excel (xlsx/xls) files into DataFrames.

    Enhanced JSON handling (non-breaking):
      * Previous simple behavior retained for simple list[dict] or dict with single list.
      * NEW: Recursively extract every list of objects (list[dict]) found anywhere in the JSON tree.
      * Each extracted table gets a generated name: ``<base>__<path_segments_joined>``.
      * Parent/child relationships preserved via surrogate integer primary keys:
           - Each table gets ``__row_id`` (surrogate PK)
           - Child tables get ``__parent_id`` referencing the parent's ``__row_id``
      * Nested objects (dicts) are flattened into parent row columns with dot-path names.
      * Lists of primitives are ignored (can be added later if needed).

    Args:
        uploaded_files: Iterable of file-like objects with a ``name`` attribute.

    Returns:
        Dictionary mapping table name to DataFrame. For simple JSON there will still
        be a single table named after the file (backward compatible). For nested JSON
        multiple tables may be returned.
    """

    def _extract_tables(obj: Any, base_name: str, path: List[str], parent_id: int | None,
                        tables: Dict[str, List[Dict[str, Any]]], parent_table: str | None,
                        table_meta: Dict[str, Dict[str, Any]]):
        """Recursive walk collecting list[dict] structures as separate tables.

        Args:
            obj: Current JSON node.
            base_name: Root file base name.
            path: Accumulated path segments.
            parent_id: Surrogate id of parent row if inside a list element.
            tables: Accumulator mapping table_name -> list of row dicts.
            parent_table: Name of parent table.
            table_meta: Metadata store for relationships.
        """
        if isinstance(obj, list):
            if all(isinstance(el, dict) for el in obj) and obj:
                # This list becomes/augments a table
                table_name = base_name if not path else f"{base_name}__{'__'.join(path)}"
                if table_name not in tables:
                    tables[table_name] = []
                    table_meta[table_name] = {"parent_table": parent_table}
                for idx, row in enumerate(obj):
                    if not isinstance(row, dict):  # safety
                        continue
                    flat_row: Dict[str, Any] = {}
                    # Flatten nested object fields first (will recurse for deeper lists)
                    for k, v in row.items():
                        if isinstance(v, (dict, list)):
                            # Defer to recursion for lists, flatten dicts later
                            continue
                        flat_row[k] = v
                    flat_row['__row_id'] = len(tables[table_name]) + 1
                    if parent_id is not None:
                        flat_row['__parent_id'] = parent_id
                    tables[table_name].append(flat_row)
                    # Recurse into nested dicts to append flattened keys to the same row
                    for k, v in row.items():
                        if isinstance(v, dict):
                            for dk, dv in v.items():
                                col_name = f"{k}.{dk}"
                                tables[table_name][-1][col_name] = dv
                        elif isinstance(v, list):
                            # Child list -> separate child table
                            _extract_tables(v, base_name, path + [k], tables[table_name][-1]['__row_id'],
                                            tables, table_name, table_meta)
            else:
                # List of primitives or empty: ignore for now
                return
        elif isinstance(obj, dict):
            # Explore dict values
            for k, v in obj.items():
                _extract_tables(v, base_name, path + [k], parent_id, tables, parent_table, table_meta)
        else:
            return

    dataframes: Dict[str, pd.DataFrame] = {}
    for uploaded_file in uploaded_files:
        base_name = uploaded_file.name.rsplit('.', 1)[0]
        ext = uploaded_file.name.rsplit('.', 1)[-1].lower()
        try:
            if ext == 'csv':
                df_csv = pd.read_csv(uploaded_file)
                if df_csv.empty or df_csv.shape[1] == 0:
                    raise ValueError('File is empty or has no columns.')
                dataframes[base_name] = df_csv
                continue
            if ext in {'xlsx', 'xls'}:
                # Load each worksheet as separate table: base__sheetname
                xls = pd.ExcelFile(uploaded_file)
                for sheet in xls.sheet_names:
                    df_sheet = xls.parse(sheet)
                    if df_sheet.empty or df_sheet.shape[1] == 0:
                        continue
                    tname = base_name if sheet.lower() in {"sheet1", "data", base_name.lower()} else f"{base_name}__{sheet}"
                    dataframes[tname] = df_sheet
                continue
            if ext != 'json':
                raise ValueError('Unsupported file type. Only CSV, JSON, and Excel are allowed.')

            raw_json = json.load(uploaded_file)

            # Use recursive extraction
            tables: Dict[str, List[Dict[str, Any]]] = {}
            table_meta: Dict[str, Dict[str, Any]] = {}
            _extract_tables(raw_json, base_name, [], None, tables, None, table_meta)

            if not tables:
                # Fallback to simple normalization
                if isinstance(raw_json, list) and raw_json and isinstance(raw_json[0], dict):
                    tables[base_name] = [row for row in raw_json if isinstance(row, dict)]
                elif isinstance(raw_json, dict):
                    tables[base_name] = [raw_json]
                else:
                    raise ValueError('Unsupported JSON structure for normalization.')

            # Convert collected rows to DataFrames
            for tname, rows in tables.items():
                df = pd.json_normalize(rows)
                if df.empty or df.shape[1] == 0:
                    continue
                dataframes[tname] = df
        except Exception as e:  # noqa: BLE001
            print(f"Error reading {uploaded_file.name}: {e}")
            dataframes[base_name] = None
    return dataframes


def analyze_primary_key_candidates(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Analyse each DataFrame for primary-key suitability per column.

    For each column we record:
      has_nulls, is_unique, is_constant, has_duplicates, dtype, can_be_primary_key.

    Args:
        dataframes: Mapping of name -> DataFrame (or None).

    Returns:
        Nested dictionary: df_name -> column -> metrics.
    """
    analysis: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for df_name, df in dataframes.items():
        if df is None:
            analysis[df_name] = {}
            continue
        columns_info: Dict[str, Dict[str, Any]] = {}
        for col in df.columns:
            col_data = df[col]
            # Handle unhashable elements (e.g., lists, dicts) by converting to string
            # for uniqueness/null analysis to avoid 'unhashable type: list' errors.
            if col_data.dtype == 'object':
                sample_non_null = col_data.dropna().head(1)
                needs_string = False
                if not sample_non_null.empty:
                    first_val = sample_non_null.iloc[0]
                    if isinstance(first_val, (list, dict, set)):
                        needs_string = True
                if needs_string:
                    safe_series = col_data.apply(lambda v: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict, set)) else v)
                else:
                    safe_series = col_data
            else:
                safe_series = col_data
            columns_info[col] = {
                "has_nulls": safe_series.isnull().any(),
                "is_unique": safe_series.is_unique,
                "is_constant": safe_series.nunique() == 1,
                "has_duplicates": (not safe_series.is_unique),
                "dtype": str(col_data.dtype),
                    "can_be_primary_key": (not safe_series.isnull().any() and safe_series.is_unique),
            }
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

            # 1. Detect conventional shared-column relationships
            shared_cols = set(df1.columns).intersection(set(df2.columns))
            relation_info = []
            for col in shared_cols:
                overlap = set(df1[col].dropna()).intersection(set(df2[col].dropna()))
                overlap_ratio = (
                    len(overlap) / min(df1[col].nunique(), df2[col].nunique())
                    if min(df1[col].nunique(), df2[col].nunique()) > 0 else 0
                )
                if overlap_ratio > 0.5:
                    relation_info.append({
                        "column": col,
                        "child_column": col,
                        "parent_column": col,
                        "overlap_ratio": overlap_ratio,
                        "df1_unique": df1[col].is_unique,
                        "df2_unique": df2[col].is_unique,
                        "cardinality": "one-to-many" if df1[col].is_unique and not df2[col].is_unique else "unknown",
                    })

            # 2. Detect JSON-derived parent/child pattern (__row_id -> __parent_id)
            # Pattern A: df1 is parent, df2 is child
            if "__row_id" in df1.columns and "__parent_id" in df2.columns:
                overlap = set(df1['__row_id'].dropna()).intersection(set(df2['__parent_id'].dropna()))
                if overlap:
                    overlap_ratio = len(overlap) / max(1, df2['__parent_id'].nunique())
                    relation_info.append({
                        "column": "__parent_id",  # legacy field for backward usage
                        "child_column": "__parent_id",
                        "parent_column": "__row_id",
                        "overlap_ratio": overlap_ratio,
                        "df1_unique": df1['__row_id'].is_unique,
                        "df2_unique": df2['__parent_id'].is_unique,
                        "cardinality": "one-to-many",
                    })
            # Pattern B: df2 is parent, df1 is child
            if "__row_id" in df2.columns and "__parent_id" in df1.columns:
                overlap = set(df2['__row_id'].dropna()).intersection(set(df1['__parent_id'].dropna()))
                if overlap:
                    overlap_ratio = len(overlap) / max(1, df1['__parent_id'].nunique())
                    # store relation under (parent, child) ordering
                    relations.setdefault((df2_name, df1_name), []).append({
                        "column": "__parent_id",
                        "child_column": "__parent_id",
                        "parent_column": "__row_id",
                        "overlap_ratio": overlap_ratio,
                        "df1_unique": df2['__row_id'].is_unique,
                        "df2_unique": df1['__parent_id'].is_unique,
                        "cardinality": "one-to-many",
                    })

            if relation_info:
                relations.setdefault((df1_name, df2_name), []).extend(relation_info)
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


# (map_columns_to_abbreviations function removed)


def abbreviate_columns(columns: Iterable[str], custom_rules: Optional[Dict[str, str]] = None,
                       max_token_length: int = 12) -> Dict[str, str]:
    """Create shortened/abbreviated variants of column names.

    Strategy:
      1. Split each name into tokens on underscores, spaces, camelCase boundaries.
      2. Apply rule-based replacements (e.g. name->nm, number->num, description->desc).
      3. For long tokens not covered by a rule, drop interior vowels after the first letter
         until token length <= 4 (simple heuristic) while preserving leading character sequence.
      4. Rejoin with underscores.
      5. Ensure uniqueness; if collision occurs, append numeric suffix (_2, _3, ...).

    Args:
        columns: Original column names.
        custom_rules: Optional dict overriding / extending built-in token rules.
        max_token_length: Hard upper bound for any individual token (after abbreviation).

    Returns:
        Dict mapping original column name -> abbreviated column name.
    """
    # Base rules (token -> abbreviation)
    rules: Dict[str, str] = {
        'identifier': 'id', 'identity': 'id', 'id': 'id',
        'number': 'num', 'nbr': 'num', 'count': 'cnt',
        'name': 'nm', 'first': 'first', 'last': 'last',
        'description': 'desc', 'status': 'sts', 'quantity': 'qty', 'amount': 'amt',
        'date': 'dt', 'datetime': 'dt', 'timestamp': 'ts',
        'price': 'prc', 'total': 'tot', 'average': 'avg', 'minimum': 'min', 'maximum': 'max',
        'customer': 'cust', 'employee': 'emp', 'department': 'dept', 'product': 'prod',
        'address': 'addr', 'street': 'st', 'state': 'st', 'country': 'ctry', 'city': 'city',
        'phone': 'ph', 'email': 'em', 'postal': 'pst', 'code': 'cd',
        'category': 'cat', 'type': 'typ', 'version': 'ver', 'reference': 'ref',
        'parent': 'prnt', 'child': 'chld'
    }
    if custom_rules:
        # Custom rules override defaults
        for k, v in custom_rules.items():
            rules[k.lower()] = v

    def split_tokens(name: str) -> List[str]:
        # Replace non-alphanum with underscore
        cleaned = re.sub(r'[^0-9A-Za-z]+', '_', name)
        # Insert underscore before camelCase transitions
        cleaned = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', cleaned)
        tokens = [t for t in cleaned.lower().split('_') if t]
        return tokens or ['col']

    def abbreviate_token(tok: str) -> str:
        if tok in rules:
            return rules[tok]
        # Already short enough
        if len(tok) <= 4:
            return tok
        # Remove interior vowels after first letter until length <=4
        head = tok[0]
        tail = tok[1:]
        tail_no_vowels = re.sub(r'[aeiou]', '', tail)
        candidate = (head + tail_no_vowels)[:4]
        if len(candidate) < 2:  # fallback to first 4 chars
            candidate = tok[:4]
        return candidate

    abbrev_map: Dict[str, str] = {}
    used: Dict[str, int] = {}
    for original in columns:
        tokens = split_tokens(original)
        short_tokens = [abbreviate_token(t)[:max_token_length] for t in tokens]
        short_name = '_'.join(short_tokens)
        # Enforce max length for full name (arbitrary 30 chars)
        if len(short_name) > 30:
            parts = []
            current_len = 0
            for st in short_tokens:
                if current_len + len(st) + (1 if parts else 0) > 30:
                    break
                parts.append(st)
                current_len += len(st) + (1 if parts else 0)
            short_name = '_'.join(parts)
        # Uniqueness enforcement
        base = short_name
        if base in used:
            used[base] += 1
            short_name = f"{base}_{used[base]}"
        else:
            used[base] = 1
        abbrev_map[original] = short_name

    return abbrev_map









