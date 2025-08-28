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


def abbreviate_columns(
    columns: Iterable[str],
    custom_rules: Optional[Dict[str, str]] = None,
    max_token_length: int = 12,
    apply_suffixes: bool = True,
) -> Dict[str, str]:
    """Create shortened/abbreviated variants of column names with semantic suffixes.

    Steps:
      1. Tokenize by underscores, non-alphanumerics, and camelCase boundaries -> lowercase tokens.
      2. Apply rule-based token substitutions.
      3. Compress long tokens by vowel removal (post-first letter) to <=4 chars if no rule.
      4. Assemble snake_case name from abbreviated tokens.
    5. Optionally append semantic clarity suffixes:
         • Phone-like columns -> _no (phone number)
         • Name / entity label columns (single entity like country, or containing 'name') -> _nm
         • Email columns -> _id (email treated as natural identifier) e.g. EmailAddress -> eml_id
         • Status columns -> _cd (status code) e.g. order_status -> order_sts_cd
         • Percentage columns -> _pct when containing percentage / percent tokens
         Skip suffix if column already ends with it or appears to be an id.
      6. Enforce uniqueness by incrementing suffix (_2, _3, ...).

    Args:
        columns: Source column names.
        custom_rules: Overrides / additions to base token rules.
        max_token_length: Maximum length per token post-abbreviation.
        apply_suffixes: Toggle semantic suffix addition.

    Returns:
        Mapping original_name -> abbreviated_name.
    """
    rules: Dict[str, str] = {
        # Identity / keys
        'identifier': 'id', 'identity': 'id', 'id': 'id', 'key': 'key', 'sequence': 'seq',
        # Numeric / measures
        'number': 'num', 'nbr': 'num', 'count': 'cnt', 'total': 'tot', 'average': 'avg',
        'minimum': 'min', 'maximum': 'max', 'amount': 'amt', 'quantity': 'qty', 'score': 'scr',
        'rating': 'rtg', 'value': 'val', 'balance': 'bal', 'percent': 'pct', 'percentage': 'pct',
        # Names / descriptive
        'name': 'nm', 'description': 'desc', 'comment': 'cmt', 'remarks': 'rmk', 'remark': 'rmk',
        'note': 'note', 'message': 'msg', 'status': 'sts', 'priority': 'prio', 'level': 'lvl',
        # Temporal
        'date': 'dt', 'datetime': 'dt', 'timestamp': 'ts', 'created': 'crt', 'updated': 'upd',
        'modified': 'mod', 'start': 'start', 'end': 'end',
        # Domain entities
        'customer': 'cust', 'employee': 'emp', 'department': 'dept', 'product': 'prod',
        'project': 'proj', 'organization': 'org', 'account': 'acct', 'transaction': 'txn',
        'invoice': 'inv', 'supplier': 'sup', 'vendor': 'vend', 'purchase': 'purch', 'order': 'ord',
        # Address / geo
        'address': 'addr', 'street': 'st', 'state': 'st', 'country': 'cntry', 'city': 'city',
        'zipcode': 'zip', 'zip': 'zip', 'postal': 'pst', 'location': 'loc', 'latitude': 'lat',
        'longitude': 'lon', 'region': 'rgn', 'province': 'prov',
        # Contact / comms
        'phone': 'ph', 'mobile': 'mob', 'email': 'eml', 'url': 'url', 'ip': 'ip',
        # Classification / codes
        'code': 'cd', 'category': 'cat', 'type': 'typ', 'version': 'ver', 'reference': 'ref',
        'statuscode': 'sts_cd',
        # Parent/child
        'parent': 'prnt', 'child': 'chld',
        # Security / integrity
        'hash': 'hash', 'checksum': 'chksum', 'signature': 'sig',
        # Config / system
        'config': 'cfg', 'configuration': 'cfg', 'setting': 'setg', 'policy': 'plcy', 'rule': 'rule',
        'error': 'err', 'response': 'resp', 'request': 'req',
        # Files / paths
        'file': 'file', 'filename': 'fname', 'extension': 'ext', 'path': 'path', 'size': 'sz',
        # Misc
        'index': 'idx', 'position': 'pos', 'flag': 'flg', 'active': 'actv', 'inactive': 'inactv',
        'enabled': 'enbl', 'disabled': 'dsbl', 'currency': 'ccy'
    }
    if custom_rules:
        for k, v in custom_rules.items():
            rules[k.lower()] = v

    PHONE_TOKENS = {
        'phone', 'telephone', 'tel', 'mobile', 'mob', 'cell', 'cellphone', 'mobilephone'
    }
    EMAIL_TOKENS = {'email', 'emailaddress', 'e_mail', 'mail'}
    NAME_ENTITY_TOKENS = {'country', 'city', 'state', 'province', 'region'}
    STATUS_TOKENS = {'status'}
    PERCENT_TOKENS = {'percent', 'percentage'}

    def split_tokens(name: str) -> List[str]:
        cleaned = re.sub(r'[^0-9A-Za-z]+', '_', name)
        cleaned = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', cleaned)
        toks = [t for t in cleaned.lower().split('_') if t]
        return toks or ['col']

    def abbreviate_token(tok: str) -> str:
        if tok in rules:
            return rules[tok]
        if len(tok) <= 4:
            return tok
        head, tail = tok[0], tok[1:]
        tail_no_vowels = re.sub(r'[aeiou]', '', tail)
        candidate = (head + tail_no_vowels)[:4]
        if len(candidate) < 2:
            candidate = tok[:4]
        return candidate

    abbrev_map: Dict[str, str] = {}
    used: Dict[str, int] = {}

    for original in columns:
        tokens = split_tokens(original)
        short_tokens = [abbreviate_token(t)[:max_token_length] for t in tokens]
        short_name = '_'.join(short_tokens)

        if apply_suffixes:
            has_id = any(t == 'id' or t.endswith('id') for t in tokens) or short_name.endswith('_id')
            token_set = set(tokens)
            lower_original = original.lower()
            # Email rule (precedence)
            is_email = any(t in EMAIL_TOKENS for t in token_set) or 'email' in lower_original
            if is_email and not has_id and not short_name.endswith('_id'):
                short_name += '_id'
            else:
                # Phone rule
                is_phone = any(t in PHONE_TOKENS for t in token_set) or 'phone' in lower_original
                if is_phone and not has_id and not short_name.endswith('_no'):
                    if not re.search(r'(?:_|^)no$', short_name):
                        short_name = re.sub(r'_num$', '', short_name)
                        short_name += '_no'
                else:
                    # Name rule
                    contains_name_token = 'name' in token_set or any(tok.endswith('name') for tok in token_set)
                    single_entity = len(token_set) == 1 and next(iter(token_set)) in NAME_ENTITY_TOKENS
                    if (contains_name_token or single_entity) and not has_id and not short_name.endswith('_nm'):
                        short_name += '_nm'
            # Status rule
            is_status = any(t in STATUS_TOKENS for t in token_set) or 'status' in lower_original
            if is_status and not short_name.endswith('_cd'):
                short_name += '_cd'
            # Percentage rule
            is_percent = any(t in PERCENT_TOKENS for t in token_set)
            if is_percent and not short_name.endswith('_pct') and 'pct' not in short_name.split('_')[-1]:
                short_name += '_pct'

        if len(short_name) > 30:
            parts: List[str] = []
            current_len = 0
            for st in short_tokens:
                next_len = current_len + len(st) + (1 if parts else 0)
                if next_len > 30:
                    break
                parts.append(st)
                current_len = next_len
            short_name = '_'.join(parts)

        base = short_name
        if base in used:
            used[base] += 1
            short_name = f"{base}_{used[base]}"
        else:
            used[base] = 1
        abbrev_map[original] = short_name

    return abbrev_map









