"""
Streamlit Frontend for Intelligent Schema Analyzer

This module provides a simple web interface for:
1. Uploading CSV files (single or multiple)
2. Selecting LLM provider and model
3. Running schema analysis with ReAct agent
4. Displaying SQL DDL scripts and ER diagram specifications
"""

import streamlit as st
import pandas as pd
import os
import json
from typing import Dict, Any, Optional
import traceback

from intelligent_schema_analyzer import IntelligentSchemaAnalyzer
from tool import csvs_jsons_to_dataframes  # Abbreviation/vector store features removed
try:  # Optional import; UI should still work without DB libs
    import psycopg2  # type: ignore
except Exception:  # noqa: BLE001
    psycopg2 = None  # type: ignore


# Helper to JSON-serialize numpy/pandas types

def _json_default(obj):
    import numpy as np
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, set):
        return list(obj)
    return str(obj)


def _normalize_er_for_ui(er: Any) -> Dict[str, Any]:
    """Normalize ER spec so entities is always a dict with attributes list.
    This guards the UI from agent variations (list or dict entities, etc.).
    """
    if not isinstance(er, dict):
        return {"entities": {}, "relationships": []}

    out: Dict[str, Any] = dict(er)
    entities = out.get("entities", {})

    def _attrs_to_list(attrs: Any) -> list[dict]:
        if isinstance(attrs, list):
            # Ensure each item has at least a name
            result: list[dict] = []
            for a in attrs:
                if isinstance(a, dict):
                    if "name" not in a:
                        a = {"name": a.get("column", ""), **a}
                    result.append(a)
                else:
                    result.append({"name": str(a)})
            return result
        if isinstance(attrs, dict):
            return [
                ({"name": k} | (v if isinstance(v, dict) else {}))
                for k, v in attrs.items()
            ]
        return []

    if isinstance(entities, list):
        new_entities: Dict[str, Any] = {}
        for i, e in enumerate(entities, 1):
            if not isinstance(e, dict):
                new_entities[f"entity_{i}"] = {
                    "attributes": [],
                    "row_count": 0,
                }
                continue
            name = (
                e.get("name")
                or e.get("table")
                or e.get("table_name")
                or f"entity_{i}"
            )
            new_entities[name] = {
                "attributes": _attrs_to_list(e.get("attributes", [])),
                "row_count": int(e.get("row_count", 0) or 0),
            }
        out["entities"] = new_entities
    elif isinstance(entities, dict):
        for k, info in list(entities.items()):
            if not isinstance(info, dict):
                entities[k] = {"attributes": [], "row_count": 0}
                continue
            info["attributes"] = _attrs_to_list(info.get("attributes", []))
            info["row_count"] = int(info.get("row_count", 0) or 0)
        out["entities"] = entities
    else:
        out["entities"] = {}

    # Ensure relationships is a list
    rels = out.get("relationships", [])
    if not isinstance(rels, list):
        out["relationships"] = []

    return out


# Configure Streamlit page
st.set_page_config(
    page_title="AI Data Modeller",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Initialize session state variables."""
    if 'dataframes' not in st.session_state:
        st.session_state.dataframes = {}
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'db_config' not in st.session_state:
        st.session_state.db_config = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'mypoc',
            'user': 'sagarbhagwatkar',
            'password': ''
        }
    if 'db_status' not in st.session_state:
        st.session_state.db_status = None
    if 'db_tables' not in st.session_state:
        st.session_state.db_tables = []
    if 'db_selected_tables' not in st.session_state:
        st.session_state.db_selected_tables = []
    if 'db_row_limit' not in st.session_state:
        st.session_state.db_row_limit = 10000


def load_sample_data():
    """Return small in-memory demo dataset."""
    return {
        'customers': pd.DataFrame({
            'customer_id': [1, 2, 3, 4, 5],
            'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Prince', 'Eve Wilson'],
            'email': ['alice@email.com', 'bob@email.com', 'charlie@email.com', 'diana@email.com', 'eve@email.com'],
            'registration_date': pd.to_datetime(['2024-01-15','2024-01-16','2024-01-17','2024-01-18','2024-01-19'])
        }),
        'orders': pd.DataFrame({
            'order_id': [101,102,103,104,105,106],
            'customer_id': [1,1,2,3,4,5],
            'order_date': pd.to_datetime(['2024-01-15','2024-01-16','2024-01-17','2024-01-18','2024-01-19','2024-01-20']),
            'total_amount': [150.99,75.50,200.00,120.25,89.99,300.00],
            'status': ['completed','completed','pending','shipped','completed','processing']
        })
    }


def display_dataframe_summary(dataframes: Dict[str, pd.DataFrame]):
    """Display summary of loaded DataFrames."""
    st.subheader("📊 Loaded Data Summary")
    
    cols = st.columns(len(dataframes))
    
    for i, (name, df) in enumerate(dataframes.items()):
        with cols[i % len(cols)]:
            st.metric(
                label=f"Table: {name}",
                value=f"{len(df)} rows",
                delta=f"{len(df.columns)} columns"
            )
            
            with st.expander(f"View {name} data"):
                st.dataframe(df.head(), use_container_width=True)


def create_analyzer(provider: str, model_name: str,
                    api_key: Optional[str] = None):
    """Create and cache the analyzer instance."""
    try:
        analyzer = IntelligentSchemaAnalyzer(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            verbose=False  # Reduce verbosity for web interface
        )
        st.session_state.analyzer = analyzer
        return analyzer
    except Exception as e:
        st.error(f"Failed to initialize analyzer: {str(e)}")
        return None


def run_analysis():
    """Run the schema analysis."""
    if not st.session_state.dataframes:
        st.error("No data loaded. Please upload CSV or JSON files first.")
        return
    
    if not st.session_state.analyzer:
        st.error("Analyzer not initialized. Please configure LLM settings.")
        return
    
    try:
        # Load data into analyzer
        st.session_state.analyzer.load_data(st.session_state.dataframes)
        
        # Run analysis with progress bar
        with st.spinner("🧠 Running comprehensive schema analysis..."):
            results = st.session_state.analyzer.analyze_schema_with_agent()
            st.session_state.analysis_results = results
        
        st.success("✅ Analysis complete!")
        return results
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        st.error("Please check your LLM configuration and try again.")
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        return None


def display_analysis_results(results: Dict[str, Any]):
    """Display the analysis results in organized tabs."""
    if not results:
        return
    
    # Create tabs for different result sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Summary", "💾 SQL DDL", "🔗 ER Diagram", "🧠 Agent Process"
    ])
    
    with tab1:
        st.subheader("Analysis Summary")
        st.text(results.get("summary", "No summary available"))
        
        # Show basic statistics
        if "er_diagram" in results and "entities" in results["er_diagram"]:
            normalized = _normalize_er_for_ui(results["er_diagram"])
            entities = normalized["entities"]
            relationships = normalized.get("relationships", [])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tables Analyzed", len(entities))
            with col2:
                st.metric("Relationships Found", len(relationships))
            with col3:
                total_attributes = sum(
                    len(ent.get("attributes", [])) for ent in entities.values()
                )
                st.metric("Total Attributes", total_attributes)
    
    with tab2:
        st.subheader("Generated SQL DDL Script")
        sql_ddl = results.get("sql_ddl", "No SQL DDL generated")
        st.code(sql_ddl, language="sql")
        
        # Download button for DDL
        st.download_button(
            label="📥 Download SQL DDL",
            data=sql_ddl,
            file_name="schema.sql",
            mime="text/sql"
        )
    
    with tab3:
        st.subheader("Entity-Relationship Diagram Specification")
        
        if "er_diagram" in results:
            er_data = results["er_diagram"]
            er_data = _normalize_er_for_ui(er_data)
            
            # Display entities
            if "entities" in er_data:
                st.write("**Entities:**")
                for entity_name, entity_info in er_data["entities"].items():
                    with st.expander(f"📋 {entity_name}"):
                        st.write(f"**Row Count:** "
                                 f"{entity_info.get('row_count', 'N/A')}")
                        
                        # Show attributes in a table
                        if "attributes" in entity_info:
                            attrs_df = pd.DataFrame(entity_info["attributes"])
                            st.dataframe(attrs_df, use_container_width=True)
            
            # Display relationships
            if "relationships" in er_data and er_data["relationships"]:
                st.write("**Relationships:**")
                for rel in er_data["relationships"]:
                    st.write(
                        f"• **{rel.get('parent_entity', '')}** → "
                        f"**{rel.get('child_entity', '')}** "
                        f"(via `{rel.get('foreign_key', '')}`)"
                    )

            # Visual ER diagram (Graphviz)
            st.subheader("Visual ER Diagram")
            dot = _er_spec_to_dot(er_data)
            st.graphviz_chart(dot, use_container_width=True)

            # Downloads
            st.download_button(
                label="📥 Download ER Specification",
                data=json.dumps(er_data, indent=2, default=_json_default),
                file_name="er_diagram_spec.json",
                mime="application/json"
            )
            st.download_button(
                label="📥 Download ERD (DOT)",
                data=dot,
                file_name="erd.dot",
                mime="text/vnd.graphviz"
            )
    
    with tab4:
        st.subheader("Agent Reasoning Process")
        
        if ("analysis" in results and
                "reasoning_chain" in results["analysis"]):
            reasoning_chain = results["analysis"]["reasoning_chain"]
            
            if reasoning_chain:
                st.write("**Agent's Step-by-Step Analysis:**")
                for i, step in enumerate(reasoning_chain, 1):
                    with st.expander(f"Step {i}: {step['action']}"):
                        st.text(step['observation'])
            else:
                st.info("No detailed reasoning chain available.")
        
        # Show agent output
        if ("analysis" in results and
                "agent_output" in results["analysis"]):
            st.write("**Final Agent Output:**")
            st.text(results["analysis"]["agent_output"])


def _er_spec_to_dot(er_data: Dict[str, Any]) -> str:
    """Convert ER JSON spec to Graphviz DOT string (vertical layout)."""
    entities = er_data.get("entities", {})
    relationships = er_data.get("relationships", [])

    lines: list[str] = [
        "digraph ERD {",
        "  rankdir=TB;",  # top-to-bottom (vertical)
        '  node [shape=record, fontsize=12, fontname="Helvetica"];',
        '  edge [color="#2563eb", penwidth=1.4];',
    ]

    # Nodes (tables)
    for table, info in entities.items():
        attrs = info.get("attributes", [])
        # Build record-style label with PK marker
        attr_lines = []
        for attr in attrs:
            name = attr.get("name", "")
            is_pk = bool(attr.get("is_primary_key", False))
            marker = " 🔑" if is_pk else ""
            # Use left-justified line terminator \l
            attr_lines.append(f"{name}{marker}\\l")
        body = "".join(attr_lines)
        label = f"{{{table}|{body}}}"
        lines.append(f'  "{table}" [label="{label}"];')

    # Edges (relationships)
    for rel in relationships:
        parent = rel.get("parent_entity")
        child = rel.get("child_entity")
        fk = rel.get("foreign_key", "")
        lines.append(
            f'  "{parent}" -> "{child}" [label="{fk}", arrowsize=0.8];'
        )

    lines.append("}")
    return "\n".join(lines)


def main():
    """Main Streamlit application."""
    init_session_state()
    
    # Header
    st.title("🧠 AI Data Modeller")
    st.markdown("**Intelligent Schema Analysis with ReAct Agent**")
    st.markdown(
        "Upload CSV and/or JSON files (nested JSON supported). Nested list-of-object structures become separate tables; the analyzer will generate SQL DDL and ER diagrams."
    )

    # Vector store / abbreviation functionality removed per user request.
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data Upload Section
        st.subheader("📂 Data Upload")
        
        # Option to use sample data
        if st.button("📊 Use Sample Data", type="secondary"):
            st.session_state.dataframes = load_sample_data()
            st.success("Sample data loaded!")
        
        # File uploader (CSV + JSON + Excel)
        uploaded_files = st.file_uploader(
            "Upload CSV, JSON, or Excel files",
            type=["csv", "json", "xlsx", "xls"],
            accept_multiple_files=True,
            help=(
                "Upload CSV, JSON (nested supported), or Excel workbooks. "
                "Nested JSON lists of objects become separate tables; each Excel sheet becomes a table."
            )
        )

        if uploaded_files:
            try:
                loaded = csvs_jsons_to_dataframes(uploaded_files)
                loaded = {k: v for k, v in loaded.items() if v is not None}
                if not loaded:
                    st.error("All uploaded files failed to load.")
                else:
                    st.session_state.dataframes = loaded
                    for name, df in loaded.items():
                        st.success(f"✅ Loaded: {name} ({len(df)} rows, {len(df.columns)} cols)")
                    nested_tables = [n for n in loaded if '__' in n]
                    if nested_tables:
                        st.info("Nested JSON produced tables: " + ", ".join(nested_tables))
            except Exception as e:  # noqa: BLE001
                st.error(f"Failed to process uploaded files: {e}")
        
        st.divider()

        
        # LLM Configuration
        st.subheader("🤖 LLM Configuration")
        
        provider = st.selectbox(
            "Choose LLM Provider",
            options=["groq", "openai", "ollama"],
            format_func=lambda x: {
                "groq": "GROQ (Fast, requires API key)",
                "openai": "OpenAI (High quality, requires API key)",
                "ollama": "Ollama (Local, no API key needed)"
            }[x]
        )
        
        # Model selection based on provider
        if provider == "groq":
            model_name = st.selectbox(
                "Model",
                ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"]
            )
            api_key = st.text_input(
                "GROQ API Key", type="password",
                help="Enter your GROQ API key"
            )
            if not api_key:
                api_key = os.getenv("GROQ_API_KEY")
        
        elif provider == "openai":
            model_name = st.selectbox(
                "Model",
                [
                    "gpt-3.5-turbo",
                    "gpt-4",
                    "gpt-4-turbo",
                ],
            )
            api_key = st.text_input(
                "OpenAI API Key", type="password",
                help="Enter your OpenAI API key"
            )
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
        
        else:  # ollama
            model_name = st.selectbox(
                "Model", ["llama3", "qwen3:14b", "gpt-oss:20b", "deepseek-r1:8b"]
            )
            api_key = None
        
        # Initialize analyzer button
        if st.button("🚀 Initialize Analyzer", type="primary"):
            analyzer = create_analyzer(provider, model_name, api_key)
            if analyzer:
                st.success(
                    f"✅ Analyzer initialized with {provider}/{model_name}"
                )
        
        st.divider()

        # Database Configuration
        st.subheader("🗄️ Database Connection")
        st.caption("Configure PostgreSQL connection (optional)")
        with st.expander("Configure Database", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                host = st.text_input("Host", value=st.session_state.db_config['host'])
                dbname = st.text_input("Database", value=st.session_state.db_config['dbname'])
                user = st.text_input("User", value=st.session_state.db_config['user'])
            with col_b:
                port = st.number_input("Port", value=int(st.session_state.db_config['port']), step=1)
                password = st.text_input("Password", value=st.session_state.db_config['password'], type="password")
                fetch_button = st.button("🔌 Test Connection & List Tables", use_container_width=True)

            # Update session state with latest entered values
            st.session_state.db_config.update({
                'host': host,
                'port': int(port),
                'dbname': dbname,
                'user': user,
                'password': password,
            })

            if fetch_button:
                if psycopg2 is None:
                    st.error("psycopg2 not installed. Install dependencies to enable DB connectivity.")
                else:
                    cfg = st.session_state.db_config
                    try:
                        with st.spinner("Connecting to database..."):
                            conn = psycopg2.connect(
                                host=cfg['host'],
                                port=cfg['port'],
                                dbname=cfg['dbname'],
                                user=cfg['user'],
                                password=cfg['password'] or None,
                            )
                            with conn.cursor() as cur:
                                cur.execute(
                                    """
                                    SELECT table_schema, table_name
                                    FROM information_schema.tables
                                    WHERE table_schema NOT IN ('pg_catalog','information_schema')
                                    ORDER BY table_schema, table_name
                                    """
                                )
                                rows = cur.fetchall()
                            conn.close()
                        st.session_state.db_tables = rows
                        st.session_state.db_status = "success"
                        st.success(f"Connected. Found {len(rows)} tables.")
                    except Exception as e:  # noqa: BLE001
                        st.session_state.db_status = "error"
                        st.session_state.db_tables = []
                        st.error(f"Connection failed: {e}")
        # Show tables if available
        if st.session_state.db_tables:
            with st.expander("📋 Database Tables (Current Session)", expanded=False):
                # Selection controls
                all_table_labels = [f"{schema}.{table}" for schema, table in st.session_state.db_tables]
                select_all = st.checkbox("Select All Tables", value=False, key="db_select_all")
                if select_all:
                    current_selection = all_table_labels
                else:
                    current_selection = st.multiselect(
                        "Choose tables to load", all_table_labels,
                        default=st.session_state.db_selected_tables
                    )
                st.session_state.db_selected_tables = current_selection
                st.session_state.db_row_limit = st.number_input(
                    "Row Limit (0 = all)", min_value=0, value=int(st.session_state.db_row_limit), step=1000
                )
                load_btn = st.button("⬇️ Load Selected Tables", use_container_width=True)

                if load_btn and current_selection:
                    if psycopg2 is None:
                        st.error("psycopg2 not installed.")
                    else:
                        loaded = {}
                        cfg = st.session_state.db_config
                        row_limit = int(st.session_state.db_row_limit)
                        try:
                            with st.spinner("Loading tables from database..."):
                                conn = psycopg2.connect(
                                    host=cfg['host'], port=cfg['port'], dbname=cfg['dbname'],
                                    user=cfg['user'], password=cfg['password'] or None,
                                )
                                for label in current_selection:
                                    schema, table = label.split('.', 1)
                                    sql = f'SELECT * FROM "{schema}"."{table}"'
                                    if row_limit > 0:
                                        sql += f" LIMIT {row_limit}"
                                    df = pd.read_sql(sql, conn)  # type: ignore
                                    # Key naming: if schema != public, prefix schema
                                    key = table if schema == 'public' else f"{schema}_{table}"
                                    loaded[key] = df
                                conn.close()
                            # Merge with existing dataframes (overwrite duplicates)
                            st.session_state.dataframes.update(loaded)
                            st.success(f"Loaded {len(loaded)} tables into workspace.")
                        except Exception as e:  # noqa: BLE001
                            st.error(f"Failed loading tables: {e}")
                elif load_btn and not current_selection:
                    st.warning("No tables selected.")
        
        # Analysis controls
        st.subheader("🔍 Analysis")
        
        analyze_button = st.button(
            "🧠 Run Analysis",
            type="primary",
            disabled=not (
                st.session_state.dataframes and st.session_state.analyzer
            ),
            help="Analyze loaded data and generate schema",
        )
    
    # Main content area
    if st.session_state.dataframes:
        display_dataframe_summary(st.session_state.dataframes)
        st.divider()
    
    # Run analysis when button is clicked
    if analyze_button:
        results = run_analysis()
        if results:
            display_analysis_results(results)
    
    # Display existing results if available
    elif st.session_state.analysis_results:
        st.subheader("📋 Analysis Results")
        display_analysis_results(st.session_state.analysis_results)
    
    # Instructions when no data is loaded
    else:
        st.info("""
        👋 **Welcome to AI Data Modeller!**
        
        To get started:
    1. **Upload CSV or JSON files** using the sidebar file uploader, or click
           "Use Sample Data"
        2. **Configure your LLM provider** (GROQ, OpenAI, or Ollama)
        3. **Initialize the analyzer** with your chosen settings
        4. **Run the analysis** to generate SQL DDL and ER diagrams
        
        The ReAct agent will systematically analyze your data and provide
        comprehensive database schema recommendations.
        """)


if __name__ == "__main__":
    main()
