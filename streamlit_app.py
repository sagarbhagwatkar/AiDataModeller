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


def load_sample_data():
    """Load sample e-commerce data for demo."""
    return {
        'customers': pd.DataFrame({
            'customer_id': [1, 2, 3, 4, 5],
            'name': [
                'Alice Johnson', 'Bob Smith', 'Charlie Brown',
                'Diana Prince', 'Eve Wilson'
            ],
            'email': [
                'alice@email.com', 'bob@email.com', 'charlie@email.com',
                'diana@email.com', 'eve@email.com'
            ],
            'registration_date': pd.to_datetime([
                '2024-01-15', '2024-01-16', '2024-01-17',
                '2024-01-18', '2024-01-19'
            ])
        }),
        'orders': pd.DataFrame({
            'order_id': [101, 102, 103, 104, 105, 106],
            'customer_id': [1, 1, 2, 3, 4, 5],
            'order_date': pd.to_datetime([
                '2024-01-15', '2024-01-16', '2024-01-17',
                '2024-01-18', '2024-01-19', '2024-01-20'
            ]),
            'total_amount': [150.99, 75.50, 200.00, 120.25, 89.99, 300.00],
            'status': [
                'completed', 'completed', 'pending',
                'shipped', 'completed', 'processing'
            ]
        }),
        'products': pd.DataFrame({
            'product_id': [201, 202, 203, 204, 205],
            'name': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones'],
            'category': [
                'Electronics', 'Accessories', 'Accessories',
                'Electronics', 'Electronics'
            ],
            'price': [999.99, 29.99, 79.99, 299.99, 149.99],
            'stock_quantity': [50, 100, 75, 25, 60]
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
        st.error("No data loaded. Please upload CSV files first.")
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
            entities = results["er_diagram"]["entities"]
            relationships = results["er_diagram"].get("relationships", [])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tables Analyzed", len(entities))
            with col2:
                st.metric("Relationships Found", len(relationships))
            with col3:
                total_attributes = sum(
                    len(entity["attributes"]) for entity in entities.values()
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
                        f"• **{rel['parent_entity']}** → "
                        f"**{rel['child_entity']}** "
                        f"(via `{rel['foreign_key']}`)"
                    )
            
            # Download button for ER spec
            st.download_button(
                label="📥 Download ER Specification",
                data=json.dumps(er_data, indent=2, default=_json_default),
                file_name="er_diagram_spec.json",
                mime="application/json"
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


def main():
    """Main Streamlit application."""
    init_session_state()
    
    # Header
    st.title("🧠 AI Data Modeller")
    st.markdown("**Intelligent Schema Analysis with ReAct Agent**")
    st.markdown(
        "Upload your CSV files and generate comprehensive database schemas "
        "with SQL DDL and ER diagrams."
    )
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data Upload Section
        st.subheader("📂 Data Upload")
        
        # Option to use sample data
        if st.button("📊 Use Sample Data", type="secondary"):
            st.session_state.dataframes = load_sample_data()
            st.success("Sample data loaded!")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload CSV Files",
            type=['csv'],
            accept_multiple_files=True,
            help="Upload one or more CSV files for analysis"
        )
        
        if uploaded_files:
            dataframes = {}
            for uploaded_file in uploaded_files:
                try:
                    df = pd.read_csv(uploaded_file)
                    table_name = uploaded_file.name.replace('.csv', '')
                    dataframes[table_name] = df
                    st.success(f"✅ Loaded: {table_name} ({len(df)} rows)")
                except Exception as e:
                    st.error(
                        f"❌ Failed to load {uploaded_file.name}: {str(e)}"
                    )
            
            if dataframes:
                st.session_state.dataframes = dataframes
        
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
                    "gpt-5-preview",
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
                "Model", ["llama3", "codellama", "phi3", "mistral"]
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
        1. **Upload CSV files** using the sidebar file uploader, or click
           "Use Sample Data"
        2. **Configure your LLM provider** (GROQ, OpenAI, or Ollama)
        3. **Initialize the analyzer** with your chosen settings
        4. **Run the analysis** to generate SQL DDL and ER diagrams
        
        The ReAct agent will systematically analyze your data and provide
        comprehensive database schema recommendations.
        """)


if __name__ == "__main__":
    main()
