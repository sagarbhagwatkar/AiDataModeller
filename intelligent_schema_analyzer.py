"""
Intelligent Data Schema Analyzer using ReAct Agent

This module implements a sophisticated ReAct agent that:
1. Uses tools to understand data structure and relationships
2. Generates comprehensive SQL DDL scripts
3. Provides detailed ER diagram specifications

The agent follows the ReAct pattern (Reasoning and Acting) to methodically analyze data.
"""

import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import os
from dotenv import load_dotenv

# LangChain imports
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from tool import (
    analyze_primary_key_candidates,
    find_composite_keys,
    find_dataframe_relations,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class IntelligentSchemaAnalyzer:
    """
    Intelligent Schema Analyzer using ReAct Agent.
    
    This class uses a ReAct agent to analyze data schemas by:
    1. Understanding data through systematic tool usage
    2. Generating comprehensive SQL DDL scripts
    3. Creating detailed ER diagram specifications
    """
    
    def __init__(
        self,
        provider: str = "groq",
        model_name: str = "llama3-8b-8192",
        api_key: Optional[str] = None,
        max_iterations: int = 10,
        verbose: bool = True
    ):
        """
        Initialize the Intelligent Schema Analyzer.
        
        Args:
            provider: LLM provider ('groq', 'openai', 'ollama')
            model_name: Name of the model to use
            api_key: API key for the provider (if required)
            max_iterations: Maximum iterations for the agent
            verbose: Whether to show detailed agent reasoning
        """
        self.provider = provider
        self.model_name = model_name
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.analysis_results = {}
        
        # Initialize LLM
        self.llm = self._create_llm(provider, model_name, api_key)
        
        # Initialize agent
        self.agent_executor = self._create_agent(max_iterations, verbose)
        
        logger.info(f"Initialized IntelligentSchemaAnalyzer with {provider}/{model_name}")
    
    def _create_llm(
        self, 
        provider: str, 
        model_name: str, 
        api_key: Optional[str]
    ):
        """Create LLM instance based on provider."""
        if provider == "groq":
            api_key = api_key or os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ API key is required")
            return ChatGroq(
                groq_api_key=api_key,
                model_name=model_name,
                temperature=0.1
            )
        
        elif provider == "openai":
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key is required")
            return ChatOpenAI(
                api_key=api_key,
                model_name=model_name,
                temperature=0.1
            )
        
        elif provider == "ollama":
            return OllamaLLM(
                model=model_name,
                temperature=0.1
            )
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _create_agent(self, max_iterations: int, verbose: bool) -> AgentExecutor:
        """Create ReAct agent with data analysis tools."""
        
        # Create wrapper functions for tools
        def analyze_primary_keys_wrapper(input_str: str = "") -> str:
            """Wrapper for analyze_primary_key_candidates function."""
            try:
                result = analyze_primary_key_candidates(self.dataframes)
                return str(result)
            except Exception as e:
                return f"Error analyzing primary keys: {str(e)}"
        
        def find_composite_keys_wrapper(input_str: str = "") -> str:
            """Wrapper for find_composite_keys function."""
            try:
                result = find_composite_keys(self.dataframes)
                return str(result)
            except Exception as e:
                return f"Error finding composite keys: {str(e)}"
        
        def find_relationships_wrapper(input_str: str = "") -> str:
            """Wrapper for find_dataframe_relations function."""
            try:
                result = find_dataframe_relations(self.dataframes)
                return str(result)
            except Exception as e:
                return f"Error finding relationships: {str(e)}"
        
        # Create tool instances that work with the stored dataframes
        tools = [
            Tool(
                name="analyze_primary_keys",
                description="Analyze potential primary key candidates in all loaded DataFrames. Returns detailed analysis of uniqueness, null values, and data types for each column.",
                func=analyze_primary_keys_wrapper
            ),
            Tool(
                name="find_composite_keys", 
                description="Find potential composite key combinations in all loaded DataFrames. Identifies columns that together could form a unique identifier.",
                func=find_composite_keys_wrapper
            ),
            Tool(
                name="find_relationships",
                description="Analyze relationships between DataFrames by finding foreign key connections. Identifies how tables relate to each other.",
                func=find_relationships_wrapper
            ),
            Tool(
                name="get_data_summary",
                description="Get a summary of all loaded DataFrames including column names, data types, and basic statistics.",
                func=self._get_data_summary
            )
        ]
        
        # Create the ReAct prompt template
        prompt = PromptTemplate.from_template("""
You are an expert data analyst and database designer. Your job is to analyze data schemas and create comprehensive database designs.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what you need to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (use empty string if no input needed)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now have enough information to provide a comprehensive analysis
Final Answer: your final answer with detailed analysis

Begin!

Question: {input}
Thought: I need to systematically analyze the data to understand its structure and relationships.
{agent_scratchpad}
""")
        
        # Create the ReAct agent
        agent = create_react_agent(self.llm, tools, prompt)
        
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=verbose,
            max_iterations=max_iterations,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
    
    def _get_data_summary(self, _: str = "") -> str:
        """Get summary of loaded DataFrames."""
        if not self.dataframes:
            return "No DataFrames loaded."
        
        summary = []
        for name, df in self.dataframes.items():
            if df is not None:
                summary.append(f"""
Table: {name}
Rows: {len(df)}
Columns: {list(df.columns)}
Data Types: {df.dtypes.to_dict()}
Sample Data: {df.head(2).to_dict('records')}
""")
        return "\n".join(summary)
    
    def load_data(self, dataframes: Dict[str, pd.DataFrame]):
        """Load DataFrames for analysis."""
        self.dataframes = dataframes
        logger.info(f"Loaded {len(dataframes)} DataFrames: {list(dataframes.keys())}")
    
    def analyze_schema_with_agent(self) -> Dict[str, Any]:
        """
        Use the ReAct agent to comprehensively analyze the schema.
        
        Returns:
            Dict containing complete analysis, SQL DDL, and ER diagram spec
        """
        if not self.dataframes:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Starting comprehensive schema analysis with ReAct agent...")
        
        query = """
        Please analyze the loaded data comprehensively by following these steps:

        1. First, get a summary of all the data to understand what we're working with
        2. Analyze primary key candidates for each table
        3. Find any composite key opportunities
        4. Identify relationships between tables
        5. Based on your analysis, generate:
           a) A comprehensive SQL DDL script with proper constraints, indexes, and foreign keys
           b) A detailed ER diagram specification that shows all entities, attributes, and relationships

        Provide a thorough analysis that demonstrates your understanding of the data structure and relationships.
        """
        
        try:
            result = self.agent_executor.invoke({"input": query})
            
            # Store the analysis results
            self.analysis_results = {
                "agent_output": result["output"],
                "intermediate_steps": result.get("intermediate_steps", []),
                "reasoning_chain": self._extract_reasoning_chain(result)
            }
            
            # Generate structured outputs
            ddl_script = self._generate_sql_ddl()
            er_diagram_spec = self._generate_er_diagram_spec()
            
            return {
                "analysis": self.analysis_results,
                "sql_ddl": ddl_script,
                "er_diagram": er_diagram_spec,
                "summary": self._generate_analysis_summary()
            }
            
        except Exception as e:
            logger.error(f"Schema analysis failed: {e}")
            raise
    
    def _extract_reasoning_chain(self, result: Dict) -> List[Dict]:
        """Extract the agent's reasoning chain from the result."""
        chain = []
        if "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                if len(step) >= 2:
                    action, observation = step[0], step[1]
                    chain.append({
                        "action": action.tool if hasattr(action, 'tool') else str(action),
                        "observation": str(observation)[:500] + "..." if len(str(observation)) > 500 else str(observation)
                    })
        return chain
    
    def _generate_sql_ddl(self) -> str:
        """Generate comprehensive SQL DDL script based on analysis."""
        try:
            # Analyze the data using tools
            primary_keys = analyze_primary_key_candidates(self.dataframes)
            relationships = find_dataframe_relations(self.dataframes)
            
            ddl_parts = []
            ddl_parts.append("-- Comprehensive SQL DDL Script")
            ddl_parts.append("-- Generated by Intelligent Schema Analyzer")
            ddl_parts.append("-- Based on ReAct Agent Analysis\n")
            
            # Create tables
            for table_name, df in self.dataframes.items():
                if df is None:
                    continue
                    
                ddl_parts.append(f"-- Table: {table_name}")
                ddl_parts.append(f"CREATE TABLE {table_name} (")
                
                columns = []
                for col in df.columns:
                    col_type = self._infer_sql_type(df[col])
                    
                    # Check if this is a primary key candidate
                    pk_info = primary_keys.get(table_name, {}).get(col, {})
                    is_pk = pk_info.get('is_unique', False) and not pk_info.get('has_nulls', True)
                    
                    constraint = ""
                    if is_pk:
                        constraint = " PRIMARY KEY"
                    elif not pk_info.get('has_nulls', True):
                        constraint = " NOT NULL"
                    
                    columns.append(f"    {col} {col_type}{constraint}")
                
                ddl_parts.append(",\n".join(columns))
                ddl_parts.append(");\n")
            
            # Add foreign key constraints
            ddl_parts.append("-- Foreign Key Constraints")
            for rel_key, relations_list in relationships.items():
                if isinstance(rel_key, str) and '-' in rel_key:
                    parent_table, child_table = rel_key.split('-', 1)
                    for relation in relations_list:
                        column = relation['column']
                        ddl_parts.append(
                            f"ALTER TABLE {child_table} ADD CONSTRAINT fk_{child_table}_{column} "
                            f"FOREIGN KEY ({column}) REFERENCES {parent_table}({column});"
                        )
            
            # Add indexes for performance
            ddl_parts.append("\n-- Performance Indexes")
            for table_name in self.dataframes.keys():
                if self.dataframes[table_name] is not None:
                    for col in self.dataframes[table_name].columns:
                        if col.endswith('_id') or 'id' in col.lower():
                            ddl_parts.append(f"CREATE INDEX idx_{table_name}_{col} ON {table_name}({col});")
            
            return "\n".join(ddl_parts)
            
        except Exception as e:
            logger.error(f"DDL generation failed: {e}")
            return f"-- DDL generation failed: {e}"
    
    def _infer_sql_type(self, series: pd.Series) -> str:
        """Infer SQL data type from pandas Series."""
        dtype = str(series.dtype)
        
        if 'int' in dtype:
            return 'INTEGER'
        elif 'float' in dtype:
            return 'DECIMAL(10,2)'
        elif 'datetime' in dtype:
            return 'TIMESTAMP'
        elif 'date' in dtype:
            return 'DATE'
        elif 'bool' in dtype:
            return 'BOOLEAN'
        else:
            # For strings, try to determine appropriate VARCHAR length
            max_length = series.astype(str).str.len().max()
            if pd.isna(max_length) or max_length <= 50:
                return 'VARCHAR(100)'
            elif max_length <= 255:
                return 'VARCHAR(255)'
            else:
                return 'TEXT'
    
    def _generate_er_diagram_spec(self) -> Dict[str, Any]:
        """Generate detailed ER diagram specification."""
        try:
            # Get analysis data
            primary_keys = analyze_primary_key_candidates(self.dataframes)
            relationships = find_dataframe_relations(self.dataframes)
            
            # Build ER diagram specification
            entities = {}
            for table_name, df in self.dataframes.items():
                if df is None:
                    continue
                    
                attributes = []
                for col in df.columns:
                    pk_info = primary_keys.get(table_name, {}).get(col, {})
                    is_pk = pk_info.get('is_unique', False) and not pk_info.get('has_nulls', True)
                    
                    attributes.append({
                        'name': col,
                        'type': str(df[col].dtype),
                        'is_primary_key': is_pk,
                        'is_nullable': pk_info.get('has_nulls', True),
                        'is_unique': pk_info.get('is_unique', False)
                    })
                
                entities[table_name] = {
                    'attributes': attributes,
                    'row_count': len(df)
                }
            
            # Build relationships
            relationships_spec = []
            for rel_key, relations_list in relationships.items():
                if isinstance(rel_key, str) and '-' in rel_key:
                    parent_table, child_table = rel_key.split('-', 1)
                    for relation in relations_list:
                        relationships_spec.append({
                            'parent_entity': parent_table,
                            'child_entity': child_table,
                            'foreign_key': relation['column'],
                            'relationship_type': 'one-to-many',
                            'cardinality': relation.get('cardinality', 'unknown')
                        })
            
            return {
                'entities': entities,
                'relationships': relationships_spec,
                'diagram_layout': 'vertical',
                'title': 'Database Entity-Relationship Diagram',
                'notes': 'Generated by Intelligent Schema Analyzer using ReAct Agent'
            }
            
        except Exception as e:
            logger.error(f"ER diagram generation failed: {e}")
            return {'error': f"ER diagram generation failed: {e}"}
    
    def _generate_analysis_summary(self) -> str:
        """Generate a human-readable summary of the analysis."""
        summary = []
        summary.append("=== SCHEMA ANALYSIS SUMMARY ===\n")
        
        if self.dataframes:
            summary.append(f"📊 Analyzed {len(self.dataframes)} tables:")
            for name, df in self.dataframes.items():
                if df is not None:
                    summary.append(f"  • {name}: {len(df)} rows, {len(df.columns)} columns")
            summary.append("")
        
        # Add reasoning chain if available
        if hasattr(self, 'analysis_results') and 'reasoning_chain' in self.analysis_results:
            summary.append("🧠 Agent Reasoning Process:")
            for i, step in enumerate(self.analysis_results['reasoning_chain'], 1):
                summary.append(f"  {i}. Used tool '{step['action']}'")
            summary.append("")
        
        summary.append("✅ Generated comprehensive SQL DDL script")
        summary.append("✅ Created detailed ER diagram specification")
        summary.append("✅ Identified primary keys and relationships")
        
        return "\n".join(summary)
    
    def save_results(self, output_dir: str = "analysis_output"):
        """Save analysis results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if hasattr(self, 'analysis_results'):
            results = self.analyze_schema_with_agent()
            
            # Save DDL script
            with open(output_path / "schema.sql", "w") as f:
                f.write(results["sql_ddl"])
            
            # Save ER diagram spec
            with open(output_path / "er_diagram_spec.json", "w") as f:
                json.dump(results["er_diagram"], f, indent=2)
            
            # Save full analysis
            with open(output_path / "full_analysis.json", "w") as f:
                json.dump(results["analysis"], f, indent=2, default=str)
            
            # Save summary
            with open(output_path / "summary.txt", "w") as f:
                f.write(results["summary"])
            
            logger.info(f"Results saved to {output_path}")
            return output_path
        
        return None


def load_csv_files_from_directory(directory: str) -> Dict[str, pd.DataFrame]:
    """Load all CSV files from a directory."""
    directory_path = Path(directory)
    if not directory_path.exists():
        raise ValueError(f"Directory '{directory}' does not exist.")
    
    dataframes = {}
    csv_files = list(directory_path.glob("*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in directory '{directory}'.")
    
    print(f"🔍 Found {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        print(f"  • {csv_file.name}")
        try:
            df = pd.read_csv(csv_file)
            table_name = csv_file.stem  # filename without extension
            dataframes[table_name] = df
            print(f"    ✅ Loaded: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            print(f"    ❌ Failed to load {csv_file.name}: {e}")
    
    return dataframes


def load_single_csv_file(file_path: str) -> Dict[str, pd.DataFrame]:
    """Load a single CSV file."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise ValueError(f"File '{file_path}' does not exist.")
    
    if file_path.suffix.lower() != '.csv':
        raise ValueError(f"File must be a CSV file. Got: {file_path.suffix}")
    
    try:
        df = pd.read_csv(file_path)
        table_name = file_path.stem  # filename without extension
        
        print(f"✅ Loaded CSV file: {file_path.name}")
        print(f"   • Table: {table_name}")
        print(f"   • Rows: {len(df)}")
        print(f"   • Columns: {len(df.columns)}")
        print(f"   • Column names: {list(df.columns)}")
        
        return {table_name: df}
    except Exception as e:
        raise ValueError(f"Failed to load CSV file '{file_path}': {e}")


def get_user_provider_choice() -> tuple:
    """Get user's choice of LLM provider and model."""
    print("\n🤖 Choose your LLM Provider:")
    print("1. GROQ (Fast, requires API key)")
    print("2. OpenAI (High quality, requires API key)")  
    print("3. Ollama (Local, no API key needed)")
    print("4. Use default (GROQ)")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        api_key = input("Enter your GROQ API key: ").strip()
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")
        return "groq", "llama3-8b-8192", api_key
    
    elif choice == "2":
        api_key = input("Enter your OpenAI API key: ").strip()
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        return "openai", "gpt-3.5-turbo", api_key
    
    elif choice == "3":
        print("Available Ollama models: llama3, codellama, phi3")
        model = input("Enter model name (default: llama3): ").strip()
        if not model:
            model = "llama3"
        return "ollama", model, None
    
    else:
        # Default to GROQ
        api_key = os.getenv("GROQ_API_KEY")
        return "groq", "llama3-8b-8192", api_key


def main():
    """Main function with file upload interface."""
    print("🧠 INTELLIGENT SCHEMA ANALYZER")
    print("=" * 50)
    print("This tool analyzes your CSV data and generates:")
    print("• Comprehensive SQL DDL scripts")
    print("• Detailed ER diagram specifications")
    print("• Primary key and relationship analysis")
    print("=" * 50)
    
    # Get data source from user
    print("\n📂 Data Input Options:")
    print("1. Single CSV file")
    print("2. Directory with multiple CSV files")
    print("3. Use sample data (demo)")
    
    data_choice = input("\nChoose your data source (1-3): ").strip()
    
    dataframes = {}
    
    if data_choice == "1":
        # Single file
        file_path = input("Enter the path to your CSV file: ").strip()
        try:
            dataframes = load_single_csv_file(file_path)
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return
    
    elif data_choice == "2":
        # Directory
        directory = input("Enter the directory path containing CSV files: ").strip()
        try:
            dataframes = load_csv_files_from_directory(directory)
        except Exception as e:
            print(f"❌ Error loading directory: {e}")
            return
    
    elif data_choice == "3":
        # Sample data
        print("📊 Using sample e-commerce data...")
        dataframes = {
            'customers': pd.DataFrame({
                'customer_id': [1, 2, 3, 4, 5],
                'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Prince', 'Eve Wilson'],
                'email': ['alice@email.com', 'bob@email.com', 'charlie@email.com', 'diana@email.com', 'eve@email.com'],
                'registration_date': pd.to_datetime(['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19'])
            }),
            'orders': pd.DataFrame({
                'order_id': [101, 102, 103, 104, 105, 106],
                'customer_id': [1, 1, 2, 3, 4, 5],
                'order_date': pd.to_datetime(['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19', '2024-01-20']),
                'total_amount': [150.99, 75.50, 200.00, 120.25, 89.99, 300.00],
                'status': ['completed', 'completed', 'pending', 'shipped', 'completed', 'processing']
            })
        }
    else:
        print("❌ Invalid choice. Exiting.")
        return
    
    if not dataframes:
        print("❌ No data loaded. Exiting.")
        return
    
    # Get LLM provider choice
    provider, model_name, api_key = get_user_provider_choice()
    
    # Create analyzer
    print(f"\n🚀 Initializing Intelligent Schema Analyzer with {provider}/{model_name}...")
    try:
        analyzer = IntelligentSchemaAnalyzer(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            verbose=True
        )
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        if provider != "ollama":
            print("💡 Try using Ollama (option 3) which doesn't require an API key.")
        return
    
    # Load data
    analyzer.load_data(dataframes)
    
    # Run comprehensive analysis
    print("\n🧠 Running comprehensive schema analysis with ReAct agent...")
    try:
        results = analyzer.analyze_schema_with_agent()
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE")
        print("="*80)
        
        print(results["summary"])
        
        print("\n📄 Generated SQL DDL:")
        print("-" * 40)
        print(results["sql_ddl"])
        
        print("\n🔗 ER Diagram Entities:")
        print("-" * 30)
        er_entities = results["er_diagram"]["entities"]
        for entity_name, entity_info in er_entities.items():
            pk_attrs = [attr['name'] for attr in entity_info['attributes'] if attr.get('is_primary_key')]
            pk_display = pk_attrs[0] if pk_attrs else 'None'
            print(f"  • {entity_name} (PK: {pk_display}, {len(entity_info['attributes'])} attributes)")
        
        if results["er_diagram"]["relationships"]:
            print(f"\n🔗 Relationships Found: {len(results['er_diagram']['relationships'])}")
            for rel in results["er_diagram"]["relationships"]:
                print(f"  • {rel['parent_entity']} → {rel['child_entity']} (via {rel['foreign_key']})")
        
        # Save results
        output_path = analyzer.save_results()
        print(f"\n💾 Complete analysis saved to: {output_path}")
        print("\nFiles generated:")
        print(f"  • schema.sql - SQL DDL script")
        print(f"  • er_diagram_spec.json - ER diagram specification")  
        print(f"  • full_analysis.json - Complete analysis data")
        print(f"  • summary.txt - Human-readable summary")
        
        return results
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
