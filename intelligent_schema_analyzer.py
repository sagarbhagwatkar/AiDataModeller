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
    abbreviate_columns,
    abbreviate_table_names,
)
import re

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
        max_iterations: int = 25,
        max_execution_time: Optional[int] = 120,
        verbose: bool = True,
    ) -> None:
        """Initializer."""
        self.provider = provider
        self.model_name = model_name
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.analysis_results: Dict[str, Any] = {}
        self.abbrev_map: Dict[str, Dict[str, str]] = {}
        self.table_name_map: Dict[str, str] = {}
        self._tables_abbreviated: bool = False  # internal guard to avoid double remap

        self.llm = self._create_llm(provider, model_name, api_key)
        self.agent_executor = self._create_agent(
            max_iterations, max_execution_time, verbose
        )
        logger.info(
            "Initialized IntelligentSchemaAnalyzer with %s/%s", provider, model_name
        )
    
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
    
    def _create_agent(
        self,
        max_iterations: int,
        max_execution_time: Optional[int],
        verbose: bool,
    ) -> AgentExecutor:
        """Create ReAct agent with data analysis tools."""
        
        # Create wrapper functions for tools
        def analyze_primary_keys_wrapper(input_str: str = "") -> str:
            """Wrapper for analyze_primary_key_candidates function."""
            try:
                result = analyze_primary_key_candidates(self.dataframes)
                return str(result)
            except Exception as e:
                return f"Error analyzing primary keys: {str(e)}"
        def abbreviate_table_names_wrapper(input_str: str = "") -> str:
            """Generate/read table name abbreviations (dictionary only) and optionally remap DataFrames.

            Logic (tool.abbreviate_table_names):
              * Split name into tokens (underscore / camel)
              * For each token: if in internal token_map, use its abbreviation; else keep original token
              * Preserve overall case style: all lowercase input => lowercase output; otherwise UPPER
              * No compression heuristics; strictly dictionary or original token
              * Examples: employees_phones -> empl_ph, Hierarchy_Code -> HIER_CD
              * Uniqueness assured with numeric suffixes

            Optional JSON input: {preview_only: bool, force: bool}
            Returns: {"mapping": {...}, "applied": bool, "reason": str}
            """
            try:
                preview_only = False
                force = False
                if input_str:
                    try:
                        payload = json.loads(input_str)
                        preview_only = bool(payload.get("preview_only"))
                        force = bool(payload.get("force"))
                    except Exception:  # noqa: BLE001
                        pass

                if self._tables_abbreviated and not force:
                    return json.dumps({
                        "mapping": self.table_name_map,
                        "applied": False,
                        "reason": "Already abbreviated; use force=true to recompute",
                    })

                mapping = abbreviate_table_names(self.dataframes.keys())

                if not preview_only:
                    remapped: Dict[str, pd.DataFrame] = {}
                    for orig, df in self.dataframes.items():
                        new_name = mapping.get(orig, orig)
                        if new_name in remapped and new_name != orig:
                            suffix = 2
                            candidate = f"{new_name}_{suffix}"
                            while candidate in remapped:
                                suffix += 1
                                candidate = f"{new_name}_{suffix}"
                            new_name = candidate
                        remapped[new_name] = df
                    self.table_name_map = mapping
                    self.dataframes = remapped
                    self._tables_abbreviated = True
                    return json.dumps({
                        "mapping": mapping,
                        "applied": True,
                        "reason": "Applied and internal dataframes renamed",
                    })
                return json.dumps({
                    "mapping": mapping,
                    "applied": False,
                    "reason": "Preview only",
                })
            except Exception as e:  # noqa: BLE001
                return json.dumps({"error": f"Error abbreviating table names: {e}"})
        
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

        def abbreviate_columns_wrapper(input_str: str = "") -> str:
            """Generate abbreviated column names per table and store in analyzer.

            Optional JSON input example:
              {"custom_rules": {"description": "desc"}, "max_token_length": 10, "tables": ["orders"]}
            Returns JSON: {table: {original: abbreviated}}
            """
            try:
                custom_rules = None
                max_token_length = 12
                table_filter = None
                if input_str:
                    try:
                        payload = json.loads(input_str)
                        custom_rules = payload.get("custom_rules")
                        max_token_length = payload.get("max_token_length", 12)
                        if payload.get("tables"):
                            table_filter = set(payload["tables"])  # type: ignore[arg-type]
                    except Exception:
                        pass
                mapping: Dict[str, Dict[str, str]] = {}
                for tname, df in self.dataframes.items():
                    if df is None:
                        continue
                    if table_filter and tname not in table_filter:
                        continue
                    mapping[tname] = abbreviate_columns(df.columns, custom_rules=custom_rules, max_token_length=max_token_length)
                # Persist mapping
                for t, m in mapping.items():
                    self.abbrev_map.setdefault(t, {}).update(m)
                return json.dumps(mapping)
            except Exception as e:  # noqa: BLE001
                return f"Error abbreviating columns: {e}"
        
        # Create tool instances that work with the stored dataframes
        tools = [
            Tool(
                name="abbreviate_table_names",
                description=(
                    "Create abbreviated table names (dictionary only, no compression). Call FIRST. "
                    "Examples: employees_phones->empl_ph, Hierarchy_Code->HIER_CD. Optional JSON: {preview_only, force}. "
                    "Returns JSON {mapping, applied, reason}."
                ),
                func=abbreviate_table_names_wrapper,
            ),
            Tool(
                name="analyze_primary_keys",
                description=(
                    "Analyze potential primary key candidates in all "
                    "loaded DataFrames. Returns detailed analysis of "
                    "uniqueness, null values, and data types for each "
                    "column."
                ),
                func=analyze_primary_keys_wrapper,
            ),
            Tool(
                name="abbreviate_columns",
                description=(
                    "Create standardized short forms for column names across tables. "
                    "Call this early; subsequent outputs should use the abbreviated names. "
                    "Optional JSON input with custom_rules, max_token_length, tables."
                ),
                func=abbreviate_columns_wrapper,
            ),
            Tool(
                name="find_composite_keys",
                description=(
                    "Find potential composite key combinations in all "
                    "loaded DataFrames. Identifies columns that together "
                    "could form a unique identifier."
                ),
                func=find_composite_keys_wrapper,
            ),
            Tool(
                name="find_relationships",
                description=(
                    "Analyze relationships between DataFrames by finding "
                    "foreign key connections. Identifies how tables relate "
                    "to each other."
                ),
                func=find_relationships_wrapper,
            ),
            Tool(
                name="get_data_summary",
                description=(
                    "Get a summary of all loaded DataFrames including "
                    "column names, data types, and basic statistics."
                ),
                func=self._get_data_summary,
            ),
        ]
        tool_names = [
            "abbreviate_table_names",
            "analyze_primary_keys",
            "abbreviate_columns",
            "find_composite_keys",
            "find_relationships",
            "get_data_summary",
        ]

        # Create the ReAct prompt template
        prompt_lines = [
            "You are an expert data analyst and database designer.",
            "Your job is to analyze data schemas (CSV/JSON/Excel) and create comprehensive database designs.",
            "First call abbreviate_table_names to shorten all table names.",
            "Then call abbreviate_columns to establish shortened column names (with semantic suffixes).",
            "After both mappings are established, ONLY use the abbreviated table and column names in every output (SQL DDL, ER spec, reasoning).",
            "  • Phone / Telephone / Mobile columns: ensure suffix _no (e.g. Phone -> ph_no, mobile_number -> mob_no)",
            "  • Name / descriptive identity columns: ensure suffix _nm (e.g. Country -> cntry_nm, CustomerName -> cust_nm)",
            "  • If abbreviation already ends with the suffix, don't duplicate it.",
            "  • Only add a suffix when it adds clarity (avoid for pure IDs, numeric metrics, dates).",
            "After suffix adjustment use ONLY the final names in all outputs (SQL DDL, foreign keys, ER spec).",
            "Aim to finish within 4 actions. If you have enough",
            "information, provide the Final Answer immediately.",
            "",
            "You have access to the following tools:",
            "{tools}",
            "",
            "Use the following format:",
            "",
            "Question: the input question you must answer",
            "Thought: you should always think about what you need to do",
            "Action: the action to take, should be one of [{tool_names}]",
            "Action Input: the input to the action (use empty string if no",
            "input needed)",
            "Observation: the result of the action",
            "... (this Thought/Action/Action",
            "Input/Observation can repeat N times)",
            "Thought: I now have enough information to provide a",
            "comprehensive analysis",
            "Final Answer: Provide outputs in this exact structure:",
            "",
            "<ANALYSIS_SUMMARY>",
            "A concise summary of findings (note that abbreviated column names were applied).",
            "</ANALYSIS_SUMMARY>",
            "",
            "<SQL_DDL>",
            "```sql",
            "-- DDL starts here",
            "-- One or more CREATE TABLE statements with constraints",
            "-- Optional ALTER TABLE statements for FKs",
            "```",
            "</SQL_DDL>",
            "",
            "<ER_DIAGRAM_SPEC_JSON>",
            "```json",
            "Output a valid JSON object for the ER diagram specification.",
            "It MUST include top-level keys: entities, relationships,",
            "diagram_layout, title, and may include notes.",
            "For each entity, include attributes (name, type,",
            "is_primary_key, is_nullable, is_unique) and row_count.",
            "For each relationship, include parent_entity, child_entity,",
            "foreign_key, relationship_type, and cardinality.",
            "Do not include comments in the JSON.",
            "```",
            "</ER_DIAGRAM_SPEC_JSON>",
            "",
            "Begin!",
            "",
            "Question: {input}",
            "Thought: I need to systematically analyze the data",
            "to understand its structure and relationships.",
            "{agent_scratchpad}",
        ]
        prompt = PromptTemplate.from_template("\n".join(prompt_lines))
        
        # Create the ReAct agent
        agent = create_react_agent(self.llm, tools, prompt)
        
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=verbose,
            max_iterations=max_iterations,
            max_execution_time=max_execution_time,
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
        # Build table name abbreviation map (stable once per load)
        try:
            self.table_name_map = abbreviate_table_names(self.dataframes.keys())
        except Exception:
            self.table_name_map = {k: k for k in self.dataframes.keys()}
        # Physically remap internal dataframe dict so downstream logic naturally uses abbreviated names
        remapped: Dict[str, pd.DataFrame] = {}
        for orig, df in self.dataframes.items():
            new_name = self.table_name_map.get(orig, orig)
            # Avoid accidental overwrite if collision after abbreviation
            if new_name in remapped and new_name != orig:
                # Fallback to original name with suffix to preserve both
                alt_name = f"{new_name}_tbl{len(remapped)}"
                remapped[alt_name] = df
            else:
                remapped[new_name] = df
        self.dataframes = remapped
        self._tables_abbreviated = True  # mark as already abbreviated for wrapper idempotency
        logger.info(
            "Loaded %d DataFrames: %s",
            len(dataframes),
            list(dataframes.keys()),
        )
    
    def analyze_schema_with_agent(self) -> Dict[str, Any]:
        """
        Use the ReAct agent to comprehensively analyze the schema.
        
        Returns:
            Dict containing complete analysis, SQL DDL, and ER diagram spec
        """
        if not self.dataframes:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Starting schema analysis with ReAct agent...")
        
        query_lines = [
            "Please analyze the loaded data comprehensively by",
            "following these steps:",
            "",
            "1. First, get a summary of all the data to understand",
            "what we're working with",
            "2. Analyze primary key candidates for each table",
            "3. Find any composite key opportunities",
            "4. Identify relationships between tables",
            "5. Based on your analysis, generate:",
            "   a) A comprehensive SQL DDL script with proper",
            "constraints, indexes, and foreign keys",
            "   b) A detailed ER diagram specification that shows",
            "all entities, attributes, and relationships",
            "",
            "Provide a thorough analysis that demonstrates your",
            "understanding of the data structure and relationships.",
        ]
        query = "\n".join(query_lines)
        
        try:
            result = self.agent_executor.invoke({"input": query})
            final_output = result.get("output", "") or ""
            
            # Store the analysis results
            self.analysis_results = {
                "agent_output": final_output,
                "intermediate_steps": result.get("intermediate_steps", []),
                "reasoning_chain": self._extract_reasoning_chain(result),
            }
            
            # Prefer agent-generated artifacts if available
            ddl_from_agent, er_from_agent = self._parse_agent_outputs(
                final_output
            )
            
            # Fallbacks using tools if parsing failed
            ddl_script = ddl_from_agent or self._generate_sql_ddl()
            er_diagram_spec = er_from_agent or self._generate_er_diagram_spec()
            
            # Normalize ER spec shape for UI consumption
            er_diagram_spec = self._normalize_er_spec(er_diagram_spec)
            
            return {
                "analysis": self.analysis_results,
                "sql_ddl": ddl_script,
                "er_diagram": er_diagram_spec,
                "summary": self._generate_analysis_summary(),
                "table_name_map": self.table_name_map,
            }
            
        except Exception as e:
            logger.error(f"Schema analysis failed: {e}")
            raise

    def _parse_agent_outputs(
        self, text: str
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Parse agent final output to extract SQL DDL and ER diagram JSON.
        Returns (ddl_sql, er_spec_dict); either can be None if not found.
        """
        ddl_sql: Optional[str] = None
        er_spec: Optional[Dict[str, Any]] = None

        # Extract SQL
        try:
            m = re.search(r"```sql(.*?)```", text, re.IGNORECASE | re.DOTALL)
            if m:
                ddl_sql = m.group(1).strip()
            else:
                # Heuristic: capture from first CREATE TABLE to before
                # ER JSON or the next fenced block
                m2 = re.search(r"CREATE\s+TABLE[\s\S]+", text, re.IGNORECASE)
                if m2:
                    start = m2.start()
                    stop_markers = [
                        "\n```",  # next fenced block
                        "\n<ER_DIAGRAM_SPEC_JSON>",
                        "\n```json",
                        "\n</SQL_DDL>",
                        "\n<ER_",
                    ]
                    ends = [text.find(mk, start + 1) for mk in stop_markers]
                    ends = [p for p in ends if p != -1]
                    end = min(ends) if ends else len(text)
                    ddl_sql = text[start:end].strip()
                    # Remove any accidental trailing fence start
                    if "\n```" in ddl_sql:
                        ddl_sql = ddl_sql.split("\n```", 1)[0].strip()
        except Exception:
            pass

        # Extract ER JSON
        try:
            j = re.search(r"```json(.*?)```", text, re.IGNORECASE | re.DOTALL)
            if j:
                snippet = j.group(1).strip()
                er_spec = json.loads(snippet)
            else:
                brace_start = text.find("{")
                brace_end = text.rfind("}")
                if (
                    brace_start != -1 and brace_end != -1 and
                    brace_end > brace_start
                ):
                    candidate = text[brace_start: brace_end + 1]
                    try:
                        er_spec = json.loads(candidate)
                    except Exception:
                        er_spec = None
        except Exception:
            er_spec = None

        return ddl_sql, er_spec
    
    def _normalize_er_spec(self, er: Any) -> Dict[str, Any]:
        """Ensure ER spec has a consistent shape.
        - entities: dict[str, {attributes: list[dict], row_count: int}]
        - relationships: list[dict]
        """
        if not isinstance(er, dict):
            return {"entities": {}, "relationships": []}
        
        out: Dict[str, Any] = dict(er)
        entities = out.get("entities", {})

        def _normalize_attrs(attrs: Any) -> List[Dict[str, Any]]:
            res: List[Dict[str, Any]] = []
            if isinstance(attrs, dict):
                for k, v in attrs.items():
                    if isinstance(v, dict):
                        item = {"name": k, **v}
                    else:
                        item = {"name": k, "type": str(v)}
                    item.setdefault("is_primary_key", False)
                    item.setdefault("is_nullable", True)
                    item.setdefault("is_unique", False)
                    res.append(item)
            elif isinstance(attrs, list):
                for a in attrs:
                    if isinstance(a, dict):
                        a.setdefault("name", a.get("column", ""))
                        a.setdefault("type", a.get("dtype", "TEXT"))
                        a.setdefault("is_primary_key", False)
                        a.setdefault("is_nullable", True)
                        a.setdefault("is_unique", False)
                        res.append(a)
                    else:
                        res.append({
                            "name": str(a),
                            "type": "TEXT",
                            "is_primary_key": False,
                            "is_nullable": True,
                            "is_unique": False,
                        })
            else:
                # Unknown, return empty
                pass
            return res

        # Normalize entities to dict
        if isinstance(entities, list):
            new_entities: Dict[str, Any] = {}
            for i, e in enumerate(entities):
                if not isinstance(e, dict):
                    name = f"entity_{i+1}"
                    new_entities[name] = {"attributes": [], "row_count": 0}
                    continue
                name = (
                    e.get("name")
                    or e.get("entity")
                    or e.get("table")
                    or e.get("table_name")
                    or f"entity_{i+1}"
                )
                attrs = _normalize_attrs(e.get("attributes", []))
                row_count = int(e.get("row_count", 0) or 0)
                new_entities[name] = {
                    "attributes": attrs,
                    "row_count": row_count,
                }
            out["entities"] = new_entities
        elif isinstance(entities, dict):
            for k, info in list(entities.items()):
                if not isinstance(info, dict):
                    entities[k] = {"attributes": [], "row_count": 0}
                    continue
                attrs = _normalize_attrs(info.get("attributes", []))
                info["attributes"] = attrs
                info["row_count"] = int(info.get("row_count", 0) or 0)
            out["entities"] = entities
        else:
            out["entities"] = {}

        # Normalize relationships to list of dicts
        rels = out.get("relationships", [])
        norm_rels: List[Dict[str, Any]] = []
        if isinstance(rels, dict):
            # Possibly keyed by "parent-child" or similar
            for key, rel_list in rels.items():
                if isinstance(rel_list, list):
                    for item in rel_list:
                        if isinstance(item, dict):
                            parent, child = None, None
                            if isinstance(key, str) and "-" in key:
                                parent, child = key.split("-", 1)
                            norm_rels.append({
                                "parent_entity": item.get(
                                    "parent_entity", parent
                                ),
                                "child_entity": item.get(
                                    "child_entity", child
                                ),
                                "foreign_key": item.get("column")
                                or item.get("foreign_key", ""),
                                "relationship_type": item.get(
                                    "relationship_type", "one-to-many"
                                ),
                                "cardinality": item.get(
                                    "cardinality", "unknown"
                                ),
                            })
        elif isinstance(rels, list):
            for item in rels:
                if isinstance(item, dict):
                    norm_rels.append({
                        "parent_entity": item.get("parent_entity"),
                        "child_entity": item.get("child_entity"),
                        "foreign_key": item.get("foreign_key", ""),
                        "relationship_type": item.get(
                            "relationship_type", "one-to-many"
                        ),
                        "cardinality": item.get("cardinality", "unknown"),
                    })
        out["relationships"] = norm_rels

        return out
    
    def _extract_reasoning_chain(self, result: Dict) -> List[Dict]:
        """Extract the agent's reasoning chain from the result."""
        chain = []
        if "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                if len(step) >= 2:
                    action, observation = step[0], step[1]
                    chain.append({
                        "action": (
                            action.tool if hasattr(action, 'tool')
                            else str(action)
                        ),
                        "observation": (
                            str(observation)[:500] + "..."
                            if len(str(observation)) > 500
                            else str(observation)
                        ),
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
                new_table_name = self.table_name_map.get(table_name, table_name)
                if new_table_name != table_name:
                    ddl_parts.append(f"-- Table: {table_name} -> {new_table_name}")
                else:
                    ddl_parts.append(f"-- Table: {table_name}")
                ddl_parts.append(f"CREATE TABLE {new_table_name} (")
                
                columns = []
                table_abbrev = self.abbrev_map.get(table_name, {})
                for col in df.columns:
                    abbr_col = table_abbrev.get(col, col)
                    col_type = self._infer_sql_type(df[col])
                    
                    # Check if this is a primary key candidate
                    pk_info = primary_keys.get(table_name, {}).get(col, {})
                    is_pk = (
                        pk_info.get('is_unique', False)
                        and not pk_info.get('has_nulls', True)
                    )
                    
                    constraint = ""
                    if is_pk:
                        constraint = " PRIMARY KEY"
                    elif not pk_info.get('has_nulls', True):
                        constraint = " NOT NULL"
                    
                    columns.append(f"    {abbr_col} {col_type}{constraint}")
                
                ddl_parts.append(",\n".join(columns))
                ddl_parts.append(");\n")
            
            # Add foreign key constraints (supports tuple or string keys, JSON parent/child pattern)
            ddl_parts.append("-- Foreign Key Constraints")
            for rel_key, relations_list in relationships.items():
                if isinstance(rel_key, tuple) and len(rel_key) == 2:
                    parent_table, child_table = rel_key
                elif isinstance(rel_key, str) and '-' in rel_key:
                    parent_table, child_table = rel_key.split('-', 1)
                else:
                    continue
                for relation in relations_list:
                    child_col = relation.get('child_column') or relation.get('column')
                    parent_col = relation.get('parent_column') or relation.get('column')
                    if not child_col or not parent_col:
                        continue
                    # Only create FK if columns exist in respective tables
                    child_df = self.dataframes.get(child_table)
                    parent_df = self.dataframes.get(parent_table)
                    if (child_df is None or parent_df is None or
                            child_col not in child_df.columns or
                            parent_col not in parent_df.columns):
                        continue
                    # Map to abbreviated names
                    child_abbr = self.abbrev_map.get(child_table, {}).get(child_col, child_col)
                    parent_abbr = self.abbrev_map.get(parent_table, {}).get(parent_col, parent_col)
                    parent_table_abbr = self.table_name_map.get(parent_table, parent_table)
                    child_table_abbr = self.table_name_map.get(child_table, child_table)
                    ddl_parts.append(
                        (
                            f"ALTER TABLE {child_table_abbr} ADD CONSTRAINT "
                            f"fk_{child_table_abbr}_{child_abbr} "
                            f"FOREIGN KEY ({child_abbr}) REFERENCES "
                            f"{parent_table_abbr}({parent_abbr});"
                        )
                    )
            
            # Add indexes for performance
            ddl_parts.append("\n-- Performance Indexes")
            for table_name in self.dataframes.keys():
                if self.dataframes[table_name] is not None:
                    table_abbrev = self.abbrev_map.get(table_name, {})
                    new_table_name = self.table_name_map.get(table_name, table_name)
                    for col in self.dataframes[table_name].columns:
                        abbr_col = table_abbrev.get(col, col)
                        if abbr_col.endswith('_id') or 'id' in abbr_col.lower():
                            ddl_parts.append(
                                (
                                    f"CREATE INDEX idx_{new_table_name}_{abbr_col} "
                                    f"ON {new_table_name}({abbr_col});"
                                )
                            )
            
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
                new_table_name = self.table_name_map.get(table_name, table_name)
                attributes = []
                table_abbrev = self.abbrev_map.get(table_name, {})
                for col in df.columns:
                    abbr_col = table_abbrev.get(col, col)
                    pk_info = primary_keys.get(table_name, {}).get(col, {})
                    is_pk = pk_info.get('is_unique', False) and not pk_info.get('has_nulls', True)
                    
                    attributes.append({
                        'name': abbr_col,
                        'type': str(df[col].dtype),
                        'is_primary_key': is_pk,
                        'is_nullable': pk_info.get('has_nulls', True),
                        'is_unique': pk_info.get('is_unique', False)
                    })
                
                entities[new_table_name] = {
                    'attributes': attributes,
                    'row_count': len(df)
                }
            
            # Build relationships
            relationships_spec = []
            for rel_key, relations_list in relationships.items():
                if isinstance(rel_key, tuple) and len(rel_key) == 2:
                    parent_table, child_table = rel_key
                elif isinstance(rel_key, str) and '-' in rel_key:
                    parent_table, child_table = rel_key.split('-', 1)
                else:
                    continue
                for relation in relations_list:
                    fk_child_orig = relation.get('child_column') or relation.get('column')
                    fk_child = self.abbrev_map.get(child_table, {}).get(fk_child_orig, fk_child_orig)
                    parent_entity = self.table_name_map.get(parent_table, parent_table)
                    child_entity = self.table_name_map.get(child_table, child_table)
                    relationships_spec.append({
                        'parent_entity': parent_entity,
                        'child_entity': child_entity,
                        'foreign_key': fk_child,
                        'relationship_type': relation.get('relationship_type', 'one-to-many'),
                        'cardinality': relation.get('cardinality', 'unknown'),
                    })
            
            return {
                'entities': entities,
                'relationships': relationships_spec,
                'diagram_layout': 'vertical',
                'title': 'Database Entity-Relationship Diagram',
                'notes': (
                    'Generated by Intelligent Schema Analyzer using ReAct Agent. '
                    'Table abbreviations applied. Original to abbreviated mapping: '
                    f"{self.table_name_map}"
                ),
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
                    summary.append(
                        (
                            f"  • {name}: {len(df)} rows, "
                            f"{len(df.columns)} columns"
                        )
                    )
            summary.append("")
        
        # Add reasoning chain if available
        if (
            hasattr(self, 'analysis_results')
            and 'reasoning_chain' in self.analysis_results
        ):
            summary.append("🧠 Agent Reasoning Process:")
            for i, step in enumerate(
                self.analysis_results['reasoning_chain'], 1
            ):
                summary.append(
                    f"  {i}. Used tool '{step['action']}'"
                )
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