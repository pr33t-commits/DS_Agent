import json
import threading
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from .schemas import UnderstandingReport
# Compatibility exports for callers using the original module.
from inference.chat_models import local_model, make_model
from evaluation.evidence import validate_evidence


import operator
# from typing import Annotated, List, Literal, Union, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode


import code
import json
import os
import subprocess
import tempfile
import uuid
import ast
import logging
from typing import Any, Dict, List, Optional, Annotated
from typing_extensions import Annotated, List, TypedDict, runtime
from dataclasses import dataclass, field
from datetime import datetime
from langgraph.prebuilt import InjectedState
import pandas as pd
import numpy as np
import multiprocessing as mp
import traceback

from CodeExecutor import run_generated_code_in_subprocess

SYSTEM_PROMPT = """You are a dataset understanding analyst. 

GENERAL INSTRUCTIONS:-
1. All numbers and factual answers must come from data present in the df_store. Do not hallucinate.
2. STRICTLY follow USAGE INSTRUCTIONS of all tools if available.
3. Make a rough plan before starting execution.
4. Generate code, execute and extract insights from the dataframes provided in df_store.

GOAL:-
1. Logical verification of the data.
2. Relevant analysis for demand forecast modeling downstream.
3. Framing doubts and clarifying questions based on analysis.
"""

EXECUTOR_TIMEOUT = 30  # seconds
MAX_ANALYST_ITERATIONS = 10

def dict_merge(old: Dict, new: Dict) -> Dict:
    """Merge two dicts, with new keys overriding old ones"""
    return {**old, **new}

def df_summary(df: pd.DataFrame) -> str:
    """Generate summary of DataFrame"""
    dataset_summary = f"DATASET: Results ({df.shape[0]} rows × {df.shape[1]} columns) Columns Names: {', '.join(df.columns.tolist())}\n\n"
    return dataset_summary

class DataFrameInfo(TypedDict):
    """Schema for individual DataFrame store entries"""
    # DataFrame: pd.DataFrame
    Summary: str
    Description: Optional[str]

class WorkerState(TypedDict):
    # 'add_messages' ensures we append history, not overwrite it
    messages: Annotated[List[BaseMessage], operator.add]
    # current_task: str
    dataframe_info: Annotated[Dict[str, DataFrameInfo], dict_merge]
    # generated_code: str 
    coding_error: str #Annotated[Dict[str,str], dict_merge]
    coding_error_traceback: str
    iteration_count: int



def build_agent(executor, run_dir: Path, model=None, max_calls=12):
    lock = threading.Lock()
    evidence = {}
    counter = 0

    @tool
    def python_executor(code: str) -> str:
        """Execute Python against the two CSV inputs; print compact evidence. State is not persistent."""
        nonlocal counter
        with lock:
            if counter >= max_calls:
                return json.dumps({"ok": False, "error": "Execution budget exhausted. Report limitations."})
            counter += 1
            evidence_id = f"python_{counter:03d}"
            (run_dir / f"{evidence_id}.py").write_text(code, encoding="utf-8")
            result = executor.run(code)
            result["evidence_id"] = evidence_id
            evidence[evidence_id] = result
            (run_dir / f"{evidence_id}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            return json.dumps(result)

    agent = create_agent(model=model, tools=[python_executor], system_prompt=SYSTEM_PROMPT,
                         response_format=ToolStrategy(UnderstandingReport))
    
    return agent, evidence

def _tool_name(fn) -> str:
	return getattr(fn, "__name__", getattr(fn, "name", fn.__class__.__name__))

class SingleAgentAnalysisSystem:
    """High-level interface for multi-agent analysis"""
    
    def __init__(self, dataframes: Optional[Dict[str, Dict[str, Any]]] = None,
              model: str = None, llm=None):
        """
        Args:
            dataframes: Dict mapping df_id to dict with keys:
                - 'DataFrame': pd.DataFrame (required)
                - 'Description': str (optional)
                Example: {
                    'sales_data': {
                        'DataFrame': df1,
                        'Description': 'Historical sales records'
                    },
                    'forecast_results': {
                        'DataFrame': df2,
                        'Description': 'Model forecast outputs'
                    }
                }
            model: LLM model to use
            llm: Optional preconfigured LangChain chat model to reuse.
        """
        self.input_dataframes = dataframes
        self.df_store = {key:value['DataFrame'] for key,value in self.input_dataframes.items()} if len(self.input_dataframes) > 0 else {}
        self.model = model
        # Use the LangChain adapter rather than raw Transformers weights.  The
        # adapter implements bind_tools() and loads the underlying model lazily
        # on its first invocation, so one instance can be reused by every graph
        # iteration.
        self.llm = llm if llm is not None else make_model(self.model, provider="transformer")
        self.data_analyst_tools = self._setup_data_analyst_tools()
        self.graph = self._build_graph()
        self.result_store = {}
        # self.intersection_level = intersection_level
        # self.objective = objective

    def make_dataframe_info(self) -> Dict[str, DataFrameInfo]:
        """Generate dataframe_info dict from df_store"""
        dataframe_info: Dict[str, DataFrameInfo] = {}
        for df_id, df_info in self.input_dataframes.items():
            dataframe_info[df_id] = {
                "Summary": df_summary(df_info['DataFrame']),
                "Description": df_info.get("Description",f"Input DataFrame: {df_id}")}
        return dataframe_info
    
    def _setup_data_analyst_tools(self):
        """Setup tools with ToolRuntime for state access."""
        @tool
        def list_dfs_tool(runtime: ToolRuntime = None) -> Command:
            """From the df_store, returns list of available DataFrame ids, the shape and columns of each DataFrame and descriptions (what does the data represent)."""
            return self.list_dfs_fn(runtime)

        @tool
        def code_executor_fn(
                code,
                df_ids: List[str],
                timeout: Optional[int] = EXECUTOR_TIMEOUT, 
                runtime: ToolRuntime = None
                ) -> Dict[str, Any]:
            """Executes the generated code on specified DataFrames.
            
            DataFrames are passed through a single variable:

			df : Dict[str, pandas.DataFrame]

	    	Keys are DataFrame IDs (strings). Values are pandas DataFrames.
		    In the code, ALWAYS access DataFrames using:
			    df = df["<df_id>"]
   
            Args:
                code: The code which is to be executed
                df_ids: The IDs of the DataFrames to be used
            """
            print(f"\n🛠️  CODE EXECUTOR INVOKED with code :\n{code}")
            state = runtime.state
            # code = state.get("generated_code")

            if not code:
                print("No generated code found in state.")
                return {"messages": ToolMessage(content="No code available. Generate code first.", tool_call_id=runtime.tool_call_id)}
            
            try:
                if not df_ids:
                    return {"messages": ToolMessage(content="Specify df_id or df_ids", tool_call_id=runtime.tool_call_id)}
                
                # Collect DataFrames
                df = {}
                missing = []
                for df_id in df_ids:
                    try:
                        df[df_id] = self.df_store[df_id]
                    except KeyError:
                        missing.append(df_id)
                
                if missing:
                    print(f"❌ DataFrames not found: {missing}")
                    return {"messages": ToolMessage(content=f"df_ids not found: {missing}", tool_call_id=runtime.tool_call_id)}
                
                # Execute code
                res = run_generated_code_in_subprocess(code, df, timeout=timeout)

                if res.get("status") == "error":
                    error_msg = res.get("error")
                    print("Code Execution Traceback/Error:\n", res.get("traceback",error_msg))
                    return {
                        "messages": ToolMessage(content=f"Code execution failed with message:- {error_msg}", tool_call_id=runtime.tool_call_id),
                        "coding_error": error_msg,
                        "coding_error_traceback": str(res.get("traceback", ""))
                    }
                
                elif res.get("status") == "ok_df_generated":
                    result_dict = res.get("result")
                    new_df_id = result_dict.get("df_id") or str(uuid.uuid4())
                    description = result_dict.get("Description", "Generated dataframe")
                    summary = df_summary(result_dict['DataFrame'])
                    # Store DataFrame
                    self.df_store[new_df_id] = result_dict['DataFrame']
                    print(f"✅ Created dataframe with id: {new_df_id} with Summary: {summary} and Description: {description}")
                    
                    return Command(update={
                        "messages": ToolMessage(content=f"Created dataframe with id: {new_df_id} with Summary: {summary}", tool_call_id=runtime.tool_call_id),
                        "dataframe_info": {
                            new_df_id: {"Summary": summary, "Description": description}
                        },
                        "coding_error": "",
                        "coding_error_traceback": ""
                    })
                
                else:
                    result_text = str(res.get("result", ""))
                    print("Code Execution Result:\n", result_text[:500])
                    return Command(update={"messages": ToolMessage(content = f"Code execution result: {result_text}",
                                                                   tool_call_id=getattr(runtime, "tool_call_id", None)), 
                            "coding_error": "", "coding_error_traceback": ""})
            
            except Exception as e:
                tb = traceback.format_exc()
                print(f"Exception during code execution: {str(e)}")
                return Command(update={"messages": ToolMessage(content=f"Exception during code execution: {str(e)}", tool_call_id=getattr(runtime, "tool_call_id", None)), 
                                       "coding_error": str(e), "coding_error_traceback": tb})

        # Return tools: list_dfs_tool, calculate_accuracy_fn, combined code_run_tool_fn
        return [list_dfs_tool, code_executor_fn]

    def list_dfs_fn(self, runtime: ToolRuntime) -> Command:
        """From the df_store, returns list of available DataFrame ids, their summaries and descriptions."""
        
        df_info = runtime.state.get("dataframe_info")
        if df_info is None:
            return Command(update={
                "messages": [ToolMessage(
                    content="No dataframe_info in state",
                    tool_call_id=runtime.tool_call_id
                )]
            })
        
        print("Available dataframes:", df_info.keys())
        return Command(update={
            "messages": [ToolMessage(
                content=json.dumps(df_info, indent=2),
                tool_call_id=runtime.tool_call_id
            )]
        })
    
    # def code_generator_fn(
    # 	self,
    # 	goal: str,
    # 	df_ids: List[str],
    # 	store_df: bool,
    # 	runtime: ToolRuntime
    # ) -> Dict[str, Any]:
    
    # 	"""Generate code using LLM."""
    # 	print(f"\n📝 CODE GENERATOR INVOKED")
        
        
    # 	state = runtime.state
    # 	dep_versions = get_dependency_versions()
    # 	code_prompt = """
    # 	You are a Python coding assistant. Produce ONLY Python code (no explanation).

    # 	You will be provided required DataFrames through a single variable:

    # 		dfs : Dict[str, pandas.DataFrame]

    # 	Keys are DataFrame IDs (strings). Values are pandas DataFrames.
    # 	ALWAYS access DataFrames using:
    # 		df = dfs["<df_id>"]

    # 	Your Goal: {goal}

    # 	Available DataFrames' IDs: {available_ids}
    # 	Summary of available DataFrames (ID -> Summary): {input_df_summary}
    # 	Must generate output as DataFrame: {store_output_df}

    # 	OUTPUT CONTRACT:
    # 	1. The final output must always be stored in a variable named 'result'.
    # 	2. If store_output_df is False:
    # 		- result must be a string that directly answers the goal.
    # 		- Any intermediate calculations must be converted to clear text in result.
    # 		- NEVER print or return dataframes entirely; summarize or extract insights and include in result. 
    # 	3. If store_output_df is True:
    # 		- result must be a dict with EXACTLY these keys:
    # 			a. DataFrame (pd.DataFrame): The output DataFrame
    # 			b. df_id (str, optional): Contextual id for the resulting DataFrame
    # 			c. Description (str, optional): Description of what the contained data represents
    # 		- No other formats allowed when store_output_df is True.
    # 	4. Include metric values in final output where applicable.

    # 	INSTRUCTIONS:-
    # 		1. Never assume any variable named 'df', 'df_<id>', or others.
    # 		2. NEVER import anything except pandas as pd and numpy as np (already provided).
    # 	"""
        
    # 	code_prompt += f""" Never generate a new dataframe with same ID as input dataframes':- {list(self.input_dataframes.keys())}.

    # 	ENVIRONMENT INFO:
    # 	- pandas version: {dep_versions.get('pandas', 'unknown')}
    # 	- numpy version: {dep_versions.get('numpy', 'unknown')}
    # 	- Python 3.8+"""
  
    # 	df_info = state.get("dataframe_info", {})
    # 	summaries = {}
    # 	missing = []
        
    # 	for df_id in df_ids:
    # 		try:
    # 			summaries[df_id] = df_info.get(df_id, {}).get('Summary', '')
    # 		except Exception:
    # 			missing.append(df_id)
        
    # 	if missing:
    # 		print(f"df_ids not found: {missing}")
    # 		return {
    # 			"messages": [ToolMessage(
    # 				content=f"df_ids not found: {missing}",
    # 				tool_call_id=runtime.tool_call_id
    # 			)]
    # 		}
    # 	input_df_summary = "\n".join([f"{df_id} : {summaries[df_id]}" for df_id in df_ids])

    # 	# Include error context from previous attempt if available
    # 	if state.get("coding_error"):
            
    # 		code_prompt += (
    # 			f"\n\nThe previous attempt for the code: \n"
    # 			f"{state.get('generated_code')}\n\n"
    #    			f"failed with the following error:\n"
    # 			f"{state.get('coding_error')}\n"
    # 			f"Traceback: {state.get('coding_error_traceback')}\n"
    # 			f"Please fix the bug and regenerate the code defensively."
    # 		)
    # 	_template_vars = ["goal", "available_ids", "input_df_summary", "store_output_df"]
    # 	safe_prompt = code_prompt.replace("{", "{{").replace("}", "}}")
    # 	for v in _template_vars:
    # 		safe_prompt = safe_prompt.replace("{{" + v + "}}", "{" + v + "}")
   
    # 	code_prompt_template = PromptTemplate(
    # 		input_variables=_template_vars,
    # 		template=safe_prompt
    # 	)

    # 	code_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.01)
    # 	code_chain = code_prompt_template | code_llm 

    # 	# Generate code
    # 	code_response = code_chain.invoke({
    # 		"goal": goal,
    # 		"available_ids": ", ".join(df_ids),
    # 		"input_df_summary": input_df_summary,
    # 		"store_output_df": store_df
    # 	})

    # 	print("Generated Code:\n", code_response.content)
        
    # 	# Return a plain dict (no Command)
    # 	return {
    # 		# "messages": [ToolMessage(
    # 		# 	content="Code generated and stored in state. Ready for execution",
    # 		# 	tool_call_id=runtime.tool_call_id
    # 		# )],
    # 		"messages": "Code generated and stored in state. Ready for execution",
    # 		"generated_code": code_response.content,
    # 		"coding_error": ""
    # 	}

    
    @staticmethod
    def _format_tools(tools: List[str]) -> str:
        """Format tools list for prompt"""
        return "\n".join([f"- {tool}" for tool in tools])

    def _build_graph(self):
        def sequential_tool_node(state: WorkerState):
            tool_messages = []
            for tool_call in state["messages"][-1].tool_calls:  # Sequential loop
                tool = tools[tool_call["name"]]
                result = tool.invoke(tool_call["args"])
                tool_messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"]
                ))
            return {"messages": tool_messages}

        def data_analyst_node(state: WorkerState):
            print("\n" + "="*80)
            print("🔍 DATA ANALYST NODE INVOKED")
            # if state['iteration_count'] %  (MAX_ANALYST_ITERATIONS // 2) == 0:	
                # print(f"Current Task: {state['current_task']}")
            print(f"Iteration: {state['iteration_count']}/{MAX_ANALYST_ITERATIONS}")
            print("="*80)
            
            # analyst_prompt = f"""You are a data analysis agent who is supposed to execute tasks given by upstream supervisor. Your output must be useful enough for further reasoning. Use the available tools and 
            #                     dataframes stored in df_store.
            
            #                     GENERAL INSTRUCTIONS:-
            #                     1. Use code_generator_tool_fn only when it is impossible to get anything useful from the output of other tools.
            #                     2. STRICTLY follow USAGE INSTRUCTIONS of all tools if available.
            #                     3. Make a rough plan before starting execution. 
            #                     4. Your primary function - Supervisor delegated Current task. However, think long term based on the Supervisor's objective of :- '{self.objective}'. 
            #                         Eg.:- Store intermediate dataframes in the DataFrame store which will be used repeatdly.

            #                     CRITICAL RULES:
            #                     1. Every numeric or factual answer must come from data or Observation.
            #                     2. Summarize your actions and results but DO NOT suggest next steps in final answer.
            #                     3. NEVER attempt to solve for the Suervisor's objective. That is just additional context for you.

            #                     Include reasoning/thought process behind particular action in your response.
                                
            #                     Current Task: {state['current_task']}

            # """

            analyst_prompt = SYSTEM_PROMPT
            
            # print(f" Last 3 messages in history:")
            # for i, msg in enumerate(state["messages"][-3:]):
            # 	print(f" Message {i}: {type(msg).__name__} - {msg.content[:100]}")

            sys_msg = SystemMessage(content=(analyst_prompt))
            messages = [sys_msg] + state["messages"]
            llm_with_tools = self.llm.bind_tools(self.data_analyst_tools)
            
            # The LLM returns an AIMessage. 
            # If it calls a tool, this message contains `tool_calls`.
            # If it's reasoning, it contains `content`.
            
            if state.get("iteration_count") < MAX_ANALYST_ITERATIONS:
                response = llm_with_tools.invoke(messages)
                print(f"💭 Analyst Response Content:\n{response.content}")
                if getattr(response, "tool_calls", None):
                    print(f"🛠️  Tool Calls Detected: {len(response.tool_calls)}")
                    for tc in response.tool_calls:
                        print(f"   - Tool: {tc['name']}")
                        for arg, value in tc.get('args', {}).items():
                            print(f"      {arg}: {value}")
                    messages_out = [AIMessage(
                                        content=(
                                            f"Reasoning: {response.content},\n"
                                            + "Tool Calls: "
                                            + json.dumps(
                                                [{"Name": tc["name"], "args": tc.get("args", {})} for tc in response.tool_calls],
                                                indent=2,
                                            )
                                        ),
                                        tool_calls=response.tool_calls,
                                    )
                                ]
                else:
                    messages_out = [AIMessage(content=f"Final Answer:\n{response.content}")]
            else:
                messages = messages + [SystemMessage(content="""Maximum analyst iterations reached. Summarize from whatever results 
                                         you have and specify the pending tasks that couldn't be completed. No more tool calls.""")]
                response = llm_with_tools.invoke(messages)
                messages_out = [response]
                print(f"Max Analyst iterations. Analyst Final Response Content:\n{response.content}")
                # response = AIMessage(content="Maximum analyst iterations reached. Ending analysis. Give smaller/simpler tasks")
                
            return {"messages": messages_out, "iteration_count": state["iteration_count"] + 1}

        # Define the Analyst Subgraph
        analyst_builder = StateGraph(WorkerState)
        analyst_builder.add_node("reasoner", data_analyst_node)
        analyst_builder.add_node("tools", ToolNode(self.data_analyst_tools))
        # analyst_builder.add_node("tools", sequential_tool_node)

        analyst_builder.add_edge(START, "reasoner")

        def analyst_router(state):
            # If the last message has tool calls, go to tools
            last_msg = state["messages"][-1]
            if last_msg.tool_calls and state['iteration_count'] <= MAX_ANALYST_ITERATIONS:
                return "tools"
            # Otherwise end
            return END

        analyst_builder.add_conditional_edges("reasoner", analyst_router, {"tools": "tools", END: END})
        analyst_builder.add_edge("tools", "reasoner") # Loop back after tool execution
        data_analyst_graph = analyst_builder.compile()

        # # --- 5. BRIDGE FUNCTION ---
        # def call_data_analyst(state: GlobalState):
        #     # Map Global -> Worker
        #     last_task_list = state["task_lists"][-1] if state.get("task_lists") else []
        #     print(f"📋 Tasks to Execute: {len(last_task_list)} tasks")
            
        #     task_responses = []
        #     df_info = state.get("dataframe_info", {})
            
        #     for i,task in enumerate(last_task_list, 1):
        #         # Map GlobalState -> WorkerState
        #         worker_inputs: WorkerState = {
        #             "messages": [],
        #             "current_task": task,
        #             "dataframe_info": self.make_dataframe_info() if len(df_info) == 0 else df_info,
        #             "generated_code": "",
        #             "coding_error": "",
        #             "coding_error_traceback": "",
        #             "iteration_count": 0
        #         }
    
        #         # Execute Subgraph
        #         final_worker_state = data_analyst_graph.invoke(worker_inputs)
        #         df_info = final_worker_state.get("dataframe_info", {})
        #         # Extract Final Output
        #         final_response = final_worker_state["messages"][-1].content
        #         task_responses.append(f"Task {i} Result: {final_response}")
        #         print(f"   ✓ Task {i} Completed")
                
        #     print(f"\n✅ All {len(last_task_list)} tasks completed")
        #     return {
        #         "DataAnalyst_reports": [task_responses],
        #         "dataframe_info": final_worker_state.get("dataframe_info", {})
        #     }

        
        # # Execute Subgraph
        # final_worker_state = data_analyst_graph.invoke(worker_inputs)

        return data_analyst_graph
    
    def analyze(self, dataframes: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Run analysis on user query
        
        Args:
            query: User query string
            dataframes: Optional dict of {df_id: {'DataFrame': df, 'Description': str}} to override self.input_dataframes
        """
        if dataframes is not None:
            self.input_dataframes = dataframes
        if not self.input_dataframes:
            raise ValueError("No DataFrames provided")
        
        df_info = {}
        
        initial_state: WorkerState = {
                            "messages": [],
                            # "current_task": task,
                            "dataframe_info": self.make_dataframe_info() if len(df_info) == 0 else df_info,
                            # "generated_code": "",
                            "coding_error": "",
                            "coding_error_traceback": "",
                            "iteration_count": 0
                        }
        
        final_state = self.graph.invoke(initial_state)
        return {
            "query_response": final_state.get("query_response"),
            "messages": final_state.get("messages"),
            "DataAnalyst_reports": final_state.get("DataAnalyst_reports"),
            "dataframe_info": final_state.get("dataframe_info"),
            "iteration_count": final_state.get("iteration_count")
        }


