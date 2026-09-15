import json
import uuid
import traceback
import ast
import multiprocessing as mp
from multiprocessing import Queue, Process
from typing import Dict, Any
import time
import logging
from langchain.tools import tool
import pandas as pd, numpy as np
import builtins
import tempfile, os
from queue import Empty

# Configure logging in the main part of your script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
)

DANGEROUS_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "os", "sys", "subprocess",
    "socket", "ftplib", "shutil", "pathlib", "requests", "http", "urllib"
}
DANGEROUS_ATTR_PREFIX = "__"

def ast_safety_check(code: str):
    """
    Basic AST-based checks:
      - reject Import and ImportFrom nodes
      - reject ClassDef, AsyncFunctionDef
      - reject usage of dangerous names (open, exec, eval, __import__, os, sys, subprocess...)
      - reject attr names starting with __ (dunder attribute access)
    Throws ValueError if checks fail.
    """
    tree = ast.parse(code)
    for node in ast.walk(tree):
        # if isinstance(node, (ast.Import, ast.ImportFrom)):
        #     raise ValueError("Imports are not allowed in generated code.")
        # if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef)):
        #     raise ValueError("Defining classes or async functions is not allowed.")
        if isinstance(node, ast.Call):
            # check for direct calls to dangerous names
            func = node.func
            if isinstance(func, ast.Name) and func.id in DANGEROUS_NAMES:
                raise ValueError(f"Use of dangerous function/name '{func.id}' is not allowed.")
            if isinstance(func, ast.Attribute) and getattr(func.attr, "startswith", lambda *_: False)(DANGEROUS_ATTR_PREFIX):
                raise ValueError("Calling dunder attributes is not allowed.")
        if isinstance(node, ast.Attribute):
            if isinstance(node.attr, str) and node.attr.startswith(DANGEROUS_ATTR_PREFIX):
                raise ValueError("Dunder attribute access is not allowed.")
        if isinstance(node, ast.Name) and node.id in DANGEROUS_NAMES:
            raise ValueError(f"Reference to dangerous name '{node.id}' is not allowed.")

# -------------------------
# Safe execution in separate process
# -------------------------

# def _worker_exec(code_str: str, dfs:  Dict[str, pd.DataFrame], queue: Queue):
#     """
#     Worker: executed in a separate process.
#     - Unpickle df (already pickled by Process args)
#     - Exec code_str in restricted environment (limited builtins)
#     - If transform(df) exists -> call it
#       else if 'result' variable exists -> use it directly
#     - Send result or detailed error info to queue
#     """
#     try:
#         logging.info("Starting execution...")
#         print("[Worker] Code to execute:\n", code_str, flush=True)
        
#         # def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
#         #     # allow only a small set of safe modules (expand carefully if needed)
#         #     _real_import = builtins.__import__
            
#         #     allowed_root = {
#         #         "math", "statistics", "datetime", "json", "re",
#         #         "itertools", "functools", "operator", "collections",
#         #         "pandas", "numpy"
#         #     }
#         #     root = name.split(".")[0]
            
#         #     if root in allowed_root:
#         #         # if module already loaded use that, else delegate to real import
#         #         return _real_import(name, globals, locals, fromlist, level)
#         #     raise ImportError(f"Import of module '{name}' is restricted.")

#         # allowed_builtins_names = [
#         #     "abs", "all", "any", "bool", "chr", "complex", "dict", "divmod",
#         #     "enumerate", "float", "int", "len", "list", "map", "max", "min",
#         #     "next", "pow", "range", "repr", "round", "sorted", "sum", "zip",
#         #     "print"
#         # ]
#         # safe_builtins = {name: getattr(builtins, name)
#         #                     for name in allowed_builtins_names if hasattr(builtins, name)}
        
#         # # Create globals with pandas and numpy available
#         # restricted_globals = {"pd": pd, "np": np, "__builtins__": safe_builtins}
#         # # inject safe __import__ so import statements in generated code don't fail
#         # safe_builtins["__import__"] = _safe_import
#         # # Local namespace for exec
#         # local_ns = {"df": df}
#         # logging.info("[Worker] DataFrame injected into local_ns.")

#         # # Execute the user-provided code
#         # logging.info("[Worker] Executing code...")
#         # exec(code_str, restricted_globals, local_ns)
#         # logging.info("[Worker] Execution completed successfully.")
        
#         import builtins

#         # Restricted builtins
#         # allowed = ["abs","all","any","bool","dict","float","int","len","list",
#         #            "max","min","range","sum","sorted","print"]
#         # safe_builtins = {k: getattr(builtins, k) for k in allowed}

#         # def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
#         #     root = name.split(".")[0]
#         #     if root in ["pandas", "numpy"]:
#         #         return builtins.__import__(name, globals, locals, fromlist, level)
#         #     raise ImportError(f"Import blocked: {name}")

#         # safe_builtins["__import__"] = safe_import

#         # glb = {"pd": pd, "np": np, "__builtins__": builtins.__dict__}
#         local_ns = {"pd": pd, "np": np, "__builtins__": builtins.__dict__, "dfs": dfs}

#         # Execute generated code
#         exec(code_str, local_ns)

#         # --- Flexible output handling ---
#         if "transform" in local_ns:
#             print("[Worker] Found transform(df) function, calling it...")
#             result = local_ns["transform"](dfs)
#         elif "result" in local_ns:
#             print("[Worker] Found 'result' variable, using it directly.")
#             result = local_ns["result"]
#         else:
#             msg = "Neither transform(df) function nor result variable found after execution."
#             print("[Worker] ERROR:", msg)
#             queue.put({"status": "error", "error": msg})
#             return

#         # --- Validate result type ---
#         if isinstance(result, dict):
#             if isinstance(result.get("DataFrame"), pd.DataFrame):
#                 print("[Worker] Result is a DataFrame, sending back.")
#                 queue.put({"status": "ok_df_generated", "result": result})
#             else:
#                 msg = "Result dict must contain a DataFrame under key 'DataFrame'."
#                 print("[Worker] ERROR:", msg)
#                 queue.put({"status": "error", "result": msg})
#         elif isinstance(result, str):
#             print("[Worker] Result is string.")
#             queue.put({"status": "ok_str_generated", "result": result})
            
#         else:
#             print(f"[Worker] Result is of type {type(result)}, converting to string.")
#             queue.put({"status": "ok_random_generated", "result": str(result)})

#         print("[Worker] Successfully pushed result to queue.")

#     except Exception as e:
#         logging.info('here')
#         tb = traceback.format_exc()
#         print("[Worker] EXCEPTION OCCURRED:", e)
#         logging.info(tb)
#         queue.put({"status": "error", "result": str(e), 
#                    "traceback": tb})


# def run_generated_code_in_subprocess(code_str: str, dfs: Dict[str, pd.DataFrame], timeout: int = 30):
#     """
#     Runs AST checks, then runs code in separate process with timeout.
#     Returns dict with status and either new_df or serialized output.
#     """
    
#     if code_str.startswith("```"):
#         # Split by lines
#         lines = code_str.split("\n")
#         # Remove first line (```python or ```)
#         if lines[0].startswith("```"):
#             lines = lines[1:]
#         # Remove last line (```) if present
#         if lines and lines[-1].strip() == "```":
#             lines = lines[:-1]
#         # Rejoin
#         code_str = "\n".join(lines).strip()
        
#     try:
#         ast_safety_check(code_str)
#     except Exception as e:
#         return {"status": "error", "error": f"AST safety check failed: {e}", "traceback": traceback.format_exc()}
    
#     q = mp.Queue()
#     p = Process(target=_worker_exec, args=(code_str, dfs, q)
#                 ,daemon=True
#                 )
    
#     p.start()
    
#     p.join(timeout)
    
#     if p.is_alive():
#         p.terminate()
#         return {"status": "error", "error": "Execution timed out.", "traceback": None}

#     if not q.empty():
#         out = q.get()
#     else:
#         return {"status": "error", "error": "No result returned from worker.", "traceback": None}

#     # Include traceback for error cases
#     if out.get("status", "").startswith("ok"):
#         print("Execution successful, returning output.")
#         return out
#     else:
#         return {
#             "status": "error",
#             "error": out.get("result", "Unknown error"),
#             "traceback": out.get("traceback", "No traceback captured")
#         }


def _worker_exec(code_str: str, df:  Dict[str, pd.DataFrame], queue: Queue, stdout_path: str, stderr_path: str):
    """
    Worker executed in a separate process.
    Redirects stdout/stderr to files, enables faulthandler and returns detailed error info.
    """
    import sys
    import faulthandler
    so = None
    se = None
    original_stdout, original_stderr = sys.stdout, sys.stderr
    try:
        # Redirect stdout/stderr to files so parent can inspect after crash
        so = open(stdout_path, "w", encoding="utf-8", errors="replace")
        se = open(stderr_path, "w", encoding="utf-8", errors="replace")
        sys.stdout = so
        sys.stderr = se

        faulthandler.enable(file=se)

        print("[Worker] Starting execution...", flush=True)
        print("[Worker] Code to execute:\n", code_str, flush=True)

        local_ns = {"pd": pd, "np": np, "__builtins__": builtins.__dict__, "df": df}

        ############ UNCOMMENT BELOW #############################
        
        # exec(code_str, local_ns)

        # if "transform" in local_ns:
        #     print("[Worker] Found transform(dfs) function, calling it...", flush=True)
        #     result = local_ns["transform"](dfs)
        # elif "result" in local_ns:
        #     print("[Worker] Found 'result' variable, using it directly.", flush=True)
        #     result = local_ns["result"]
        # else:
        #     msg = "Neither transform(dfs) function nor result variable found after execution."
        #     print("[Worker] ERROR:", msg, flush=True)
        #     queue.put({"status": "error", "error": msg, "traceback": None})
        #     so.close()
        #     se.close()
        #     return

        # if isinstance(result, dict) and isinstance(result.get("DataFrame"), pd.DataFrame):
        #     print("[Worker] Result is a DataFrame, sending back.", flush=True)
        #     queue.put({"status": "ok_df_generated", "result": result})
        # elif isinstance(result, str):
        #     print("[Worker] Result is string.", flush=True)
        #     queue.put({"status": "ok_str_generated", "result": result})
        # else:
        #     queue.put({"status": "ok_random_generated", "result": str(result)})

        # so.close()
        
        ############ UNCOMMENT ABOVE #############################
        
        # ... inside _worker_exec function ...

        # Execute generated code
        exec(code_str, local_ns)

        if "transform" in local_ns:
            print("[Worker] Found transform(dfs) function, calling it...", flush=True)
            result = local_ns["transform"](df)
        elif "result" in local_ns:
            print("[Worker] Found 'result' variable, using it directly.", flush=True)
            result = local_ns["result"]
        else:
            msg = "Neither transform(dfs) function nor result variable found after execution."
            queue.put({"status": "error", "error": msg, "traceback": msg})
            return

        # --- MODIFIED RESULT HANDLING ---
        # Check for the new dual-output format: { "response": str, "dataframes": dict }
        if isinstance(result, dict) and ("response" in result or "dataframes" in result):
            print("[Worker] Result is a Dual-Output Dictionary.", flush=True)
            # Basic validation of the inner structure could go here if needed
            queue.put({"status": "ok_mixed_generated", "result": result})
            
        # Legacy support (Single DataFrame wrapper)
        elif isinstance(result, dict) and isinstance(result.get("DataFrame"), pd.DataFrame):
            print("[Worker] Result is a single DataFrame wrapper.", flush=True)
            queue.put({"status": "ok_df_generated", "result": result})
            
        elif isinstance(result, str):
            print("[Worker] Result is string.", flush=True)
            queue.put({"status": "ok_str_generated", "result": result})
        else:
            queue.put({"status": "ok_random_generated", "result": str(result)})

    except BaseException as e:
        # Catch BaseException as well as Exception: generated code can call
        # exit(), which raises SystemExit and otherwise bypasses an Exception
        # handler, leaving the parent with no traceback.
        tb = traceback.format_exc()
        try:
            # The queue is the primary channel; the file is a durable fallback
            # if the worker dies before its queue feeder flushes.
            if se is not None:
                se.write("\n[Worker Exception Traceback]\n")
                se.write(tb)
                se.flush()
            else:
                with open(stderr_path, "a", encoding="utf-8", errors="replace") as err_file:
                    err_file.write("\n[Worker Exception Traceback]\n")
                    err_file.write(tb)
        except Exception:
            pass
        try:
            queue.put({"status": "error", "error": str(e), "traceback": tb})
        except Exception:
            pass
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        try:
            if so is not None:
                so.close()
        except Exception:
            pass
        try:
            if se is not None:
                se.close()
        except Exception:
            pass


def run_generated_code_in_subprocess(code_str: str, df: Dict[str, pd.DataFrame], timeout: int = 30):
    """
    Runs AST checks, then runs code in separate process with timeout.
    Returns dict with status and either new_df or diagnostics including exitcode, stdout and stderr.
    """
    if code_str.startswith("```"):
        lines = code_str.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code_str = "\n".join(lines).strip()

    try:
        ast_safety_check(code_str)
    except Exception as e:
        return {"status": "error", "error": f"AST safety check failed: {e}", "traceback": traceback.format_exc()}

    # Ensure spawn method on platforms that need it
    try:
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    q = mp.Queue()

    # Create temp files for stdout/stderr from worker
    stdout_tf = tempfile.NamedTemporaryFile(delete=False, prefix="worker_stdout_", suffix=".log", mode="w", encoding="utf-8")
    stderr_tf = tempfile.NamedTemporaryFile(delete=False, prefix="worker_stderr_", suffix=".log", mode="w", encoding="utf-8")
    stdout_tf.close()
    stderr_tf.close()
    stdout_path = stdout_tf.name
    stderr_path = stderr_tf.name

    p = Process(target=_worker_exec, args=(code_str, df, q, stdout_path, stderr_path))
    p.daemon = False
    p.start()

    out = None
    try:
        out = q.get(timeout=timeout)
    except Empty:
        # The worker may have crashed before it could publish its result.
        # Its stderr file and exit code below are then the diagnostics.
        pass

    # Give short time for process to exit cleanly
    p.join(2)

    # Read stdout/stderr for diagnostics
    def _read_file_safe(path):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except Exception:
            return ""

    stdout_content = _read_file_safe(stdout_path)
    stderr_content = _read_file_safe(stderr_path)

    # Cleanup temp files
    try:
        os.unlink(stdout_path)
    except Exception:
        pass
    try:
        os.unlink(stderr_path)
    except Exception:
        pass

    # If we got a result from the queue, return it (including any traceback field)
    if out:
        if out.get("status", "").startswith("ok"):
            return out
        else:
            tb = out.get("traceback") or stderr_content or "No traceback captured"
            return {"status": "error", "error": out.get("error", "Unknown error"), "traceback": tb, "exitcode": p.exitcode, "stdout": stdout_content, "stderr": stderr_content}

    # No queue result -> inspect exitcode / captured logs
    exitcode = p.exitcode
    if p.is_alive():
        p.terminate()
        p.join(1)
        exitcode = p.exitcode

    # Build helpful diagnostic message
    diag = {
        "status": "error",
        "error": "No result returned from worker.",
        "traceback": stderr_content or "No traceback captured",
        "exitcode": exitcode,
        "stdout": stdout_content,
        "stderr": stderr_content
    }

    # If process exited with non-zero exitcode provide hint about signal on POSIX
    if exitcode is not None and exitcode != 0:
        diag["error"] = f"Worker process exited with code {exitcode}."
        # On POSIX negative exitcode indicates signal number
        try:
            if exitcode < 0:
                diag["error"] += f" Killed by signal {-exitcode}."
        except Exception:
            pass

    return diag


if __name__ == "__main__":
    import multiprocessing as mp
    import pandas as pd
    mp.set_start_method("spawn", force=True)
    code = "import pandas as pd\nresult = df.dtypes.to_string()"
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    out = run_generated_code_in_subprocess(code, df)
    print(out)
