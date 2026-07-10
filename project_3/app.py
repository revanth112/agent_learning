import ast
import json
import os
import subprocess
import sys
import tempfile
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, BadRequestError

from tool_registry import ToolRegistry

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "openai/gpt-oss-20b"
registry = ToolRegistry()

DATAFRAME = pd.DataFrame({
    "product": ["Laptop", "Mouse", "Keyboard", "Monitor"],
    "price": [50000, 1000, 3000, 15000],
    "quantity": [2, 20, 10, 5],
})


# ---------------- EXISTING TOOLS ----------------

def list_columns():
    return {"columns": DATAFRAME.columns.tolist()}


def describe_data():
    return {
        "rows": len(DATAFRAME),
        "columns": DATAFRAME.columns.tolist(),
        "records": DATAFRAME.to_dict(orient="records"),
    }


EMPTY_SCHEMA = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
}

registry.register(
    "list_columns",
    "List the columns in the sales dataset.",
    EMPTY_SCHEMA,
    list_columns,
)

registry.register(
    "describe_data",
    "Return the dataset columns and records.",
    EMPTY_SCHEMA,
    describe_data,
)


# ---------------- GENERATED CODE VALIDATION ----------------

BLOCKED_NODES = (
    ast.Import, ast.ImportFrom, ast.ClassDef,
    ast.AsyncFunctionDef, ast.Lambda,
    ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith,
)

BLOCKED_NAMES = {
    "eval", "exec", "compile", "open", "__import__",
    "input", "globals", "locals", "vars",
    "getattr", "setattr", "delattr",
}

BLOCKED_ATTRS = {
    "system", "popen", "spawn", "fork",
    "remove", "unlink", "rmdir", "rmtree",
    "socket", "connect", "urlopen",
}


def validate_code(code):
    tree = ast.parse(code)

    functions = [
        n for n in tree.body
        if isinstance(n, ast.FunctionDef)
    ]

    if len(functions) != 1:
        raise ValueError("Exactly one top-level function is required.")

    fn = functions[0]

    if fn.name != "generated_tool":
        raise ValueError("Function must be named generated_tool.")

    if len(fn.args.args) != 1 or fn.args.args[0].arg != "records":
        raise ValueError(
            "Required signature: generated_tool(records)"
        )

    if fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs:
        raise ValueError("Extra function arguments are not allowed.")

    for node in ast.walk(tree):
        if isinstance(node, BLOCKED_NODES):
            raise ValueError(
                f"Blocked syntax: {type(node).__name__}"
            )

        if isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            raise ValueError(f"Blocked name: {node.id}")

        if isinstance(node, ast.Attribute) and node.attr in BLOCKED_ATTRS:
            raise ValueError(f"Blocked attribute: {node.attr}")


# ---------------- TOOL BUILDER ----------------

BUILDER_PROMPT = """
You generate one restricted Python function.

Return only Python source code.

Requirements:
- exact function name: generated_tool
- exact signature: generated_tool(records)
- records is a list of dictionaries
- return JSON-serializable data
- no imports
- no file access
- no network access
- no environment access
- no subprocesses or shell commands
- no eval, exec, compile, open, or __import__
- no printing
- no Markdown fences
- deterministic implementation
"""


def clean_code(code):
    code = code.strip()

    if code.startswith("```"):
        lines = code.splitlines()
        lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        code = "\n".join(lines).strip()

    return code


def build_tool(requirement):

    response = client.chat.completions.create(

        model=MODEL,

        temperature=0,

        messages=[
            {
                "role": "system",
                "content": BUILDER_PROMPT
            },
            {
                "role": "user",
                "content": f"""
                Create a tool for this requirement:

                {requirement}
                """
            }
        ]

    )


    code = clean_code(

        response.choices[0].message.content or ""

    )


    if not code:

        raise ValueError(

            "Tool Builder returned empty code."

        )


    validate_code(code)


    return code


# ---------------- SUBPROCESS RUNNER ----------------
# Educational isolation only; not a production sandbox.

def run_generated(code, records, timeout=5):
    validate_code(code)

    runner = f"""
import json

{code}

records = {repr(records)}
result = generated_tool(records)
print(json.dumps(result))
"""

    path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(runner)
            path = f.name

        process = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if process.returncode != 0:
            return {
                "success": False,
                "stage": "execution",
                "error": process.stderr.strip(),
            }

        try:
            output = json.loads(process.stdout.strip())
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "stage": "json_validation",
                "error": str(e),
            }

        return {"success": True, "output": output}

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stage": "timeout",
            "error": "Execution timed out.",
        }

    finally:
        if path and os.path.exists(path):
            os.remove(path)


# ---------------- SEMANTIC TEST ----------------

def test_generated_tool(code):
    test_records = [
        {"product": "A", "price": 100, "quantity": 2},
        {"product": "B", "price": 200, "quantity": 1},
    ]

    result = run_generated(code, test_records)

    if not result["success"]:
        return {"passed": False, **result}

    output = result["output"]

    try:
        if not isinstance(output, list) or len(output) != 2:
            raise ValueError("Expected a list with two rows.")

        by_product = {row["product"]: row for row in output}

        for product in ("A", "B"):
            row = by_product[product]

            if row["revenue"] != 200:
                raise ValueError(
                    f"Wrong revenue for {product}"
                )

            pct = float(row["contribution_percentage"])

            if abs(pct - 50.0) > 0.001:
                raise ValueError(
                    f"Wrong contribution for {product}"
                )

    except (KeyError, TypeError, ValueError) as e:
        return {
            "passed": False,
            "stage": "semantic_validation",
            "error": str(e),
            "output": output,
        }

    return {
        "passed": True,
        "stage": "complete",
        "output": output,
    }


# ---------------- META TOOL ----------------

def request_new_tool(requirement):
    print("\n========== TOOL CREATION REQUEST ==========")
    print(requirement)

    try:
        code = build_tool(requirement)
    except Exception as e:
        return {
            "status": "failed",
            "stage": "generation_or_static_validation",
            "error": str(e),
        }

    print("\n========== GENERATED CODE ==========")
    print(code)

    test = test_generated_tool(code)

    print("\n========== TEST RESULT ==========")
    print(test)

    if not test["passed"]:
        return {
            "status": "failed",
            "stage": test.get("stage"),
            "error": test.get("error"),
        }

    def generated_dataset_analysis():
        records = DATAFRAME.to_dict(orient="records")
        execution = run_generated(code, records)

        if not execution["success"]:
            raise RuntimeError(str(execution))

        return execution["output"]

    registry.register(
        "generated_dataset_analysis",
        f"Run the generated capability: {requirement}",
        EMPTY_SCHEMA,
        generated_dataset_analysis,
    )

    return {
        "status": "success",
        "new_tool_name": "generated_dataset_analysis",
        "message": "Tool generated, tested, and registered.",
    }


META_TOOL = {
    "type": "function",
    "function": {
        "name": "request_new_tool",
        "description": (
            "Create a missing analytical capability when existing "
            "tools cannot perform the required calculation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "requirement": {
                    "type": "string",
                    "description": (
                        "Precise calculation requirement including "
                        "input fields, formulas, and output fields."
                    ),
                }
            },
            "required": ["requirement"],
            "additionalProperties": False,
        },
    },
}


def current_tools():
    return registry.schemas() + [META_TOOL]


def execute_agent_tool(name, arguments):
    if name == "request_new_tool":
        return request_new_tool(arguments["requirement"])

    return registry.execute(name, arguments)


# ---------------- MAIN AGENT ----------------

SYSTEM_PROMPT = """
You are a data-analysis agent.

Goal:
Complete the user's analytical request using tools.

Rules:
1. Inspect the dataset when useful.
2. Use an existing tool if it can complete the task.
3. If the calculation capability is missing, call
   request_new_tool exactly once with:
   - input fields
   - formulas
   - expected output fields
4. After successful creation, call
   generated_dataset_analysis.
5. Use its observation for the final answer.
6. Never invent results.
7. Never repeatedly request the same capability.
8. Do not finish until the analysis is complete.
"""

USER_GOAL = """
Calculate revenue contribution percentage for every product.

Formula:
revenue = price * quantity

contribution_percentage =
product revenue / total revenue * 100

Return:
- product
- revenue
- contribution_percentage
"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_GOAL},
]


def call_agent(max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=current_tools(),
                tool_choice="auto",
                temperature=0
            )

        except BadRequestError as error:
            error_text = str(error)


            parse_failure = (("output_parse_failed" in error_text) or ("Parsing failed" in error_text))

            if not parse_failure or attempt == max_retries:
                raise

            print(
                f"\nParse failure. Retry {attempt}/{max_retries}..."
            )
            time.sleep(1)


MAX_ITERATIONS = 10
response = call_agent()


for iteration in range(1, MAX_ITERATIONS + 1):
    print(
        f"\n\n========== ITERATION {iteration} =========="
    )

    assistant = response.choices[0].message

    print("\nASSISTANT TEXT:")
    print(repr(assistant.content))

    tool_calls = assistant.tool_calls or []

    if tool_calls:
        messages.append(
            assistant.model_dump(exclude_none=True)
        )

        for call in tool_calls:
            name = call.function.name
            raw_arguments = call.function.arguments

            print("\nAGENT ACTION:")
            print(name)

            print("\nARGUMENTS:")
            print(raw_arguments)

            try:
                arguments = json.loads(raw_arguments)
                result = execute_agent_tool(
                    name,
                    arguments,
                )
            except Exception as e:
                result = {
                    "status": "error",
                    "error": str(e),
                }

            print("\nOBSERVATION:")
            print(result)

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": name,
                "content": json.dumps(result),
            })

        print("\nCURRENT TOOLS:")
        for tool in current_tools():
            print("-", tool["function"]["name"])

        response = call_agent()
        continue

    final_text = (assistant.content or "").strip()

    if final_text:
        print("\n========== FINAL ANSWER ==========")
        print(final_text)
        break

    messages.append(
        assistant.model_dump(exclude_none=True)
    )

    messages.append({
        "role": "user",
        "content": (
            "The task is incomplete. Continue. If the capability "
            "is missing, request it once. If the generated tool "
            "exists, call it. If results exist, answer now."
        ),
    })

    response = call_agent()

else:
    print("\nMaximum iterations reached.")
