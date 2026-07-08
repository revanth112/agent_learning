import os
import ast
import json
import tempfile
import subprocess

import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI

from tool_registry import ToolRegistry


# --------------------------------------------------
# 1. SETUP
# --------------------------------------------------

load_dotenv()


client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
)


MODEL = "grok-4.3"


registry = ToolRegistry()


# --------------------------------------------------
# 2. SAMPLE DATA
# --------------------------------------------------

DATAFRAME = pd.DataFrame(
    {
        "product": [
            "Laptop",
            "Mouse",
            "Keyboard",
            "Monitor",
        ],

        "price": [
            50000,
            1000,
            3000,
            15000,
        ],

        "quantity": [
            2,
            20,
            10,
            5,
        ],
    }
)


# --------------------------------------------------
# 3. EXISTING NORMAL TOOLS
# --------------------------------------------------

def list_columns() -> dict:

    return {
        "columns": DATAFRAME.columns.tolist()
    }


def describe_data() -> dict:

    return {
        "rows": len(DATAFRAME),

        "columns": DATAFRAME.columns.tolist(),

        "sample": DATAFRAME.head().to_dict(
            orient="records"
        )
    }


# --------------------------------------------------
# 4. REGISTER EXISTING TOOLS
# --------------------------------------------------

registry.register(

    name="list_columns",

    description=(
        "List all column names available in the sales dataset."
    ),

    parameters={
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False
    },

    function=list_columns
)


registry.register(

    name="describe_data",

    description=(
        "Return the sales dataset structure and sample records."
    ),

    parameters={
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False
    },

    function=describe_data
)


# --------------------------------------------------
# 5. CODE SAFETY CHECK
# --------------------------------------------------

BLOCKED_NODES = (
    ast.Import,
    ast.ImportFrom,
)


BLOCKED_NAMES = {
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "input",
    "globals",
    "locals",
    "vars",
}


BLOCKED_ATTRIBUTES = {
    "system",
    "popen",
    "remove",
    "unlink",
    "rmdir",
    "rmtree",
    "socket",
}


def validate_generated_code(code: str):

    tree = ast.parse(code)


    for node in ast.walk(tree):

        if isinstance(node, BLOCKED_NODES):

            raise ValueError(
                "Generated code contains imports."
            )


        if isinstance(node, ast.Name):

            if node.id in BLOCKED_NAMES:

                raise ValueError(
                    f"Blocked name used: {node.id}"
                )


        if isinstance(node, ast.Attribute):

            if node.attr in BLOCKED_ATTRIBUTES:

                raise ValueError(
                    f"Blocked attribute used: {node.attr}"
                )


    return True


# --------------------------------------------------
# 6. ASK LLM TO BUILD A TOOL
# --------------------------------------------------

def build_tool_with_llm(
    requirement: str
) -> dict:

    builder_prompt = f"""
You are a Python tool builder.

Create exactly one Python function.

REQUIREMENT:

{requirement}


RULES:

1. The function must be named:

generated_tool


2. The function must accept exactly one argument:

records


3. records will be a list of dictionaries.

Example:

[
    {{
        "product": "Laptop",
        "price": 50000,
        "quantity": 2
    }}
]


4. Return only JSON-serializable Python values.

Allowed return values:

- dict
- list
- string
- integer
- float
- boolean
- None


5. Do not import anything.

6. Do not access:
- filesystem
- network
- environment variables
- subprocesses
- shell commands

7. Do not use:
- eval
- exec
- compile
- open
- __import__

8. Return ONLY Python code.

9. Do not use Markdown code fences.

10. The code must be deterministic.
"""


    response = client.responses.create(

        model=MODEL,

        input=[
            {
                "role": "user",
                "content": builder_prompt
            }
        ]
    )


    code = response.output_text.strip()


    return {
        "code": code
    }


# --------------------------------------------------
# 7. TEST GENERATED TOOL
# --------------------------------------------------

def test_generated_tool(
    code: str
) -> dict:

    validate_generated_code(code)


    test_records = [
        {
            "product": "A",
            "price": 100,
            "quantity": 2,
        },
        {
            "product": "B",
            "price": 200,
            "quantity": 1,
        },
    ]


    runner_code = f"""
import json

{code}

records = {repr(test_records)}

result = generated_tool(records)

print(json.dumps(result))
"""


    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False
    ) as temp_file:

        temp_file.write(runner_code)

        temp_path = temp_file.name


    try:

        process = subprocess.run(

            [
                "python",
                temp_path
            ],

            capture_output=True,

            text=True,

            timeout=5
        )


        if process.returncode != 0:

            return {
                "passed": False,
                "error": process.stderr
            }


        output = process.stdout.strip()


        parsed_output = json.loads(output)


        return {
            "passed": True,
            "output": parsed_output
        }


    except subprocess.TimeoutExpired:

        return {
            "passed": False,
            "error": "Generated tool exceeded time limit."
        }


    except Exception as error:

        return {
            "passed": False,
            "error": str(error)
        }


    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)


# --------------------------------------------------
# 8. CREATE CALLABLE FUNCTION FROM GENERATED CODE
# --------------------------------------------------

def create_callable_function(
    code: str
):

    validate_generated_code(code)


    safe_builtins = {
        "len": len,
        "sum": sum,
        "min": min,
        "max": max,
        "round": round,
        "sorted": sorted,
        "enumerate": enumerate,
        "range": range,
        "zip": zip,
        "list": list,
        "dict": dict,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "abs": abs,
    }


    namespace = {
        "__builtins__": safe_builtins
    }


    exec(
        code,
        namespace
    )


    return namespace["generated_tool"]


# --------------------------------------------------
# 9. META-TOOL: REQUEST NEW TOOL
# --------------------------------------------------

def request_new_tool(
    requirement: str
) -> dict:

    print("\n========== TOOL CREATION REQUEST ==========")

    print("\nRequirement:")
    print(requirement)


    # ----------------------------------------------
    # BUILD
    # ----------------------------------------------

    build_result = build_tool_with_llm(
        requirement
    )


    code = build_result["code"]


    print("\n========== GENERATED CODE ==========")

    print(code)


    # ----------------------------------------------
    # TEST
    # ----------------------------------------------

    test_result = test_generated_tool(
        code
    )


    print("\n========== TEST RESULT ==========")

    print(test_result)


    if not test_result["passed"]:

        return {
            "status": "failed",
            "error": test_result["error"]
        }


    # ----------------------------------------------
    # CREATE CALLABLE
    # ----------------------------------------------

    generated_function = create_callable_function(
        code
    )


    # ----------------------------------------------
    # WRAPPER THAT AUTOMATICALLY USES DATASET
    # ----------------------------------------------

    def dynamic_dataset_tool():

        records = DATAFRAME.to_dict(
            orient="records"
        )

        return generated_function(
            records
        )


    # ----------------------------------------------
    # REGISTER NEW TOOL
    # ----------------------------------------------

    registry.register(

        name="generated_dataset_analysis",

        description=requirement,

        parameters={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },

        function=dynamic_dataset_tool
    )


    return {
        "status": "success",

        "new_tool_name":
            "generated_dataset_analysis",

        "description":
            requirement,

        "message":
            "The new tool was generated, tested, and registered."
    }


# --------------------------------------------------
# 10. META TOOL SCHEMA
# --------------------------------------------------

META_TOOL = {

    "type": "function",

    "name": "request_new_tool",

    "description": (
        "Request creation of a new analytical tool when "
        "the currently available tools cannot perform the "
        "calculation required to complete the user's goal."
    ),

    "parameters": {

        "type": "object",

        "properties": {

            "requirement": {

                "type": "string",

                "description": (
                    "A precise description of the missing "
                    "calculation capability."
                )
            }
        },

        "required": [
            "requirement"
        ],

        "additionalProperties": False
    }
}


# --------------------------------------------------
# 11. EXECUTE NORMAL OR META TOOL
# --------------------------------------------------

def execute_agent_tool(
    tool_name: str,
    arguments: dict
):

    if tool_name == "request_new_tool":

        return request_new_tool(
            requirement=arguments["requirement"]
        )


    return registry.execute(
        name=tool_name,
        arguments=arguments
    )


# --------------------------------------------------
# 12. GET CURRENT TOOLS
# --------------------------------------------------

def get_current_tools():

    normal_tools = (
        registry.get_openai_tool_schemas()
    )

    return normal_tools + [META_TOOL]


# --------------------------------------------------
# 13. USER GOAL
# --------------------------------------------------

user_goal = """
Calculate the revenue contribution percentage
of every product in the sales dataset.

Revenue for a product is:

price * quantity

Contribution percentage is:

product revenue / total revenue * 100

Return every product with:
- revenue
- contribution percentage
"""


# --------------------------------------------------
# 14. FIRST AGENT CALL
# --------------------------------------------------

response = client.responses.create(

    model=MODEL,

    instructions="""
You are an autonomous data analysis agent.

Your goal is to complete the user's analytical request.

First inspect the available tools.

Use existing tools when they can complete the task.

If no existing tool can perform a required calculation,
call request_new_tool with a precise requirement.

After a new tool is successfully created,
use the newly available tool.

Continue until the user's goal is complete.

Do not manually invent calculation results.
""",

    input=[
        {
            "role": "user",
            "content": user_goal
        }
    ],

    tools=get_current_tools()
)


# --------------------------------------------------
# 15. MAIN AGENT LOOP
# --------------------------------------------------

MAX_ITERATIONS = 10


for iteration in range(MAX_ITERATIONS):

    print(
        f"\n\n========== AGENT ITERATION {iteration + 1} =========="
    )


    tool_outputs = []


    for item in response.output:

        if item.type == "function_call":

            print("\nAGENT CHOSE TOOL:")
            print(item.name)


            print("\nARGUMENTS:")
            print(item.arguments)


            arguments = json.loads(
                item.arguments
            )


            result = execute_agent_tool(
                tool_name=item.name,
                arguments=arguments
            )


            print("\nOBSERVATION:")
            print(result)


            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(result)
                }
            )


    # ----------------------------------------------
    # FINISH CONDITION
    # ----------------------------------------------

    if not tool_outputs:

        print(
            "\n========== FINAL ANSWER =========="
        )

        print(
            response.output_text
        )

        break


    # ----------------------------------------------
    # IMPORTANT:
    # REFRESH TOOLS BECAUSE A NEW TOOL
    # MAY HAVE BEEN REGISTERED
    # ----------------------------------------------

    response = client.responses.create(

        model=MODEL,

        previous_response_id=response.id,

        input=tool_outputs,

        tools=get_current_tools()
    )


else:

    print(
        "Agent stopped because maximum iterations were reached."
    )
