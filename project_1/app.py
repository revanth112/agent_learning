import os
import json

from dotenv import load_dotenv
from openai import OpenAI


# --------------------------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# 2. CREATE GROQ CLIENT USING OPENAI SDK
# --------------------------------------------------

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


MODEL = "openai/gpt-oss-20b"


# --------------------------------------------------
# 3. OUR REAL PYTHON FUNCTION
# --------------------------------------------------

def get_customer_profile(customer_id: str) -> dict:
    """
    Simulates fetching customer information from a database.
    """

    fake_database = {
        "CUST001": {
            "name": "Revanth",
            "city": "Bengaluru",
            "membership": "Gold",
        }
    }

    return fake_database.get(
        customer_id,
        {
            "error": "Customer not found"
        }
    )


# --------------------------------------------------
# 4. TOOL DEFINITION GIVEN TO THE LLM
# --------------------------------------------------

tools = [
    {
        "type": "function",
        "name": "get_customer_profile",
        "description": (
            "Retrieve the current customer's profile information. "
            "Use this tool when answering questions about the customer's "
            "name, city, or membership details."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "The unique customer ID."
                }
            },
            "required": ["customer_id"],
            "additionalProperties": False
        }
    }
]


# --------------------------------------------------
# 5. TOOL EXECUTOR
# --------------------------------------------------

def execute_tool(tool_name: str, arguments: dict):

    if tool_name == "get_customer_profile":

        return get_customer_profile(
            customer_id=arguments["customer_id"]
        )

    raise ValueError(
        f"Unknown tool: {tool_name}"
    )


# --------------------------------------------------
# 6. USER QUESTION
# --------------------------------------------------

user_question = "What all the info you know about me?"


# --------------------------------------------------
# 7. FIRST LLM CALL
# --------------------------------------------------

input_list=[
        {
            "role": "user",
            "content": user_question
        }
    ]

response = client.responses.create(

    model=MODEL,

    instructions="""
You are a customer support assistant.

The current authenticated customer's ID is CUST001.

Answer the user's question.

Use an available tool only when the information required
to answer the question must be retrieved from that tool.
""",

    input=input_list,

    tools=tools
)


# --------------------------------------------------
# 8. CHECK WHETHER LLM REQUESTED A TOOL
# --------------------------------------------------

tool_outputs = []


for item in response.output:

    if item.type == "function_call":

        print("\nLLM decided to call a tool.")

        print("Tool name:")
        print(item.name)

        print("Arguments:")
        print(item.arguments)


        arguments = json.loads(
            item.arguments
        )


        result = execute_tool(
            tool_name=item.name,
            arguments=arguments
        )


        print("\nTool result:")
        print(result)


        tool_outputs.append(
            {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps(result)
            }
        )


# --------------------------------------------------
# 9. IF TOOL WAS CALLED, SEND RESULT BACK
# --------------------------------------------------

if tool_outputs:

    input_list+=response.output

    input_list+=tool_outputs

    final_response = client.responses.create(

        model=MODEL,

        instructions="""
            You are a customer support assistant.

            The current authenticated customer's ID is CUST001.

            Answer the user's question using the tool result.
            """,

        input=input_list,

        tools=tools
    )

    print("\nFINAL ANSWER:")
    print(final_response.output_text)


else:

    print("\nFINAL ANSWER:")
    print(response.output_text)
