import os
import json

from dotenv import load_dotenv
from openai import OpenAI


# --------------------------------------------------
# 1. SETUP
# --------------------------------------------------

load_dotenv()


client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
)


MODEL = "grok-4.3"


# --------------------------------------------------
# 2. REAL PYTHON TOOLS
# --------------------------------------------------

def get_hotel_cost(city: str, days: int) -> dict:

    price_per_night = {
        "Goa": 2500,
        "Hyderabad": 2000,
        "Chennai": 2200,
    }

    nightly_price = price_per_night.get(
        city,
        2000
    )

    total = nightly_price * days

    return {
        "city": city,
        "days": days,
        "nightly_price": nightly_price,
        "hotel_total": total,
    }


def get_travel_cost(
    from_city: str,
    to_city: str
) -> dict:

    routes = {
        ("Bengaluru", "Goa"): 6000,
        ("Hyderabad", "Goa"): 7000,
        ("Chennai", "Goa"): 8000,
    }

    total = routes.get(
        (from_city, to_city),
        5000
    )

    return {
        "from_city": from_city,
        "to_city": to_city,
        "round_trip_cost": total,
    }


def get_food_cost(
    city: str,
    days: int
) -> dict:

    daily_food_cost = 1200

    total = daily_food_cost * days

    return {
        "city": city,
        "days": days,
        "daily_food_cost": daily_food_cost,
        "food_total": total,
    }


def calculate_total_cost(
    hotel_cost: float,
    travel_cost: float,
    food_cost: float
) -> dict:

    total = (
        hotel_cost
        + travel_cost
        + food_cost
    )

    return {
        "hotel_cost": hotel_cost,
        "travel_cost": travel_cost,
        "food_cost": food_cost,
        "total_trip_cost": total,
    }


# --------------------------------------------------
# 3. TOOL SCHEMAS FOR THE LLM
# --------------------------------------------------

tools = [

    {
        "type": "function",
        "name": "get_hotel_cost",
        "description": (
            "Get the estimated hotel cost for a city "
            "for a specified number of days."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string"
                },
                "days": {
                    "type": "integer"
                }
            },
            "required": [
                "city",
                "days"
            ],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "get_travel_cost",
        "description": (
            "Get estimated round-trip travel cost "
            "between two cities."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "from_city": {
                    "type": "string"
                },
                "to_city": {
                    "type": "string"
                }
            },
            "required": [
                "from_city",
                "to_city"
            ],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "get_food_cost",
        "description": (
            "Get estimated food expenses for a city "
            "and number of days."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string"
                },
                "days": {
                    "type": "integer"
                }
            },
            "required": [
                "city",
                "days"
            ],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "calculate_total_cost",
        "description": (
            "Calculate total trip cost from hotel, "
            "travel, and food costs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "hotel_cost": {
                    "type": "number"
                },
                "travel_cost": {
                    "type": "number"
                },
                "food_cost": {
                    "type": "number"
                }
            },
            "required": [
                "hotel_cost",
                "travel_cost",
                "food_cost"
            ],
            "additionalProperties": False
        }
    }
]


# --------------------------------------------------
# 4. TOOL ROUTER
# --------------------------------------------------

def execute_tool(
    tool_name: str,
    arguments: dict
):

    if tool_name == "get_hotel_cost":

        return get_hotel_cost(
            city=arguments["city"],
            days=arguments["days"]
        )


    elif tool_name == "get_travel_cost":

        return get_travel_cost(
            from_city=arguments["from_city"],
            to_city=arguments["to_city"]
        )


    elif tool_name == "get_food_cost":

        return get_food_cost(
            city=arguments["city"],
            days=arguments["days"]
        )


    elif tool_name == "calculate_total_cost":

        return calculate_total_cost(
            hotel_cost=arguments["hotel_cost"],
            travel_cost=arguments["travel_cost"],
            food_cost=arguments["food_cost"]
        )


    else:

        raise ValueError(
            f"Unknown tool: {tool_name}"
        )


# --------------------------------------------------
# 5. USER GOAL
# --------------------------------------------------

user_question = """
I have a budget of ₹20,000.

Can I afford a 3-day trip
from Bengaluru to Goa?

Consider:
- hotel
- round-trip travel
- food

Give me the total cost and remaining money.
"""


# --------------------------------------------------
# 6. FIRST AGENT CALL
# --------------------------------------------------

response = client.responses.create(

    model=MODEL,

    instructions="""
You are a trip budget planning agent.

Your goal is to determine whether the user's budget
is enough for the requested trip.

Use the available tools to collect all required costs.

Do not invent prices.

Continue using tools until you have enough information.

When all required information is available,
give the final answer.
""",

    input=[
        {
            "role": "user",
            "content": user_question
        }
    ],

    tools=tools
)


# --------------------------------------------------
# 7. AGENT LOOP
# --------------------------------------------------

MAX_ITERATIONS = 10


for iteration in range(MAX_ITERATIONS):

    print(
        f"\n========== ITERATION {iteration + 1} =========="
    )


    tool_outputs = []


    for item in response.output:

        if item.type == "function_call":

            print("\nAGENT ACTION:")
            print(item.name)

            print("\nARGUMENTS:")
            print(item.arguments)


            arguments = json.loads(
                item.arguments
            )


            result = execute_tool(
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
    # NO TOOL CALL = AGENT HAS FINISHED
    # ----------------------------------------------

    if not tool_outputs:

        print("\n========== FINAL ANSWER ==========")

        print(
            response.output_text
        )

        break


    # ----------------------------------------------
    # SEND OBSERVATIONS BACK TO MODEL
    # ----------------------------------------------

    response = client.responses.create(

        model=MODEL,

        previous_response_id=response.id,

        input=tool_outputs,

        tools=tools
    )


else:

    print(
        "Agent stopped because maximum iterations were reached."
    )
