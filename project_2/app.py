import os
import json

from dotenv import load_dotenv
from openai import OpenAI


# ==================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()


# ==================================================
# 2. CREATE GROQ CLIENT USING OPENAI SDK
# ==================================================

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


MODEL = "openai/gpt-oss-20b"


# ==================================================
# 3. REAL PYTHON TOOLS
# ==================================================


def get_hotel_cost(
    city: str,
    days: int
) -> dict:

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


# ==================================================
# 4. TOOL SCHEMAS GIVEN TO THE LLM
# ==================================================


tools = [

    # --------------------------------------------------
    # HOTEL TOOL
    # --------------------------------------------------

    {
        "type": "function",

        "name": "get_hotel_cost",

        "description": (
            "Get the estimated total hotel cost "
            "for a destination city and number of nights."
        ),

        "parameters": {

            "type": "object",

            "properties": {

                "city": {

                    "type": "string",

                    "description":
                        "The destination city."

                },


                "days": {

                    "type": "integer",

                    "description":
                        "The number of hotel nights."

                }

            },


            "required": [

                "city",

                "days"

            ],


            "additionalProperties": False

        }

    },


    # --------------------------------------------------
    # TRAVEL TOOL
    # --------------------------------------------------

    {
        "type": "function",

        "name": "get_travel_cost",

        "description": (
            "Get the estimated round-trip travel cost "
            "between the starting city and destination city."
        ),

        "parameters": {

            "type": "object",

            "properties": {

                "from_city": {

                    "type": "string",

                    "description":
                        "The city where the journey starts."

                },


                "to_city": {

                    "type": "string",

                    "description":
                        "The destination city."

                }

            },


            "required": [

                "from_city",

                "to_city"

            ],


            "additionalProperties": False

        }

    },


    # --------------------------------------------------
    # FOOD TOOL
    # --------------------------------------------------

    {
        "type": "function",

        "name": "get_food_cost",

        "description": (
            "Get the estimated total food cost "
            "for a city and number of days."
        ),

        "parameters": {

            "type": "object",

            "properties": {

                "city": {

                    "type": "string",

                    "description":
                        "The destination city."

                },


                "days": {

                    "type": "integer",

                    "description":
                        "Number of trip days."

                }

            },


            "required": [

                "city",

                "days"

            ],


            "additionalProperties": False

        }

    },


    # --------------------------------------------------
    # TOTAL CALCULATION TOOL
    # --------------------------------------------------

    {
        "type": "function",

        "name": "calculate_total_cost",

        "description": (
            "Calculate the total trip cost using "
            "hotel cost, travel cost, and food cost. "
            "Use this only after all three individual "
            "cost components have been collected."
        ),

        "parameters": {

            "type": "object",

            "properties": {

                "hotel_cost": {

                    "type": "number",

                    "description":
                        "Total hotel cost."

                },


                "travel_cost": {

                    "type": "number",

                    "description":
                        "Total round-trip travel cost."

                },


                "food_cost": {

                    "type": "number",

                    "description":
                        "Total food cost."

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


# ==================================================
# 5. TOOL EXECUTION ROUTER
# ==================================================


def execute_tool(
    tool_name: str,
    arguments: dict
):


    # --------------------------------------------------
    # HOTEL COST
    # --------------------------------------------------

    if tool_name == "get_hotel_cost":


        return get_hotel_cost(

            city=arguments["city"],

            days=arguments["days"]

        )


    # --------------------------------------------------
    # TRAVEL COST
    # --------------------------------------------------

    elif tool_name == "get_travel_cost":


        return get_travel_cost(

            from_city=arguments["from_city"],

            to_city=arguments["to_city"]

        )


    # --------------------------------------------------
    # FOOD COST
    # --------------------------------------------------

    elif tool_name == "get_food_cost":


        return get_food_cost(

            city=arguments["city"],

            days=arguments["days"]

        )


    # --------------------------------------------------
    # TOTAL COST
    # --------------------------------------------------

    elif tool_name == "calculate_total_cost":


        return calculate_total_cost(

            hotel_cost=arguments["hotel_cost"],

            travel_cost=arguments["travel_cost"],

            food_cost=arguments["food_cost"]

        )


    # --------------------------------------------------
    # UNKNOWN TOOL
    # --------------------------------------------------

    else:


        raise ValueError(

            f"Unknown tool: {tool_name}"

        )


# ==================================================
# 6. USER QUESTION
# ==================================================


user_question = """
I have a budget of ₹20,000.

Can I afford a 3-day trip
from Bengaluru to Goa?

Consider all of these costs:

- hotel
- round-trip travel
- food

Give me:

1. Hotel cost
2. Travel cost
3. Food cost
4. Total trip cost
5. Whether I can afford the trip
6. Remaining money after the trip
"""


# ==================================================
# 7. AGENT INSTRUCTIONS
# ==================================================


AGENT_INSTRUCTIONS = """
You are a trip budget planning agent.

Your goal is to determine whether the user's budget
is sufficient for the requested trip.


REQUIRED PROCESS:

You must obtain all three cost components:

1. Hotel cost
2. Round-trip travel cost
3. Food cost


After obtaining all three costs:

4. Call calculate_total_cost using the collected values.

5. After calculate_total_cost returns its result,
   provide the final answer.


FINAL ANSWER MUST INCLUDE:

- Hotel cost
- Travel cost
- Food cost
- Total trip cost
- User budget
- Remaining money
- Whether the trip is affordable


IMPORTANT RULES:

- Do not invent prices.

- Use tools for all cost information.

- Do not finish before hotel cost is available.

- Do not finish before travel cost is available.

- Do not finish before food cost is available.

- Do not finish before calculate_total_cost has been called.

- Do not repeat a tool call if its result is already
  available in the conversation history.

- You may call one or multiple tools in a response.

- When all required calculations are complete,
  return a clear text answer.

- If the task is incomplete, continue by calling
  the next required tool.
"""


# ==================================================
# 8. LOCAL CONVERSATION HISTORY
# ==================================================


input_list = [

    {

        "role": "user",

        "content": user_question

    }

]


# ==================================================
# 9. FIRST LLM CALL
# ==================================================


response = client.responses.create(

    model=MODEL,

    instructions=AGENT_INSTRUCTIONS,

    input=input_list,

    tools=tools

)


# ==================================================
# 10. AGENT LOOP
# ==================================================


MAX_ITERATIONS = 10


for iteration in range(MAX_ITERATIONS):


    print(

        f"\n\n"
        f"========== ITERATION {iteration + 1} =========="

    )


    # ==================================================
    # DEBUG CURRENT RESPONSE
    # ==================================================


    print("\nRESPONSE STATUS:")


    print(
        response.status
    )


    print("\nRAW RESPONSE ITEMS:")


    for item in response.output:


        print(item)


    print("\nOUTPUT TEXT:")


    print(
        repr(response.output_text)
    )


    # ==================================================
    # FIND FUNCTION CALLS
    # ==================================================


    function_calls = [

        item

        for item in response.output

        if item.type == "function_call"

    ]


    # ==================================================
    # CASE 1:
    #
    # MODEL REQUESTED ONE OR MORE TOOLS
    # ==================================================


    if function_calls:


        tool_outputs = []


        for item in function_calls:


            print(

                "\n----------------------------------"
            )


            print(
                "AGENT ACTION:"
            )


            print(
                item.name
            )


            print(
                "\nARGUMENTS:"
            )


            print(
                item.arguments
            )


            # ------------------------------------------
            # PARSE MODEL-GENERATED ARGUMENTS
            # ------------------------------------------


            try:


                arguments = json.loads(

                    item.arguments

                )


            except json.JSONDecodeError as error:


                print(

                    "\nERROR:"
                )


                print(

                    "The model returned invalid JSON arguments."
                )


                print(error)


                raise


            # ------------------------------------------
            # EXECUTE REAL PYTHON FUNCTION
            # ------------------------------------------


            try:


                result = execute_tool(

                    tool_name=item.name,

                    arguments=arguments

                )


            except Exception as error:


                result = {

                    "error": str(error)

                }


            print(
                "\nOBSERVATION:"
            )


            print(
                result
            )


            # ------------------------------------------
            # CREATE TOOL OUTPUT MESSAGE
            # ------------------------------------------


            tool_outputs.append(

                {

                    "type":
                        "function_call_output",

                    "call_id":
                        item.call_id,

                    "output":
                        json.dumps(result)

                }

            )


        # ==================================================
        # UPDATE LOCAL HISTORY
        #
        # 1. Add model's tool requests
        # 2. Add application tool results
        # ==================================================


        input_list += response.output


        input_list += tool_outputs


        # ==================================================
        # CALL MODEL AGAIN
        # ==================================================


        response = client.responses.create(

            model=MODEL,

            instructions=AGENT_INSTRUCTIONS,

            input=input_list,

            tools=tools

        )


        # Return to top of loop

        continue


    # ==================================================
    # CASE 2:
    #
    # NO TOOL CALL
    # BUT VALID FINAL TEXT EXISTS
    # ==================================================


    if (

        response.output_text

        and response.output_text.strip()

    ):


        print(

            "\n\n"
            "========== FINAL ANSWER =========="

        )


        print(

            response.output_text

        )


        break


    # ==================================================
    # CASE 3:
    #
    # NO TOOL CALL
    # AND
    # NO FINAL TEXT
    #
    # THIS IS NOT A SUCCESSFUL FINISH
    # ==================================================


    print(

        "\nWARNING:"
    )


    print(

        "The model returned neither a tool call "
        "nor a usable final text answer."

    )


    print(

        "\nThe application will ask the model "
        "to continue the unfinished task."

    )


    # ==================================================
    # PRESERVE CURRENT MODEL OUTPUT
    # ==================================================


    input_list += response.output


    # ==================================================
    # ADD CONTINUATION MESSAGE
    # ==================================================


    input_list.append(

        {

            "role": "user",

            "content": """
The original task is not complete yet.

Continue working on the original request.

Review the complete conversation history and tool results.

The required trip cost components are:

1. Hotel cost
2. Round-trip travel cost
3. Food cost

Check which of these components are already available.

If a required component is missing,
call the appropriate tool.

After all three individual costs are available,
call calculate_total_cost.

After calculate_total_cost returns its result,
provide the complete final answer.

The final answer must include:

- hotel cost
- travel cost
- food cost
- total trip cost
- budget
- remaining money
- affordability decision

Do not stop without either:

1. Calling the next required tool

OR

2. Providing the complete final answer.
"""

        }

    )


    # ==================================================
    # CALL MODEL AGAIN
    # ==================================================


    response = client.responses.create(

        model=MODEL,

        instructions=AGENT_INSTRUCTIONS,

        input=input_list,

        tools=tools

    )


# ==================================================
# 11. MAX ITERATION SAFETY LIMIT
# ==================================================


else:


    print(

        "\n\n"
        "Agent stopped because the maximum "
        "number of iterations was reached."

    )
