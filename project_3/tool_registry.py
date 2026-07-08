import json


class ToolRegistry:

    def __init__(self):

        self.tools = {}


    def register(
        self,
        name,
        description,
        parameters,
        function
    ):

        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "function": function,
        }


    def list_tools(self):

        result = []

        for name, tool in self.tools.items():

            result.append(
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                }
            )

        return result


    def get_openai_tool_schemas(self):

        schemas = []

        for name, tool in self.tools.items():

            schemas.append(
                {
                    "type": "function",
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                }
            )

        return schemas


    def execute(
        self,
        name,
        arguments
    ):

        if name not in self.tools:

            raise ValueError(
                f"Tool not found: {name}"
            )

        function = self.tools[name]["function"]

        return function(**arguments)
