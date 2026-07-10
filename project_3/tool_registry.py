class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, name, description, parameters, function):
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "function": function,
        }

    def schemas(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in self.tools.values()
        ]

    def execute(self, name, arguments):
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")
        return self.tools[name]["function"](**arguments)
class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, name, description, parameters, function):
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "function": function,
        }

    def schemas(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in self.tools.values()
        ]

    def execute(self, name, arguments):
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")
        return self.tools[name]["function"](**arguments)
