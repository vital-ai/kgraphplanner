
class ToolResponse:
    def __init__(self, parameters=None):
        if parameters is None:
            self.parameters = {}
        else:
            self.parameters = parameters

    def add_parameter(self, key, value):
        self.parameters[key] = value

    def get_parameter(self, key):
        return self.parameters.get(key, None)

    def remove_parameter(self, key):
        if key in self.parameters:
            del self.parameters[key]
        else:
            pass

    def all_parameters(self):
        return self.parameters.copy()
