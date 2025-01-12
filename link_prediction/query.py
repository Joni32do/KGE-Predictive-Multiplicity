class Query():
    def __init__(self, value: str, relation: str, head_is_missing: bool):
        self.value = value
        self.relation = relation
        self.head_is_missing = head_is_missing # otherwise tail is missing

    def fill_in_missing_value(self, missing_value: str):
        if self.head_is_missing:
            return (missing_value, self.relation, self.value)
        else:
            return (self.value, self.relation, missing_value)
        
    def __str__(self):
        if self.head_is_missing:
            return "(?, " + self.relation + ", " + self.value + ")"
        else:
            return "(" + self.value + ", " + self.relation + ", ?)"
        
    def __repr__(self):
        return str(self)