from dataclasses import dataclass, field

@dataclass
class Operation:
    name: str
    params: dict = field(default_factory=dict)

    def get(self, key, default=None):
        return self.params.get(key, default)