import enum
from dataclasses import dataclass
from typing import Dict, List, Union, Optional


class ValueType(enum.Enum):
    URI = "uri"
    String = "string"
    Integer = "integer"
    Float = "float"


@dataclass
class Value:
    __slots__ = ('type', 'value')
    # uri, string, integer, float
    type: ValueType
    value: Union[str, int, float]

    def is_uri(self):
        return self.type == ValueType.URI

    def as_uri(self):
        if self.type == ValueType.URI:
            return self.value
        assert f"Cannot convert value of type {self.type} to URI"


@dataclass
class Statement:
    value: Value
    qualifiers: Dict[str, List[Value]]


@dataclass
class Entity:
    # id or uri of the entity
    uri: str
    label: str
    description: str
    properties: Dict[str, List[Statement]]

    @property
    def readable_label(self):
        return self.label


