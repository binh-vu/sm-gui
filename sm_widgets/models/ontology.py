from dataclasses import dataclass, field
from typing import List, Set, Optional


@dataclass
class OntClass:
    uri: str
    label: str
    aliases: List[str]
    description: str
    parents: List[str]
    parents_closure: Set[str] = field(default_factory=set)

    @property
    def readable_label(self):
        return self.label


@dataclass
class OntProperty:
    uri: str
    label: str
    aliases: List[str]
    description: str
    parents: List[str]
    parents_closure: Set[str] = field(default_factory=set)

    @property
    def readable_label(self):
        return self.label