from collections import Counter
from typing import List

from sm_widgets.models import Table
from sm.outputs import SemanticModel

class Session:

    def __init__(self, id: str, is_curated: bool, note: str, table: Table, graphs: List[SemanticModel]):
        self.id = id
        self.is_curated = is_curated
        self.note = note
        self.table = table
        self.graphs = graphs

        # use it for querying records
        self.table_records_index = set(range(table.size()))
        self.column2name = {}
        cname2freq = Counter(c.name for c in table.table.columns)
        for c in table.table.columns:
            if cname2freq[c.name] > 1:
                self.column2name[c.index] = f"{c.name} ({c.index})"
            else:
                self.column2name[c.index] = c.name

    def to_json(self):
        return {
            "version": 2,
            "table_id": self.table.id,
            "semantic_models": [sm.to_dict() for sm in self.graphs],
            "is_curated": self.is_curated,
            "note": self.note,
        }