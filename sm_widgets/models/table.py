from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import orjson

from sm.inputs import ColumnBasedTable, Column
import sm.misc as M
from sm_widgets.models.entity import Value


@dataclass
class ContextLevel:
    level: int
    header: str
    content: str

    def clone(self):
        return ContextLevel(self.level, self.header, self.content)


@dataclass
class Link:
    start: int
    end: int
    url: str
    # uri of entity
    entity: Optional[str]


@dataclass
class Table:
    table: ColumnBasedTable
    # list of values in the context
    context_values: List[Value]
    context_tree: List[ContextLevel]

    links: List[List[List[Link]]]

    @property
    def id(self):
        return self.table.table_id

    def size(self):
        if len(self.table.columns) == 0:
            return 0
        return len(self.table.columns[0].values)

    @staticmethod
    def from_csv_file(infile: Union[Path, str], first_row_header: bool = True, table_id: Optional[str] = None):
        infile = Path(infile)
        link_file = infile.parent / f"{infile.stem}.links.tsv"

        if table_id is None:
            table_id = infile.stem
        rows = M.deserialize_csv(infile)

        assert len(rows) > 0, "Empty table"
        columns = []
        if first_row_header:
            headers = rows[0]
            rows = rows[1:]
        else:
            headers = [f'column-{i:03d}' for i in range(len(rows[0]))]

        for ci, cname in enumerate(headers):
            columns.append(Column(ci, cname, [r[ci] for r in rows]))
        table = ColumnBasedTable(table_id, columns)
        links = []
        for ri in range(len(rows)):
            links.append([[] for ci in range(len(headers))])

        if link_file.exists():
            for row in M.deserialize_csv(link_file, delimiter="\t"):
                ri, ci, ents = int(row[0]), int(row[1]), row[2:]
                for ent in ents:
                    if ent.startswith("{"):
                        # it's json, encoding the hyperlinks
                        link = Link(**orjson.loads(ent))
                    else:
                        link = Link(0, len(table.columns[ci][ri]), ent, ent)
                    links[ri][ci].append(link)
        return Table(table, context_values=[], context_tree=[], links=links)
