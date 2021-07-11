from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from operator import attrgetter
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Union

import orjson
import sm.misc as M
import sm.outputs as O

from sm.misc import OntNS
from sm_widgets.models import Entity, OntClass, OntProperty, Table, Value
from sm_widgets.services.search import OntologyClassSearch, OntologyPropertySearch
from sm_widgets.widgets.annotator.assistant import AnnotatorAssistant
from sm_widgets.widgets.annotator.session import Session
from sm_widgets.widgets.annotator.tree_ordering import reorder2tree, IndirectDictAccess
from sm_widgets.widgets.base import BaseWidget
from sm_widgets.widgets.slider import Slider


class FilterOp(Enum):
    INCLUDE = "include"
    EXCLUDE = "exclude"


@dataclass
class EntityTypeFilter:
    column_index: int
    # qnode that is the class
    class_uri: str
    operator: FilterOp


@dataclass
class RelFilter:
    # the column we want to filter
    column_index: int
    # the endpoint, source or target or wildcard (both none): string for entity & int for column index
    source_endpoint: Optional[Union[int, str]]
    target_endpoint: Optional[Union[int, str]]
    # [0] is property uri, [1] is qualifier uri
    props: Tuple[str, str]
    operator: FilterOp


class CellTypeFilterOp(Enum):
    HAS_LINK = "hasLink"
    HAS_ENTITY = "hasEntity"
    HAS_NO_LINK = "noLink"
    HAS_NO_ENTITY = "noEntity"


@dataclass
class CellTypeFilter:
    column_index: int
    op: CellTypeFilterOp


class _Annotator(BaseWidget):
    pass


class Annotator(_Annotator):
    def __init__(self,
                 entities: Dict[str, Entity], ontclasses: Dict[str, OntClass], ontprops: Dict[str, OntProperty],
                 prop_type_uri: str, savedir: str, eshost: str,
                 dev: bool = False, assistant: Optional[AnnotatorAssistant] = None, embedded_error_log: bool = False):
        super().__init__("annotator", dev, embedded_error_log=embedded_error_log)

        self.savedir = Path(savedir)
        self.savedir.mkdir(exist_ok=True, parents=True)

        self.entities = entities
        self.ontclasses = ontclasses
        self.ontprops = ontprops

        self.ont_class_search = OntologyClassSearch(eshost)
        self.ont_prop_search = OntologyPropertySearch(eshost)
        self.ontns = OntNS.get_instance()
        self.assistant = assistant
        self.prop_type_uri = prop_type_uri

        self.wdclass_parents: Dict[str, Set[str]] = IndirectDictAccess(self.ontclasses, attrgetter("parents_closure"))
        self.cache_id2label: Dict[str, str] = {}

    def set_data(self, id: str, tbl: Table, sms: List[O.SemanticModel]=None):
        self.wait_for_app_ready(5)
        self.cache_id2label = {}
        infile = M.get_latest_path(self.savedir / id / "version.json")
        if infile is None:
            sms = sms or []
            is_curated = False
            note = ""
        else:
            data = M.deserialize_json(infile)
            assert data['version'] == 2
            sms = [O.SemanticModel.from_json(sm) for sm in data['semantic_models']]
            is_curated = data['is_curated']
            note = data['note']

        self.session = Session(id, is_curated, note, tbl, sms)
        if len(self.session.graphs) == 0:
            self.session.graphs.append(O.SemanticModel())
        self.session.graphs = [self.add_default_nodes_to_sm(sm) for sm in self.session.graphs]

        self.tunnel.send_msg(orjson.dumps([
            {
                "type": "wait_for_client_ready",
            },
            {
                "type": "set_props",
                "props": {
                    "log": {
                        "isCurated": self.session.is_curated,
                        "note": self.session.note,
                    },
                    "table": self.serialize_table_schema(),
                    "graphs": [self.serialize_sm(graph) for graph in self.session.graphs],
                    "entities": {},
                    "assistant": {
                        "id": self.session.table.id
                    },
                    "currentGraphIndex": 0,
                    "wdOntology": {
                        "username": '',
                        "password": ''
                    },
                }
            },
            {
                "type": "exec_func",
                "func": "app.tableFetchData",
                "args": []
            }
        ]).decode())

    def save_annotation(self):
        assert len(self.session.graphs) > 0
        (self.savedir / self.session.id).mkdir(exist_ok=True)
        outfile = M.get_incremental_path(self.savedir / self.session.id / "version.json")
        with open(outfile, "wb") as f:
            f.write(orjson.dumps(self.session.to_json(), option=orjson.OPT_INDENT_2))

    @_Annotator.register_handler("/table")
    def fetch_table_data(self, params: dict):
        table = self.session.table
        start = params['offset']
        end = start + params['limit']
        filters = [
            EntityTypeFilter(
                item['columnId'],
                item['uri'],
                FilterOp(item['op']))
            for item in params['typeFilters']
        ]
        filters += [
            RelFilter(
                int(item['columnId']),
                (item['endpoint'] if isinstance(item['endpoint'], str) else item['endpoint']) if
                    item['direction'] == 'incoming' and item['endpoint'] != "*" else None,
                (item['endpoint'] if isinstance(item['endpoint'], str) else item['endpoint']) if
                    item['direction'] == 'outgoing' and item['endpoint'] != "*" else None,
                (item['pred1'], item['pred2']),
                FilterOp(item['op'])
            )
            for item in params['relFilters']
        ]
        filters += [
            CellTypeFilter(int(column_id), CellTypeFilterOp(op))
            for column_id, op in params['linkFilters'].items()
        ]

        tbl_size = table.size()
        if len(filters) == 0:
            total = tbl_size
            row_index = range(start, min(total, end))
        else:
            includes = set()
            excludes = set()
            for filter in filters:
                ci = filter.column_index

                if isinstance(filter, CellTypeFilter):
                    for ri in range(table.size()):
                        links = table.links[ri][ci]
                        if filter.op == CellTypeFilterOp.HAS_LINK:
                            if len(links) == 0:
                                excludes.add(ri)
                        elif filter.op == CellTypeFilterOp.HAS_NO_LINK:
                            if len(links) > 0:
                                excludes.add(ri)
                        elif filter.op == CellTypeFilterOp.HAS_ENTITY:
                            if all(link.entity is None for link in links):
                                excludes.add(ri)
                        else:
                            assert filter.op == CellTypeFilterOp.HAS_NO_ENTITY
                            if not all(link.entity is None for link in links):
                                excludes.add(ri)
                    continue

                if filter.operator == FilterOp.EXCLUDE:
                    sat_condition = excludes
                else:
                    sat_condition = includes

                if isinstance(filter, RelFilter):
                    if filter.source_endpoint is not None:
                        assert filter.target_endpoint is None
                        # query for source
                        res = self.assistant.get_row_indices(table, filter.source_endpoint, filter.column_index,
                                                             filter.props)
                    elif filter.target_endpoint is not None:
                        assert filter.source_endpoint is None
                        # query for the target
                        res = self.assistant.get_row_indices(table, filter.column_index, filter.target_endpoint,
                                                             filter.props)
                    else:
                        res = set()
                        for ri in range(table.size()):
                            links = table.links[ri][ci]
                            sat = False
                            for link in links:
                                if link.entity is None:
                                    continue
                                qnode = self.entities[link.entity]
                                if filter.props[0] not in qnode.properties:
                                    continue
                                if filter.props[0] == filter.props[1]:
                                    # statement value
                                    sat = True
                                else:
                                    for stmt in qnode.properties[filter.props[0]]:
                                        if filter.props[1] in stmt.qualifiers:
                                            sat = True
                                            break
                                if sat:
                                    break
                            if sat:
                                res.add(ri)
                    sat_condition.update(res)
                else:
                    assert isinstance(filter, EntityTypeFilter)
                    for ri in range(table.size()):
                        links = table.links[ri][ci]
                        sat = False
                        for link in links:
                            if link.entity is None:
                                continue
                            qnode = self.entities[link.entity]
                            for stmt in qnode.properties.get(self.prop_type_uri, []):
                                classid = stmt.value.as_uri()
                                if classid not in self.ontclasses:
                                    continue
                                clsnode = self.ontclasses[classid]
                                # this entity is the same class, or its type is child of the desired class not parents
                                if classid == filter.class_uri or filter.class_uri in clsnode.parents_closure:
                                    sat = True
                                    break
                            if sat:
                                break
                        if sat:
                            sat_condition.add(ri)

            if len(includes) == 0 and len(excludes) > 0:
                includes = self.session.table_records_index
            row_index = sorted(includes.difference(excludes))
            total = len(row_index)
            row_index = row_index[start:min(total, end)]

        rows = []
        for ri in row_index:
            row = []
            for ci in range(len(table.table.columns)):
                row.append(self.serialize_table_cell(ri, ci))

            rows.append({
                "data": row,
                "rowId": ri
            })
        return {"rows": rows, "total": total}

    @_Annotator.register_handler("/entities")
    def get_entity(self, params: dict):
        return {
            uri: self.serialize_entity(uri, full=True)
            for uri in params['uris']
        }
    
    @_Annotator.register_handler("/ontology/class")
    def search_ont_class(self, params: dict):
        return self.ont_class_search.search(params['query'])

    @_Annotator.register_handler("/ontology/predicate")
    def search_ont_property(self, params: dict):
        return self.ont_prop_search.search(params['query'])

    @_Annotator.register_handler("/save")
    def save_annotation_handler(self, params: dict):
        self.session.note = params['note']
        self.session.is_curated = params['isCurated']
        self.session.graphs = [self.deserialize_sm(g['nodes'], g['edges']) for g in params['graphs']]
        self.save_annotation()

    @_Annotator.register_handler("/assistant/column")
    def suggest_column(self, params: dict):
        ci = params['columnIndex']
        table = self.session.table
        column = table.table.columns[ci]

        # compute statistics
        stats = {
            'entities/linked row': [],
            "links/row": [],
            "avg %link surface": [],
            "# rows": table.size(),
        }
        qnode2types = {}
        for ri, value in enumerate(column.values):
            nchars = len(value)
            links = table.links[ri][ci]
            n_covered_chars = sum(l.end - l.start for l in links)
            n_qnodes = sum(l.entity is not None for l in links)
            if len(links) > 0:
                stats['entities/linked row'].append(n_qnodes)
            stats['links/row'].append(len(links))
            stats['avg %link surface'].append(n_covered_chars / max(nchars, 1e-7))

            for l in links:
                if l.entity is None:
                    continue
                qnode = self.entities[l.entity]
                qnode2types[l.entity] = list({stmt.value.as_uri() for stmt in qnode.properties.get(self.prop_type_uri, [])})

        stats['# unique qnodes'] = len(qnode2types)
        for k, v in stats.items():
            if isinstance(v, list):
                if len(v) == 0:
                    stats[k] = "0%"
                else:
                    stats[k] = f"{sum(v) / max(len(v), 1e-7):.2f}%"

        supertype_freq = defaultdict(int)

        for lst in qnode2types.values():
            etypes = set()
            for item in lst:
                etypes.add(item)
                etypes.update(self.ontclasses[item].parents_closure)
            for c in etypes:
                supertype_freq[c] += 1

        flat_type_hierarchy = reorder2tree(list(supertype_freq.keys()), self.wdclass_parents)
        qnode2children = {}

        def traversal(tree, path):
            if tree.id not in qnode2children:
                qnode2children[tree.id] = set()
            for parent_tree in path:
                qnode2children[parent_tree.id].add(tree.id)

        flat_type_hierarchy.preorder(traversal)

        flat_type_hierarchy = flat_type_hierarchy.update_score(lambda n: supertype_freq[n.id]).sort(reverse=True)
        flat_type_hierarchy = [
            {
                "uri": u.id,
                "label": self.get_entity_label(u.id),
                "duplicated": u.duplicated,
                "depth": u.depth,
                "freq": supertype_freq[u.id],
                "percentage": supertype_freq[u.id] / len(qnode2types)
            }
            for u in flat_type_hierarchy.get_flatten_hierarchy(dedup=True)
        ]

        resp = {
            "stats": stats,
            "flattenTypeHierarchy": flat_type_hierarchy,
            "type2children": {k: list(v) for k, v in qnode2children.items()},
        }

        if self.assistant is not None:
            relationships = self.assistant.get_column_relationships(table, ci)
            if relationships is not None:
                for rels in relationships.values():
                    for i, rel in enumerate(rels):
                        rels[i] = dict(
                            endpoint=rel.endpoint,
                            predicates=rel.predicates,
                            freq=rel.freq,
                        )
                resp['relationships'] = relationships
        return resp

    def add_default_nodes_to_sm(self, sm: O.SemanticModel):
        table = self.session.table
        column2name = self.session.column2name
        for col in table.table.columns:
            if not any(n.is_data_node and n.col_index == col.index for n in sm.iter_nodes()):
                # no column in the model, add missing columns
                dnodeid = f'd-{col.index}'
                assert not sm.has_node(dnodeid)
                sm.add_node(O.DataNode(f"d-{col.index}", col.index, column2name[col.index]))

        # if table.context.page_qnode is not None:
        #     context_nodeid = f"context-{table.context.page_qnode}"
        #     sm.add_node(O.LiteralNode(
        #         context_nodeid,
        #         WDOnt.get_qnode_uri(table.context.page_qnode),
        #         f"{self.entities[table.context.page_qnode].label} ({table.context.page_qnode})",
        #         True,
        #         O.LiteralNodeDataType.Entity
        #     ))
        return sm

    def deserialize_sm(self, nodes: List[dict], edges: List[dict]) -> O.SemanticModel:
        sm = O.SemanticModel()
        for n in nodes:
            if n['isDataNode']:
                sm.add_node(O.DataNode(n['id'], n['columnId'], self.session.column2name[n['columnId']]))
            elif n['isClassNode']:
                rel_uri = self.ontns.get_rel_uri(n['uri'])
                sm.add_node(
                    O.ClassNode(n['id'], n['uri'], rel_uri, n['approximation']))
            else:
                assert n['isLiteralNode']
                sm.add_node(
                    O.LiteralNode(n['id'], n['uri'], n['label'], n['isInContext'], O.LiteralNodeDataType(n['datatype'])))

        for e in edges:
            rel_uri = self.ontns.get_rel_uri(e['uri'])
            sm.add_edge(O.Edge(e['source'], e['target'], e['uri'], rel_uri, e['approximation']))
        return sm

    def serialize_sm(self, sm: O.SemanticModel):
        nodes = []
        for n in sm.iter_nodes():
            if n.is_class_node:
                nodes.append({
                    "id": n.id,
                    "uri": n.abs_uri,
                    "label": self.get_entity_label(n.abs_uri) or n.rel_uri,
                    "approximation": n.approximation,
                    "isClassNode": True,
                    "isDataNode": False,
                    "isLiteralNode": False,
                })
            elif n.is_data_node:
                nodes.append({
                    "id": n.id,
                    "label": self.session.column2name[n.col_index],
                    "isClassNode": False,
                    "isDataNode": True,
                    "isLiteralNode": False,
                    "columnId": n.col_index,
                })
            else:
                nodes.append({
                    "id": n.id,
                    "uri": n.value if n.datatype == O.LiteralNodeDataType.Entity else "",
                    "label": n.label,
                    "isClassNode": False,
                    "isDataNode": False,
                    "isLiteralNode": True,
                    "isInContext": n.is_in_context,
                    "datatype": n.datatype.value
                })

        return {
            "tableID": self.session.table.id,
            "nodes": nodes,
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "uri": e.abs_uri,
                    "label": self.get_entity_label(e.abs_uri) or e.rel_uri,
                    "approximation": e.approximation
                }
                for e in sm.iter_edges()
            ]
        }

    def serialize_table_schema(self):
        """Get schema of the table"""
        table = self.session.table
        columns = [{"title": "", "dataIndex": "rowId"}]
        for col in table.table.columns:
            columns.append({
                "title": col.name,
                "columnId": col.index,
                "dataIndex": ["data", col.index],
            })
        return {
            "id": table.id,
            "columns": columns,
            "rowKey": "rowId",
            "totalRecords": table.size(),
            "metadata": {
                # "title": table.context.page_title,
                # "url": table.context.page_url,
                # "entity": {"uri": WDOnt.get_qnode_uri(table.context.page_qnode), "label": self.get_entity_label(table.context.page_qnode)} if table.context.page_qnode is not None else None
                "title": "",
                "url": "",
                "entity": None
            }
        }

    def serialize_table_cell(self, ri: int, ci: int):
        table = self.session.table
        value = table.table.columns[ci].values[ri]
        qnodes_metadata = {}
        for link in table.links[ri][ci]:
            if link.entity is not None:
                qnodes_metadata[link.entity] = self.serialize_entity(link.entity)

        return {
            "value": value,
            "links": [
                {
                    "start": link.start, "end": link.end,
                    "href": link.url,
                    "entity": link.entity
                }
                for link in table.links[ri][ci]
            ],
            "metadata": {
                "entities": qnodes_metadata
            }
        }

    def serialize_entity(self, uri: str, full: bool = False):
        wdclass_parents = self.wdclass_parents
        qnode = self.entities[uri]
        ent = {
            "uri": uri,
            "label": str(self.get_entity_label(uri)),
        }
        try:
            # get hierarchy
            if self.prop_type_uri in qnode.properties:
                forest = reorder2tree(
                    [stmt.value.as_uri() for stmt in qnode.properties.get(self.prop_type_uri, [])],
                    wdclass_parents)
                hierarchy = [{"uri": x.id,
                              "label": self.get_entity_label(x.id),
                              "depth": x.depth}
                             for x in forest.get_flatten_hierarchy()]
            else:
                hierarchy = []

            ent["types"] = hierarchy
            if not full:
                return ent

            props = {}
            for puri, stmts in qnode.properties.items():
                ser_stmts = []
                for stmt in stmts:
                    ser_stmts.append({
                        "value": self.serialize_datavalue(stmt.value),
                        "qualifiers": {
                            qid: {
                                "uri": qid,
                                "label": self.get_entity_label(qid),
                                "values": [self.serialize_datavalue(qual) for qual in quals]
                            }
                            for qid, quals in stmt.qualifiers.items()
                        }
                    })

                props[puri] = {
                    "uri": puri,
                    "label": self.get_entity_label(puri),
                    "values": ser_stmts
                }
            ent['props'] = props
            ent['description'] = str(qnode.description)
        except KeyError:
            print(f"Error while obtaining information of a qnode: {uri}")
            raise

        return ent

    def serialize_datavalue(self, val: Value):
        if val.is_uri():
            uri = val.as_uri()
            # an exception that they sometime has property instead of qnode id???
            # this is probably an error, but we need to make this code robust
            label = self.get_entity_label(uri)
            return {"uri": uri, "label": label}
        return val.value

    def get_entity_label(self, uri: str):
        if uri not in self.cache_id2label:
            if uri in self.entities:
                label = self.entities[uri].readable_label
            elif uri in self.ontclasses:
                label = self.ontclasses[uri].readable_label
            elif uri in self.ontprops:
                label = self.ontprops[uri].readable_label
            else:
                label = None

            self.cache_id2label[uri] = label
        return self.cache_id2label[uri]


class BatchAnnotator(Slider):

    def __init__(self, annotator: Annotator, dev: bool = False):
        super().__init__(annotator, annotator.set_data, dev)

    def set_data(self, tables_with_ids: List[Tuple[str, str, Table, Optional[List[O.SemanticModel]]]], start_index: int=0):
        super().set_data([
            dict(description=description, args=(table_id, table, sms))
            for table_id, description, table, sms in tables_with_ids
        ], start_index)
