from dataclasses import dataclass

from typing import Optional, TypedDict, Union, List, Tuple

from sm.misc import OntNS

from sm_widgets.models import Table

Resource = TypedDict("Resource", dict(uri=str, label=str))


@dataclass
class ColumnRelationship:
    # dict for entity, number for column index
    endpoint: Union[Resource, int]
    predicates: Tuple[Resource, Resource]
    freq: int


ColumnRelationshipResult = TypedDict("ColumnRelationshipResult",
                                     dict(incoming=List[ColumnRelationship], outgoing=List[ColumnRelationship]))


# noinspection PyMethodMayBeStatic
class AnnotatorAssistant:

    def get_row_indices(self, table: Table, source_node: Union[int, str], target_node: Union[int, str],
                        links: Tuple[str, str]):
        # return the list of indices
        return set()

    def get_column_relationships(self, table: Table, column_index: str) -> Optional[ColumnRelationshipResult]:
        # get relationships of columns, should return {
        # "incoming": {endpoint, endpoint_type, predicates, freq}[]
        # "outgoing": {endpoint, endpoint_type, predicates, freq}[]
        # }
        return None


class DummyAnnotatorAssistant(AnnotatorAssistant):

    def __init__(self):
        self.ontns = OntNS.get_instance()

    def get_column_relationships(self, table: Table, column_index: str) -> Optional[ColumnRelationshipResult]:
        return {
            "incoming": [
                ColumnRelationship(
                    endpoint=0,
                    predicates=(
                        {"uri": self.ontns.get_abs_uri("p:P585"), "label": "Point in time (P585)"},),
                    freq=10
                ),
                ColumnRelationship(
                    endpoint=0,
                    predicates=({"uri": self.ontns.get_abs_uri("p:P710"), "label": "participant (P710)"},),
                    freq=5
                ),
                ColumnRelationship(
                    endpoint={"uri": self.ontns.get_abs_uri("wd:Q5"), "label": "United States (Q5)"},
                    predicates=({"uri": self.ontns.get_abs_uri("p:P585"), "label": "Point in time (P585)"},),
                    freq=10
                )
            ],
            "outgoing": [
                ColumnRelationship(
                    endpoint=0,
                    predicates=({"uri": self.ontns.get_abs_uri("p:P585"), "label": "Point in time (P585)"},),
                    freq=10
                ),
                ColumnRelationship(
                    endpoint=0,
                    predicates=({"uri": self.ontns.get_abs_uri("p:P710"), "label": "participant (P710)"},),
                    freq=5
                ),
                ColumnRelationship(
                    endpoint={"uri": self.ontns.get_abs_uri("wd:Q5"), "label": "United States (Q5)"},
                    predicates=({"uri": self.ontns.get_abs_uri("p:P585"), "label": "Point in time (P585)"},),
                    freq=10
                )
            ]
        }


