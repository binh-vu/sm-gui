from dataclasses import dataclass

from sm.misc import OntNS

from kgdata.wikidata.models import QNode, WDClass, WDProperty, DataValue
import kgdata.wikidata.db as kg_db
from sm_widgets.models import Entity, Value, OntClass, OntProperty
from sm_widgets.models.entity import ValueType, Statement
from sm_widgets_integration.common import StoreWrapper


class GramsIntFn:
    ontns = OntNS.get_instance()

    @dataclass
    class WrapperQNode(Entity):
        id: str

        @property
        def readable_label(self):
            return f"{self.label} ({self.id})"

    @dataclass
    class WrapperWDClass(OntClass):
        id: str = None

        @property
        def readable_label(self):
            return f"{self.label} ({self.id})"

    @dataclass
    class WrapperWDProperty(OntProperty):
        id: str = None

        @property
        def readable_label(self):
            return f"{self.label} ({self.id})"

    @staticmethod
    def key_deser(uri: str):
        if uri.startswith("http://wikidata.org/"):
            uri = uri.replace("http://wikidata.org/entity/", "")
            uri = uri.replace("http://wikidata.org/prop/", "")
        return uri

    @staticmethod
    def qnode_deser(qnode: QNode):
        props = {}
        for pid, stmts in qnode.props.items():
            new_stmts = []
            for stmt in stmts:
                new_stmt = Statement(value=GramsIntFn.wd_value_deser(stmt.value),
                                     qualifiers={
                                         f'http://wikidata.org/prop/{qid}': [GramsIntFn.wd_value_deser(x) for x in lst]
                                         for qid, lst in stmt.qualifiers.items()
                                     })
                new_stmts.append(new_stmt)
            props[f'http://wikidata.org/prop/{pid}'] = new_stmts
        return GramsIntFn.WrapperQNode(
            id=qnode.id,
            uri=f"http://wikidata.org/entity/{qnode.id}",
            label=qnode.label,
            description=qnode.description,
            properties=props
        )

    @staticmethod
    def ont_class_deser(item: WDClass):
        parents = [f"http://wikidata.org/entity/{p}" for p in item.parents]
        parents_closure = {
            f"http://wikidata.org/entity/{p}" for p in item.parents_closure
        }
        return GramsIntFn.WrapperWDClass(id=item.id, uri=item.get_uri(), aliases=item.aliases,
                                         label=item.label, description=item.description,
                                         parents=parents, parents_closure=parents_closure)

    @staticmethod
    def ont_prop_deser(item: WDProperty):
        parents = [f"http://wikidata.org/entity/{p}" for p in item.parents]
        parents_closure = {
            f"http://wikidata.org/entity/{p}" for p in item.parents_closure
        }
        return GramsIntFn.WrapperWDProperty(id=item.id, uri=item.get_uri(), aliases=item.aliases,
                                            label=item.label, description=item.description,
                                            parents=parents, parents_closure=parents_closure)

    @staticmethod
    def wd_value_deser(val: DataValue):
        if val.is_qnode():
            ent_id = val.as_qnode_id()
            if ent_id.startswith("Q"):
                uri = f"http://wikidata.org/entity/{ent_id}"
            elif ent_id.startswith("P"):
                uri = f"http://wikidata.org/prop/{ent_id}"
            else:
                uri = ent_id
            return Value(type=ValueType.URI, value=uri)
        if val.is_quantity():
            return Value(type=ValueType.Float, value=val.value['amount'])
        return Value(type=ValueType.String, value=val.to_string_repr())


def get_qnode_db(dbfile: str, read_only=False, proxy: bool=False):
    db = kg_db.get_qnode_db(dbfile, read_only=read_only, proxy=proxy)
    return StoreWrapper(db, GramsIntFn.key_deser, GramsIntFn.qnode_deser)


def get_ontclass_db(dbfile: str, read_only=False, proxy: bool=False):
    db = kg_db.get_wdclass_db(dbfile, read_only=read_only, proxy=proxy)
    return StoreWrapper(db, GramsIntFn.key_deser, GramsIntFn.ont_class_deser)


def get_ontprop_db(dbfile: str, read_only=False, proxy: bool=False):
    db = kg_db.get_wdprop_db(dbfile, read_only=read_only, proxy=proxy)
    return StoreWrapper(db, GramsIntFn.key_deser, GramsIntFn.ont_prop_deser)


