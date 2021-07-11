from dataclasses import dataclass
from pathlib import Path

from rdflib import RDFS

from sm.misc import OntNS

from kgdata.wikidata.models import QNode, WDClass, WDProperty, DataValue
import kgdata.wikidata.db as kg_db
from sm_widgets.models import Entity, Value, OntClass, OntProperty
from sm_widgets.models.entity import ValueType, Statement
from sm_widgets.services.search import OntologyClassSearch, OntologyPropertySearch
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
        if uri.startswith("http://www.wikidata.org/"):
            uri = uri.replace("http://www.wikidata.org/entity/", "")
            uri = uri.replace("http://www.wikidata.org/prop/", "")
        elif uri.startswith("http://wikidata.org/"):
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
                                         f'http://www.wikidata.org/prop/{qid}': [GramsIntFn.wd_value_deser(x) for x in lst]
                                         for qid, lst in stmt.qualifiers.items()
                                     })
                new_stmts.append(new_stmt)
            props[f'http://www.wikidata.org/prop/{pid}'] = new_stmts
        return GramsIntFn.WrapperQNode(
            id=qnode.id,
            uri=f"http://www.wikidata.org/entity/{qnode.id}",
            label=qnode.label,
            description=qnode.description,
            properties=props
        )

    @staticmethod
    def ont_class_deser(item: WDClass):
        parents = [f"http://www.wikidata.org/entity/{p}" for p in item.parents]
        parents_closure = {
            f"http://www.wikidata.org/entity/{p}" for p in item.parents_closure
        }
        return GramsIntFn.WrapperWDClass(id=item.id, uri=item.get_uri(), aliases=item.aliases,
                                         label=item.label, description=item.description,
                                         parents=parents, parents_closure=parents_closure)

    @staticmethod
    def ont_prop_deser(item: WDProperty):
        parents = [f"http://www.wikidata.org/entity/{p}" for p in item.parents]
        parents_closure = {
            f"http://www.wikidata.org/entity/{p}" for p in item.parents_closure
        }
        return GramsIntFn.WrapperWDProperty(id=item.id, uri=item.get_uri(), aliases=item.aliases,
                                            label=item.label, description=item.description,
                                            parents=parents, parents_closure=parents_closure)

    @staticmethod
    def wd_value_deser(val: DataValue):
        if val.is_qnode():
            ent_id = val.as_qnode_id()
            if ent_id.startswith("Q"):
                uri = f"http://www.wikidata.org/entity/{ent_id}"
            elif ent_id.startswith("P"):
                uri = f"http://www.wikidata.org/prop/{ent_id}"
            else:
                uri = ent_id
            return Value(type=ValueType.URI, value=uri)
        if val.is_quantity():
            return Value(type=ValueType.Float, value=val.value['amount'])
        return Value(type=ValueType.String, value=val.to_string_repr())


def get_qnode_db(db_or_dbfile, read_only=False, proxy: bool=False):
    if isinstance(db_or_dbfile, (str, Path)):
        db = kg_db.get_qnode_db(db_or_dbfile, read_only=read_only, proxy=proxy)
    else:
        db = db_or_dbfile

    return StoreWrapper(db, GramsIntFn.key_deser, GramsIntFn.qnode_deser)


def get_ontclass_db(db_or_dbfile: str, read_only=False, proxy: bool=False):
    if isinstance(db_or_dbfile, (str, Path)):
        db = kg_db.get_wdclass_db(db_or_dbfile, read_only=read_only, proxy=proxy)
    else:
        db = db_or_dbfile

    return StoreWrapper(db, GramsIntFn.key_deser, GramsIntFn.ont_class_deser)


def get_ontprop_db(db_or_dbfile: str, read_only=False, proxy: bool=False):
    if isinstance(db_or_dbfile, (str, Path)):
        db = kg_db.get_wdprop_db(db_or_dbfile, read_only=read_only, proxy=proxy)
    else:
        db = db_or_dbfile

    return StoreWrapper(db, GramsIntFn.key_deser, GramsIntFn.ont_prop_deser)


def index_ontology(eshost: str):
    wdprops = WDProperty.from_file()
    docs = [
        {
            "id": wdprop.id,
            "uri": wdprop.get_uri(),
            "label": str(wdprop.label),
            "readable_label": f"{wdprop.label} ({wdprop.id})",
            "aliases": [str(x) for x in wdprop.aliases],
            "description": str(wdprop.description)
        }
        for wdprop in wdprops.values()
    ]
    docs.append({
        "id": str(RDFS.label),
        "uri": str(RDFS.label),
        "label": "rdfs:label",
        "readable_label": "rdfs:label",
        "aliases": ["label"],
        "description": ["label of a resource"]
    })
    OntologyPropertySearch(eshost).load2index(docs)

    wdclasses = WDClass.from_file()
    docs = [
        {
            "id": wdclass.id,
            "uri": wdclass.get_uri(),
            "label": str(wdclass.label),
            "readable_label": f"{wdclass.label} ({wdclass.id})",
            "aliases": [str(x) for x in wdclass.aliases],
            "description": str(wdclass.description)
        }
        for wdclass in wdclasses.values()
    ]
    docs.append({
        "id": "http://wikiba.se/ontology#Statement",
        "uri": "http://wikiba.se/ontology#Statement",
        "label": "wikibase:Statement",
        "readable_label": "wikibase:Statement",
        "aliases": ["statement"],
        "description": "Wikidata Statement"
    })
    OntologyClassSearch(eshost).load2index(docs)


if __name__ == '__main__':
    # db = get_qnode_db("/workspace/sm-dev/grams/data/qnodes.db", proxy=True)
    # url = 'http://www.w3.org/2000/01/rdf-schema#label'
    # print(url in db)
    # print(db.get(url, None))
    index_ontology("http://localhost:9200")