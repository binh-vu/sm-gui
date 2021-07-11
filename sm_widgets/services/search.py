from typing import List, Dict

from elasticsearch import Elasticsearch, helpers
import requests
from tqdm.auto import tqdm


class ElasticSearchService:
    def __init__(self, eshost: str, index: str):
        self.eshost = eshost
        self.index = index
        self.fields = ['id', 'label', 'uri', 'readable_label', 'aliases', 'description']
        self.search_fields = ['id^10', 'label^5', 'aliases^3', 'description']

    def search(self, query, top_k: int = 20):
        resp = requests.post(f"{self.eshost}/{self.index}/_search", json={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": self.search_fields
                }
            }
        })
        assert resp.status_code == 200, resp.status_code
        data = resp.json()
        docs = []
        for r in data['hits']['hits'][:top_k]:
            node = r['_source']
            docs.append({
                "uri": node['uri'],
                "label": node['readable_label'],
                "description": node['description'],
                "score": r['_score']
            })
        return docs

    def load2index(self, documents: List[dict], batch_size: int = 128):
        self.create_index()
        es = Elasticsearch([self.eshost])
        docs = [
            dict(_id=doc['id'], **{k: doc[k] for k in self.fields})
            for doc in documents
        ]
        for i in tqdm(range(0, len(docs), batch_size)):
            helpers.bulk(es, docs[i:i + batch_size], index=self.index)

    def create_index(self):
        index_settings = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "keyword_analyzer": {
                            "tokenizer": "standard",
                            "filter": ["lowercase"]
                        },
                        "default": {
                            "tokenizer": "standard",
                            "filter": ["lowercase", "2_16_edgegrams"]
                        },
                        "default_search": {
                            "tokenizer": "standard",
                            "filter": ["lowercase", "16_truncate"]
                        },
                    },
                    "filter": {
                        "2_16_edgegrams": {
                            "type": "edge_ngram",
                            "min_gram": 2,
                            "max_gram": 16
                        },
                        "16_truncate": {
                            "type": "truncate",
                            "length": 16
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "id": {
                        "type": "text",
                        "analyzer": "keyword_analyzer",
                    }
                }
            }
        }

        resp = requests.put(f"{self.eshost}/{self.index}", json=index_settings)
        assert resp.status_code == 200, resp.text


class OntologyClassSearch(ElasticSearchService):
    def __init__(self, eshost: str):
        super().__init__(eshost, "ontology-classes")


class OntologyPropertySearch(ElasticSearchService):
    def __init__(self, eshost: str):
        super().__init__(eshost, "ontology-properties")


