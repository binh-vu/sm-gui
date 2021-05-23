export const history = [
    {
        "type": "on_receive",
        "version": 2,
        "msg": "{\"type\":\"response\",\"id\":\"/init/done\",\"success\":true,\"response\":null}"
    },
    {
        "type": "on_receive",
        "version": 3,
        "msg": "[{\"type\":\"wait_for_client_ready\"},{\"type\":\"set_props\",\"props\":{\"log\":{\"isCurated\":false,\"note\":\"\"},\"table\":{\"id\":\"table_03\",\"columns\":[{\"title\":\"\",\"dataIndex\":\"rowId\"},{\"title\":\"Disease\",\"columnId\":0,\"dataIndex\":[\"data\",0]},{\"title\":\"Gene\",\"columnId\":1,\"dataIndex\":[\"data\",1]},{\"title\":\"Inheritance\",\"columnId\":2,\"dataIndex\":[\"data\",2]},{\"title\":\"Clinical Features\",\"columnId\":3,\"dataIndex\":[\"data\",3]}],\"rowKey\":\"rowId\",\"totalRecords\":3,\"metadata\":{\"title\":\"\",\"url\":\"\",\"entity\":null}},\"graphs\":[{\"tableID\":\"table_03\",\"nodes\":[{\"id\":\"d-0\",\"label\":\"Disease\",\"isClassNode\":false,\"isDataNode\":true,\"isLiteralNode\":false,\"columnId\":0},{\"id\":\"d-1\",\"label\":\"Gene\",\"isClassNode\":false,\"isDataNode\":true,\"isLiteralNode\":false,\"columnId\":1},{\"id\":\"d-2\",\"label\":\"Inheritance\",\"isClassNode\":false,\"isDataNode\":true,\"isLiteralNode\":false,\"columnId\":2},{\"id\":\"d-3\",\"label\":\"Clinical Features\",\"isClassNode\":false,\"isDataNode\":true,\"isLiteralNode\":false,\"columnId\":3}],\"edges\":[]}],\"entities\":{},\"assistant\":{\"id\":\"table_03\"},\"currentGraphIndex\":0,\"wdOntology\":{\"username\":\"\",\"password\":\"\"}}},{\"type\":\"exec_func\",\"func\":\"app.tableFetchData\",\"args\":[]}]"
    },
    {
        "type": "send_msg",
        "msg": "{\"id\":\"0\",\"url\":\"/table\",\"params\":{\"offset\":0,\"limit\":5,\"typeFilters\":[],\"relFilters\":[],\"linkFilters\":{}}}",
        "version": 3
    },
    {
        "type": "on_receive",
        "version": 4,
        "msg": "{\"type\":\"response\",\"id\":\"0\",\"success\":true,\"response\":{\"rows\":[{\"data\":[{\"value\":\"Achondroplasia\",\"links\":[{\"start\":0,\"end\":14,\"href\":\"http://www.wikidata.org/entity/Q340594\",\"entity\":\"http://wikidata.org/entity/Q340594\"}],\"metadata\":{\"entities\":{\"http://wikidata.org/entity/Q340594\":{\"uri\":\"http://wikidata.org/entity/Q340594\",\"label\":\"achondroplasia (Q340594)\",\"types\":[{\"uri\":\"http://wikidata.org/entity/Q42303753\",\"label\":\"designated intractable/rare diseases (Q42303753)\",\"depth\":0},{\"uri\":\"http://wikidata.org/entity/Q929833\",\"label\":\"rare disease (Q929833)\",\"depth\":0}]}}}},{\"value\":\"Fibroblast growth factor receptor 3 (FGR3)\",\"links\":[{\"start\":0,\"end\":42,\"href\":\"http://www.wikidata.org/entity/Q14914358\",\"entity\":\"http://wikidata.org/entity/Q14914358\"}],\"metadata\":{\"entities\":{\"http://wikidata.org/entity/Q14914358\":{\"uri\":\"http://wikidata.org/entity/Q14914358\",\"label\":\"FGFR3 (Q14914358)\",\"types\":[{\"uri\":\"http://wikidata.org/entity/Q7187\",\"label\":\"gene (Q7187)\",\"depth\":0}]}}}},{\"value\":\"Autosomal dominant (normal parents can have an affected child due to new mutation, and risk of recurrence in subsequent children is low)\",\"links\":[],\"metadata\":{\"entities\":{}}},{\"value\":\"Short limbs relative to trunk, prominent forehead, low nasal root, redundant skin folds on arms and legs\",\"links\":[],\"metadata\":{\"entities\":{}}}],\"rowId\":0},{\"data\":[{\"value\":\"Cystic Fibrosis\",\"links\":[{\"start\":0,\"end\":15,\"href\":\"http://www.wikidata.org/entity/Q178194\",\"entity\":\"http://wikidata.org/entity/Q178194\"}],\"metadata\":{\"entities\":{\"http://wikidata.org/entity/Q178194\":{\"uri\":\"http://wikidata.org/entity/Q178194\",\"label\":\"cystic fibrosis (Q178194)\",\"types\":[{\"uri\":\"http://wikidata.org/entity/Q42303753\",\"label\":\"designated intractable/rare diseases (Q42303753)\",\"depth\":0}]}}}},{\"value\":\"Cystic fibrosis transmembrane regulator (CFTR)\",\"links\":[{\"start\":0,\"end\":46,\"href\":\"http://www.wikidata.org/entity/Q14864712\",\"entity\":\"http://wikidata.org/entity/Q14864712\"}],\"metadata\":{\"entities\":{\"http://wikidata.org/entity/Q14864712\":{\"uri\":\"http://wikidata.org/entity/Q14864712\",\"label\":\"CFTR (Q14864712)\",\"types\":[{\"uri\":\"http://wikidata.org/entity/Q42303753\",\"label\":\"designated intractable/rare diseases (Q42303753)\",\"depth\":0},{\"uri\":\"http://wikidata.org/entity/Q7187\",\"label\":\"gene (Q7187)\",\"depth\":0}]}}}},{\"value\":\"Autosomal Recessive (most common genetic disorder among Caucasians in North America)\",\"links\":[],\"metadata\":{\"entities\":{}}},{\"value\":\"Pancreatic insufficiency due to fibrotic lesions, obstruction of lungs due to thick mucus, lung infections (Staph, aureus, Pseud. aeruginosa)\",\"links\":[],\"metadata\":{\"entities\":{}}}],\"rowId\":1},{\"data\":[{\"value\":\"Duchenne Muscular Dystrophy\",\"links\":[{\"start\":0,\"end\":27,\"href\":\"http://www.wikidata.org/entity/Q1648484\",\"entity\":\"http://wikidata.org/entity/Q1648484\"}],\"metadata\":{\"entities\":{\"http://wikidata.org/entity/Q1648484\":{\"uri\":\"http://wikidata.org/entity/Q1648484\",\"label\":\"Duchenne muscular dystrophy (Q1648484)\",\"types\":[{\"uri\":\"http://wikidata.org/entity/Q929833\",\"label\":\"rare disease (Q929833)\",\"depth\":0}]}}}},{\"value\":\"Dystrophin (DMD)\",\"links\":[{\"start\":0,\"end\":16,\"href\":\"http://www.wikidata.org/entity/Q14864292\",\"entity\":\"http://wikidata.org/entity/Q14864292\"}],\"metadata\":{\"entities\":{\"http://wikidata.org/entity/Q14864292\":{\"uri\":\"http://wikidata.org/entity/Q14864292\",\"label\":\"DMD (Q14864292)\",\"types\":[{\"uri\":\"http://wikidata.org/entity/Q7187\",\"label\":\"gene (Q7187)\",\"depth\":0}]}}}},{\"value\":\"X-linked recessive\",\"links\":[],\"metadata\":{\"entities\":{}}},{\"value\":\"Gradual degeneration of skeletal muscle, impaired heart and respiratory musculature\",\"links\":[],\"metadata\":{\"entities\":{}}}],\"rowId\":2}],\"total\":3}}"
    },
    {
        "type": "send_msg",
        "msg": "{\"id\":\"1\",\"url\":\"/entities\",\"params\":{\"uris\":[\"http://wikidata.org/entity/Q340594\"]}}",
        "version": 4
    },
    {
        "type": "on_receive",
        "version": 5,
        "msg": "{\"type\":\"response\",\"id\":\"1\",\"success\":true,\"response\":{\"http://wikidata.org/entity/Q340594\":{\"uri\":\"http://wikidata.org/entity/Q340594\",\"label\":\"achondroplasia (Q340594)\",\"types\":[{\"uri\":\"http://wikidata.org/entity/Q42303753\",\"label\":\"designated intractable/rare diseases (Q42303753)\",\"depth\":0},{\"uri\":\"http://wikidata.org/entity/Q929833\",\"label\":\"rare disease (Q929833)\",\"depth\":0}],\"props\":{\"http://wikidata.org/prop/P508\":{\"uri\":\"http://wikidata.org/prop/P508\",\"label\":\"BNCF Thesaurus ID (P508)\",\"values\":[{\"value\":\"42773\",\"qualifiers\":{}}]},\"http://wikidata.org/prop/P604\":{\"uri\":\"http://wikidata.org/prop/P604\",\"label\":\"MedlinePlus ID (P604)\",\"values\":[{\"value\":\"001577\",\"qualifiers\":{}}]},\"http://wikidata.org/prop/P557\":{\"uri\":\"http://wikidata.org/prop/P557\",\"label\":\"DiseasesDB (P557)\",\"values\":[{\"value\":\"80\",\"qualifiers\":{}}]},\"http://wikidata.org/prop/P493\":{\"uri\":\"http://wikidata.org/prop/P493\",\"label\":\"ICD-9 (P493)\",\"values\":[{\"value\":\"756.4\",\"qualifiers\":{}}]},\"http://wikidata.org/prop/P673\":{\"uri\":\"http://wikidata.org/prop/P673\",\"label\":\"eMedicine ID (P673)\",\"values\":[{\"value\":\"1258401\",\"qualifiers\":{}}]},\"http://wikidata.org/prop/P646\":{\"uri\":\"http://wikidata.org/prop/P646\",\"label\":\"Freebase ID (P646)\",\"values\":[{\"value\":\"/m/0fmxg\",\"qualifiers\":{}}]},\"http://wikidata.org/prop/P18\":{\"uri\":\"http://wikidata.org/prop/P18\",\"label\":\"image (P18)\",\"values\":[{\"value\":\"Dackelpferd.jpg\",\"qualifiers\":{}}]},\"http://wikidata.org/prop/P1461\":{\"uri\":\"http://wikidata.org/prop/P1461\",\"label\":\"Patientplus ID (P1461)\",\"values\":[{\"value\":\"achondroplasia\",\"qualifiers\":{}}]},\"http://wikidata.org/prop/P699\":{\"uri\":\"http://wikidata.org/prop/P699\",\"label\":\"Disease Ontology ID (P699)\",\"values\":[{\"value\":\"DOID:4480\",\"qualifiers\":{}}]},\"http://wikidata.org/prop/P1748\":{\"uri\":\"http://wikidata.org/prop/P1748\",\"label\":\"NCI Thesaurus ID (P1748)\",\"values\":[{\"value\":\"C34345\",\"qualifiers\":{}}]},\"http://wikidata.org/prop/P279\":{\"uri\":\"http://wikidata.org/prop/P279\",\"label\":\"subclass of (P279)\",\"values\":[{\"value\":{\"uri\":\"http://wikidata.org/entity/Q3251367\",\"label\":\"osteochondrodysplasia (Q3251367)\"},\"qualifiers\":{}}]},\"http://wikidata.org/prop/P1995\":{\"uri\":\"http://wikidata.org/prop/P1995\",\"label\":\"health specialty (P1995)\",\"values\":[{\"value\":{\"uri\":\"http://wikidata.org/entity/Q1071953\",\"label\":\"medical genetics (Q1071953)\"},\"qualifiers\":{}}]},\"http://wikidata.org/prop/P373\":{\"uri\":\"http://wikidata.org/prop/P373\",\"label\":\"Commons category (P373)\",\"values\":[{\"value\":\"Achondroplasia\",\"qualifiers\":{}}]},\"http://wikidata.org/prop/P2888\":{\"uri\":\"http://wikidata.org/prop/P2888\",\"label\":\"exact match (P2888)\",\"values\":[{\"value\":\"http://purl.obolibrary.org/obo/DOID_4480\",\"qualifiers\":{}},{\"value\":\"http://identifiers.org/doid/DOID:4480\",\"qualifiers\":{}}]},\"http://wikidata.org/prop/P2892\":{\"uri\":\"http://wikidata.org/prop/P2892\",\"label\":\"UMLS CUI (P2892)\",\"values\":[{\"value\":\"C0001080\",\"qualifiers\":{}}]},\"http://wikidata.org/prop/P31\":{\"uri\":\"http://wikidata.org/prop/P31\",\"label\":\"instance of (P31)\",\"values\":[{\"value\":{\"uri\":\"http://wikidata.org/entity/Q42303753\",\"label\":\"designated intractable/rare diseases (Q42303753)\"},\"qualifiers\":{}},{\"value\":{\"uri\":\"http://wikidata.org/entity/Q929833\",\"label\":\"rare disease (Q929833)\"},\"qualifiers\":{}}]},\"http://wikidata.org/prop/P3417\":{\"uri\":\"http://wikidata.org/prop/P3417\",\"label\":\"Quora topic ID (P3417)\",\"values\":[{\"value\":\"Achondroplasia\",\"qualifiers\":{}}]},\"http://wikidata.org/prop/P3827\":{\"uri\":\"http://wikidata.org/prop/P3827\",\"label\":\"JSTOR topic ID (P3827)\",\"values\":[{\"value\":\"achondroplasia\",\"qualifiers\":{}}]},\"http://wikidata.org/prop/P4229\":{\"uri\":\"http://wikidata.org/prop/P4229\",\"label\":\"ICD-10-CM (P4229)\",\"values\":[{\"value\":\"Q77.4\",\"qualifiers\":{}}]},\"http://wikidata.org/prop/P4317\":{\"uri\":\"http://wikidata.org/prop/P4317\",\"label\":\"GARD rare disease ID (P4317)\",\"values\":[{\"value\":\"8173\",\"qualifiers\":{}}]},\"http://wikidata.org/prop/P4746\":{\"uri\":\"http://wikidata.org/prop/P4746\",\"label\":\"Elhuyar ZTH ID (P4746)\",\"values\":[{\"value\":\"019952\",\"qualifiers\":{}}]},\"http://wikidata.org/prop/P1417\":{\"uri\":\"http://wikidata.org/prop/P1417\",\"label\":\"Encyclop\u00e6dia Britannica Online ID (P1417)\",\"values\":[{\"value\":\"science/achondroplasia\",\"qualifiers\":{}}]},\"http://wikidata.org/prop/P486\":{\"uri\":\"http://wikidata.org/prop/P486\",\"label\":\"MeSH descriptor ID (P486)\",\"values\":[{\"value\":\"D000130\",\"qualifiers\":{\"http://wikidata.org/prop/P1810\":{\"uri\":\"http://wikidata.org/prop/P1810\",\"label\":\"named as (P1810)\",\"values\":[\"Achondroplasia\"]}}}]},\"http://wikidata.org/prop/P672\":{\"uri\":\"http://wikidata.org/prop/P672\",\"label\":\"MeSH tree code (P672)\",\"values\":[{\"value\":\"C05.116.099.343.110\",\"qualifiers\":{}},{\"value\":\"C05.116.099.708.017\",\"qualifiers\":{}},{\"value\":\"C16.320.240.500\",\"qualifiers\":{}}]},\"http://wikidata.org/prop/P6532\":{\"uri\":\"http://wikidata.org/prop/P6532\",\"label\":\"has phenotype (P6532)\",\"values\":[{\"value\":{\"uri\":\"http://wikidata.org/entity/Q194101\",\"label\":\"dwarfism (Q194101)\"},\"qualifiers\":{}}]},\"http://wikidata.org/prop/P3471\":{\"uri\":\"http://wikidata.org/prop/P3471\",\"label\":\"WikiSkripta ID (P3471)\",\"values\":[{\"value\":\"2284\",\"qualifiers\":{}}]},\"http://wikidata.org/prop/P1325\":{\"uri\":\"http://wikidata.org/prop/P1325\",\"label\":\"external data available at (P1325)\",\"values\":[{\"value\":\"http://www.nanbyou.or.jp/entry/4570\",\"qualifiers\":{\"http://wikidata.org/prop/P407\":{\"uri\":\"http://wikidata.org/prop/P407\",\"label\":\"language of work or name (P407)\",\"values\":[{\"uri\":\"http://wikidata.org/entity/Q5287\",\"label\":\"Japanese (Q5287)\"}]}}}]},\"http://wikidata.org/prop/P2293\":{\"uri\":\"http://wikidata.org/prop/P2293\",\"label\":\"genetic association (P2293)\",\"values\":[{\"value\":{\"uri\":\"http://wikidata.org/entity/Q14914358\",\"label\":\"FGFR3 (Q14914358)\"},\"qualifiers\":{}}]},\"http://wikidata.org/prop/P1550\":{\"uri\":\"http://wikidata.org/prop/P1550\",\"label\":\"Orphanet ID (P1550)\",\"values\":[{\"value\":\"15\",\"qualifiers\":{}}]},\"http://wikidata.org/prop/P7818\":{\"uri\":\"http://wikidata.org/prop/P7818\",\"label\":\"French Vikidia ID (P7818)\",\"values\":[{\"value\":\"Achondroplasie\",\"qualifiers\":{}}]},\"http://wikidata.org/prop/P6366\":{\"uri\":\"http://wikidata.org/prop/P6366\",\"label\":\"Microsoft Academic ID (P6366)\",\"values\":[{\"value\":\"2780472190\",\"qualifiers\":{}}]},\"http://wikidata.org/prop/P5082\":{\"uri\":\"http://wikidata.org/prop/P5082\",\"label\":\"Store medisinske leksikon ID (P5082)\",\"values\":[{\"value\":\"akondroplasi\",\"qualifiers\":{\"http://wikidata.org/prop/P4390\":{\"uri\":\"http://wikidata.org/prop/P4390\",\"label\":\"mapping relation type (P4390)\",\"values\":[{\"uri\":\"http://wikidata.org/entity/Q39893449\",\"label\":\"exact match (Q39893449)\"}]}}}]},\"http://wikidata.org/prop/P8408\":{\"uri\":\"http://wikidata.org/prop/P8408\",\"label\":\"KBpedia ID (P8408)\",\"values\":[{\"value\":\"Achondroplasia\",\"qualifiers\":{}}]},\"http://wikidata.org/prop/P5008\":{\"uri\":\"http://wikidata.org/prop/P5008\",\"label\":\"on focus list of Wikimedia project (P5008)\",\"values\":[{\"value\":{\"uri\":\"http://wikidata.org/entity/Q4099686\",\"label\":\"WikiProject Medicine (Q4099686)\"},\"qualifiers\":{}}]},\"http://wikidata.org/prop/P7982\":{\"uri\":\"http://wikidata.org/prop/P7982\",\"label\":\"Hrvatska enciklopedija ID (P7982)\",\"values\":[{\"value\":\"968\",\"qualifiers\":{}}]},\"http://wikidata.org/prop/P7329\":{\"uri\":\"http://wikidata.org/prop/P7329\",\"label\":\"ICD-11 ID (MMS) (P7329)\",\"values\":[{\"value\":\"LD24.00\",\"qualifiers\":{\"http://wikidata.org/prop/P1810\":{\"uri\":\"http://wikidata.org/prop/P1810\",\"label\":\"named as (P1810)\",\"values\":[\"Achondroplasia\"]}}}]},\"http://wikidata.org/prop/P7807\":{\"uri\":\"http://wikidata.org/prop/P7807\",\"label\":\"ICD-11 (foundation) (P7807)\",\"values\":[{\"value\":\"24224082\",\"qualifiers\":{}}]},\"http://wikidata.org/prop/P665\":{\"uri\":\"http://wikidata.org/prop/P665\",\"label\":\"KEGG ID (P665)\",\"values\":[{\"value\":\"H01749\",\"qualifiers\":{}}]}},\"description\":\"osteochondrodysplasia that results in dwarfism from abnormal ossification of cartilage in long bones\"}}}"
    }
]