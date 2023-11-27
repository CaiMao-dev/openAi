
from milvus import default_server


from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)

# This example shows how to:
#   1. connect to Milvus server
#   2. create a collection
#   3. insert entities
#   4. create index
#   5. search

# Optional, if you want store all related data to specific location
# default it wil using ~/.milvus-io/milvus-server/<__version_of_milvus__>
default_server.set_base_dir('test_milvus')

# Optional, if you want cleanup previous data
default_server.cleanup()

# start you milvus server
default_server.start()

_HOST = '10.90.5.208'
# The port may be changed, by default it's 19530
_PORT = default_server.listen_port
# max file size of stored index
_INDEX_FILE_SIZE = 32  

# Index parameters
_METRIC_TYPE = 'L2'
_INDEX_TYPE = 'IVF_FLAT'
_NLIST = 1024
_NPROBE = 16
_TOPK = 3


# Create a Milvus connection
def create_connection():
    print(f"\nCreate connection...")
    connections.connect(host=_HOST, port=_PORT)
    print(f"\nList connections:")
    print(connections.list_connections())


# Create a collection named 'demo'
def create_collection(name):
    field1 = FieldSchema(name='id_field', dtype=DataType.INT64, description="int64", is_primary=True)
    field2 = FieldSchema(name='vector_field', dtype=DataType.FLOAT_VECTOR, description="float vector", dim=1536, is_primary=False)
    field3 = FieldSchema(name='content', dtype=DataType.VARCHAR, description="content", is_primary=False, max_length = 40960)

    schema = CollectionSchema(fields=[field1, field2, field3], description="collection description")
    collection = Collection(name=name, data=None, schema=schema, properties={"collection.ttl.seconds": 15})
    print("\ncollection created:", name)
    return collection


def has_collection(name):
    return utility.has_collection(name)


# Drop a collection in Milvus
def drop_collection(name):
    collection = Collection(name)
    collection.drop()
    print("\nDrop collection: {}".format(name))


def insert(collection, data):
    collection.insert(data)


def get_entity_num(collection):
    print("\nThe number of entity:")
    print(collection.num_entities)


def create_index(collection):
    index_param = {
        "index_type": _INDEX_TYPE,
        "params": {"nlist": _NLIST},
        "metric_type": _METRIC_TYPE}
    collection.create_index('vector_field', index_param)
    print("\nCreated index:\n{}".format(collection.index().params))


def drop_index(collection):
    collection.drop_index()
    print("\nDrop index sucessfully")


def load_collection(collection):
    collection.load()


def release_collection(collection):
    collection.release()


def search(collection, search_vectors):
    search_param = {
        "data": [search_vectors],
        "anns_field": 'vector_field',
        "param": {"metric_type": _METRIC_TYPE, "params": {"nprobe": _NPROBE}},
        "limit": 1,
        "output_fields":['content'],
        "expr": None
        }
    return collection.search(**search_param)


def set_properties(collection):
    collection.set_properties(properties={"collection.ttl.seconds": 1800})