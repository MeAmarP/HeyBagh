from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from feature_extractor import FeatureExtractor_CNN  # Ensure this is correctly imported

class HeyBaghMilvusClient:
    def __init__(self, host='localhost', port='19530'):
        self.host = host
        self.port = port
        self.feature_extractor = FeatureExtractor_CNN()  # Initialize feature extractor
    
    def connect_to_milvus(self):
        """Connect to the Milvus server."""
        connections.connect(alias="default", host=self.host, port=self.port)
        print(f"Connected to Milvus on {self.host}:{self.port}")
    
    def get_list_of_collections(self):
        """Get the list of collections available in Milvus."""
        return utility.list_collections()
    
    def create_collection(self, collection_name, fields, description):
        """Create a collection in Milvus with the specified fields."""
        if not utility.has_collection(collection_name):
            collection_schema = CollectionSchema(fields, description=description)
            self.collection = Collection(name=collection_name, schema=collection_schema)
            print(f"Collection '{collection_name}' created.")
        else:
            print(f"Collection '{collection_name}' already exists.")
    
    def insert_entities(self, entities):
        """Insert entities into the specified collection."""
        
        ids = self.collection.insert(entities)
        self.collection.load()  # Load data into memory after insertion
        print(f"Inserted entities into collection: '{self.collection.num_entities}'.")
        return ids
    
    def create_index(self, collection_name, field_name, index_params):
        """Create an index for a collection."""
        self.index_params = index_params
        self.collection.create_index(field_name=field_name, index_params=self.index_params)
        print(f"Index created for field '{field_name}' in collection '{collection_name}' with params {index_params}.")
    
    def load_collection(self, collection_name):
        """Load the collection into memory to prepare for searching."""
        collection = Collection(name=collection_name)
        collection.load()
        print(f"Collection '{collection_name}' loaded into memory.")
    

    # ! TODO Write a function for Vector and Hybrid Search
    def search(self, collection_name, search_params, search_data, search_fields=None):
        """Perform a search on the collection."""
        collection = Collection(name=collection_name)
        results = collection.search(data=search_data, anns_field=search_params["anns_field"],
                                    param=search_params["search_params"], limit=search_params["limit"],
                                    expr=search_params.get("expr"), output_fields=search_fields)
        return results

# Example usage:
if __name__ == "__main__":
    client = HeyBaghMilvusClient()
    client.connect_to_milvus()

    # Define fields for a new collection
    # fields = [
    #     FieldSchema(name="image_id", dtype=DataType.INT64, is_primary=True),
    #     FieldSchema(name="image_vector", dtype=DataType.FLOAT_VECTOR, dim=128)  # Adjust 'dim' as needed
    # ]
    # client.create_collection("image_collection", fields)
    print(client.get_list_of_collections())
