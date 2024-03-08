import os
import json
from dotenv import find_dotenv, load_dotenv

from pymilvus import (
    connections,
    Collection,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
)

load_dotenv(find_dotenv())


class HeyBaghMilvusClient:
    def __init__(self):
        self.host = os.getenv("HOST")
        self.port = os.getenv("PORT")
        self.collection = None
        self.index_params = None
        self.results = None
        self.search_metric = os.getenv("METRIC_TYPE")
        self.search_topk = int(os.getenv("TOPK"))
        self.anns_fields = os.getenv("ANNS_FIELDS")
        self.output_fields = json.loads(os.getenv("OUTPUT_FIELDS"))
        self.connect_to_milvus()

    def connect_to_milvus(self):
        """Connect to the Milvus server."""
        connections.connect(alias="default", host=self.host, port=self.port)
        # print(f"Connected to Milvus on {self.host}:{self.port}")

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
        self.collection.create_index(
            field_name=field_name, index_params=self.index_params
        )
        print(
            f"Index created for field '{field_name}' in collection '{collection_name}' with params {index_params}."
        )

    def load_collection(self, collection_name):
        """Load the collection into memory to prepare for searching."""
        self.collection = Collection(name=collection_name)
        self.collection.load()
        print(f"Collection '{collection_name}' loaded into memory.")

    # ! TODO Write for Hybrid Search
    def img_search(self, img_path, feature_extractor, search_type="vector"):
        """Perform a search on the collection."""
        img_vector = [feature_extractor.preprocess_extract_feature(img_path).tolist()]

        if search_type == "vector":
            search_params = {
                "metric_type": self.search_metric,
                "params": {"nprobe": self.search_topk},
            }
            self.results = self.collection.search(
                data=img_vector,
                anns_field=self.anns_fields,
                param=search_params,
                limit=self.search_topk,
                output_fields=self.output_fields,
            )
            if self.results:
                img_paths = []
                img_class_names = []
                for ele in self.results:
                    for e in ele:
                        img_paths.append(e.entity.get("img_rel_path"))
                        img_class_names.append(e.entity.get("img_cls_name"))
                return [img_paths, img_class_names]
            return None


# Example usage:
if __name__ == "__main__":
    from feature_extractor import FeatureExtractor_CNN  # Ensure this is correctly imported

    client = HeyBaghMilvusClient()
    client.connect_to_milvus()
    print(client.get_list_of_collections())
    client.load_collection("heybagh_caltech101_imgs")

    img_feat_extactor = FeatureExtractor_CNN()
    image_path = "/datasets/caltech-101/101_ObjectCategories/brontosaurus/image_0006.jpg"

    print(client.img_search(img_path=image_path, feature_extractor=img_feat_extactor))
    
