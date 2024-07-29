from pymilvus import MilvusClient

class MilvusManager:
    def __init__(self):
        self.client = MilvusClient(uri="http://localhost:19530")

    def list_collection(self): 
        return self.client.list_collections() 
    
    def drop_collection(self, collection_name):
        self.client.drop_collection(collection_name)

    def rename_collection(self, old_name, new_name):
        self.client.rename_collection(
        old_name, new_name)