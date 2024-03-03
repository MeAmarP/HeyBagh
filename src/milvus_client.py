from pymilvus import connextions, db


class MilvusClient():
    def __init__(self) -> None:
        # TODO - set connection alias_name
        # TODO - set connection host and port
        # TODO - set connection db_name

        pass
    def connect(self,alias='heybagh_conn',host="127.0.0.1", port=19530, db_name):
        conn = connections.connect(alias=alias host=host, port=port, db_name=db_name)

    def disconnect(self,alias_name: str):
        connections.disconnect(alias_name)

    def create_db(self, db_name: str, alias_name: str):
        # TODO check if db exists, then create
        database = db.create_database(db_name=db_name, using=alias_name)



