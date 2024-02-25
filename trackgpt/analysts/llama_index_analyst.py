""" LlamaIndex Analyst """

from typing import List, Optional

from llama_index.core import SQLDatabase, ServiceContext, VectorStoreIndex
from llama_index.core.indices.struct_store.sql_query import SQLTableRetrieverQueryEngine
from llama_index.core.objects import (
    SQLTableNodeMapping,
    SQLTableSchema,
    ObjectIndex,
)
from llama_index.llms.openai import OpenAI

from sqlalchemy import create_engine, MetaData       

from .base_analyst import BaseAnalyst


class LlamaIndexAnalyst(BaseAnalyst):
    """ LlamaIndexAnalyst class.
    
    Attributes:
        db_uri: The database URI.
        model: The model to use for the analyst.
        temperature: The sampling temperature, between 0 and 1.
            Higher values like 0.8 will make the output more random,
            while lower values like 0.2 will make it more focused and
            deterministic. If set to 0, the model will use log probability
            to automatically increase the temperature until certain thresholds
            are hit.
        system_message: The system message to use for the analyst.
        verbose: Whether to print debug messages.
    """

    def __init__(self,
                 db_uri: str,
                 system_message: str,
                 model: str="gpt-3.5-turbo",
                 temperature: float = 0.1,
                 verbose: bool = False):
        """ Initializes LlamaIndexAnalyst objects.

        Args:
            db_uri: The database URI.
            system_message: The system message to use for the analyst.
            model: The model to use for the analyst.
            temperature: The sampling temperature, between 0 and 1.
                Higher values like 0.8 will make the output more random,
                while lower values like 0.2 will make it more focused and
                deterministic. If set to 0, the model will use log probability
                to automatically increase the temperature until certain thresholds
                are hit.
            verbose: Whether to print debug messages.
        """
        super().__init__()

        self.db_uri = db_uri
        self.model = model
        self.temperature = temperature
        self.system_message = system_message
        self.verbose = verbose

        self.db: SQLDatabase = None
        self.query_engine: SQLTableRetrieverQueryEngine = None
        self.table_names: List[str] = None
    
        self._link_db()
        self._create_db_agent()

    def _link_db(self) -> None:
        """ Links the database. """
        engine = create_engine(self.db_uri)
        metadata = MetaData()
        metadata.reflect(engine)
        self.table_names = metadata.tables.keys()
        self.db = SQLDatabase(engine, include_tables=self.table_names)

    def _create_db_agent(self) -> None:
        """ Creates a LlamaIndex database agent. """
        llm = OpenAI(
            self.model,
            temperature=self.temperature,
        )
        service_context = ServiceContext.from_defaults(llm=llm)

        table_node_mapping = SQLTableNodeMapping(self.db)
        table_schema_objs = [
            (SQLTableSchema(table_name=table_name))
            for table_name in self.table_names
        ]  # add a SQLTableSchema for each table
        obj_index = ObjectIndex.from_objects(
            table_schema_objs,
            table_node_mapping,
            VectorStoreIndex,
        )
        self.query_engine = SQLTableRetrieverQueryEngine(
            self.db,
            obj_index.as_retriever(similarity_top_k=1),
            service_context=service_context,
        )
    
    def invoke(self, message: str, history: Optional[List[str]] = None) -> str:
        """ Queries the analyst.

        Args:
            message: The input message.
            history: The history of the conversation.

        Returns:
            The analyst's response.
        """
        return self.query_engine.query({
            "input": message, 
            "history": history
        })
