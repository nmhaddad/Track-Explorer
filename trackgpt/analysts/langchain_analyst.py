""" LangChainAnalyst class. """

import logging
from typing import List, Optional

from langchain.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor

from .analyst import Analyst


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LangChainAnalyst(Analyst):

    def __init__(self,
                 db_uri: str,
                 model: str = "gpt-3.5-turbo",
                 temperature: float = 0,
                 system_message: Optional[str] = None,
                 verbose: bool = False):
        """ Initializes LangChainAnalyst objects.

        Args:
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
        logger.info("LangChainAnalyst | Initializing")
        super().__init__()

        self.db_uri = db_uri
        self.model = model
        self.temperature = temperature
        self.system_message = system_message
        self.verbose = verbose

        self.db: SQLDatabase = None
        self.agent_executor: AgentExecutor = None

        self._link_db()
        self._create_db_agent()
        logger.info("LangChainAnalyst | SUCCESS")

    def _create_db_agent(self) -> None:
        """ Creates a LangChain database agent. """
        logger.info("_create_db_agent | Initializing LLM")
        llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            verbose=self.verbose
        )
        logger.info("_create_db_agent | Initializing Memory")
        memory = ConversationBufferMemory(input_key='input', memory_key="history")
        logger.info("_create_db_agent | Initializing AgentExecutor")
        self.agent_executor = create_sql_agent(
            llm=llm,
            db=self.db,
            agent_type="openai-tools",
            verbose=self.verbose,
            system_message=self.system_message,
            memory=memory
            # agent_type
        )
        logger.info("_create_db_agent | SUCCESS")

        # db = SQLDatabase.from_uri(
        #     self.db_uri,
        #     include_tables=include_tables,
        #     schema=postgresql_schema,
        #     sample_rows_in_table_info=3
        
        # dbchain = SQLDatabaseChain(
        #         llm_chain=LLMChain(llm=llm, prompt=prompt, memory=memory),
        #         database=db, 
        #         verbose=verbose
        #     )

    def _link_db(self) -> None:
        """ Sets up the database connection. """
        logger.info("_link_db | Setting up database connection")
        self.db = SQLDatabase.from_uri(
            self.db_uri,
        )
        logger.info("_link_db | SUCCESS")

    def query_analyst(self, message: str, history: Optional[List[str]] = None) -> str:
        """ Send a query to analyst.

        Args:
            message: The message to send to the analyst.
            history: The history of the conversation.

        Returns:
            a text response from the analyst.
        """
        logger.info("query_analyst | Invoking agent executor")
        return self.agent_executor.invoke(
            input=message,
            history=history
        )['output']
