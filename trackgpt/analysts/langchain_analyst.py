"""LangChainAnalyst class."""

import logging
from typing import List, Optional

from langchain.agents import AgentExecutor
from langchain.base_language import BaseLanguageModel
from langchain.memory import ConversationBufferMemory
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase

from .base_analyst import BaseAnalyst

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LangChainAnalyst(BaseAnalyst):
    def __init__(
        self,
        db_uri: str,
        llm: BaseLanguageModel,
        system_prompt: Optional[str] = None,
        verbose: Optional[bool] = False,
    ):
        """Initializes LangChainAnalyst objects.

        Args:
            db_uri: The database URI.
            llm: The language model to use. Must be a BaseLanguageModel object.
            system_prompt: The system message to use for the analyst.
        """
        logger.info("LangChainAnalyst | Initializing")
        super().__init__()

        self.db_uri = db_uri
        self.llm = llm
        self.system_prompt = system_prompt
        self.verbose = verbose

        self.db: SQLDatabase = None
        self.agent_executor: AgentExecutor = None

        self._link_db()
        self._create_db_agent()
        logger.info("LangChainAnalyst | SUCCESS")

    def _create_db_agent(self) -> None:
        """Creates a LangChain database agent."""
        logger.info("_create_db_agent | Initializing Memory")
        memory = ConversationBufferMemory(input_key="input", memory_key="history")
        logger.info("_create_db_agent | Initializing AgentExecutor")
        self.agent_executor = create_sql_agent(
            llm=self.llm,
            db=self.db,
            agent_type="openai-tools",
            verbose=self.verbose,
            system_prompt=self.system_prompt,
            memory=memory,
        )
        logger.info("_create_db_agent | SUCCESS")

    def _link_db(self) -> None:
        """Sets up the database connection."""
        logger.info("_link_db | Setting up database connection")
        self.db = SQLDatabase.from_uri(
            self.db_uri,
        )
        logger.info("_link_db | SUCCESS")

    def query_analyst(self, message: str, history: Optional[List[str]] = None) -> str:
        """Send a query to analyst.

        Args:
            message: The message to send to the analyst.
            history: The history of the conversation.

        Returns:
            a text response from the analyst.
        """
        logger.info("query_analyst | Invoking agent executor")
        return self.agent_executor.invoke(input=message, history=history)["output"]
