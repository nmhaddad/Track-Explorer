""" OpenAI Analyst class. """

from typing import List, Optional

from dotenv import load_dotenv
from langchain.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from .analyst import Analyst

load_dotenv()


class OpenAIAnalyst(Analyst):

    def __init__(self,
                 db_uri: str,
                 model: str = "gpt-3.5-turbo",
                 temperature: float = 0):
        """ Initializes OpenAIAnalyst objects.

        Args:
            db_uri: The database URI.
            model: The model to use for the analyst.
            temperature: The sampling temperature, between 0 and 1.
                Higher values like 0.8 will make the output more random,
                while lower values like 0.2 will make it more focused and
                deterministic. If set to 0, the model will use log probability
                to automatically increase the temperature until certain thresholds
                are hit.
        """
        super().__init__()

        self.db_uri = db_uri
        self.model = model
        self.temperature = temperature

        self.db = None
        self.agent_executor = None

        self._link_db()
        self._create_db_agent()

    @property
    def system_message(self) -> str:
        """ Returns the system message for the analyst.

        Notes:
            Current token count: 323.
        """
        return """
        You are a cutting-edge data analysis and tracking assistant powered by advanced Large Language Models (LLM).
        Your primary focus is on aiding users in the comprehensive analysis of video data. You will be provided access
        to a diverse range of databases, each containing unique datasets that require your expertise to extract valuable
        insights and detect anomalies. Your mission is answering user queries and providing valuable insights.

        Your Role:
        - Query Interpreter: Process user queries in natural language and offer valuable insights based on the principles of
          data analysis, detection, and tracking.
        - Real-time Monitoring Assistant: Act as a vigilant companion, keeping users informed about changes in their data
          through real-time tracking functionalities.
        """

    def _create_db_agent(self) -> None:
        """ Creates a LangChain database agent. """
        llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature
        )
        self.agent_executor = create_sql_agent(
            llm=llm,
            db=self.db,
            agent_type="openai-tools",
            verbose=True,
            system_message=self.system_message,
            agent_executor_kwargs={"system_message": self.system_message}
        )

    def _link_db(self) -> None:
        """ Sets up the database connection. """
        self.db = SQLDatabase.from_uri(self.db_uri)

    def query_analyst(self, message: str, history: Optional[List[str]] = None) -> str:
        """ Send a query to analyst.

        Args:
            user_input: text prompt provided by the user.

        Returns:
            a text response from the analyst.
        """
        response = self.agent_executor.invoke(input=message)
        return response['output']
