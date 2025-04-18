"""SmolAgentsAnalyst class."""

from typing import Any, Callable, Dict, Generator, List, Optional

from smolagents import ChatMessage, CodeAgent, GradioUI, stream_to_gradio
from sqlalchemy import Engine, create_engine, inspect

from ...utils import logger
from ..base_analyst import BaseAnalyst
from .retriever import RetrieverTool, get_documents_from_files
from .tools import generate_image_caption, sql_engine, visual_question_answering


class SmolAgentsAnalyst(BaseAnalyst):
    def __init__(
        self,
        db_uri: str,
        llm: Callable[[list[dict[str, str]]], ChatMessage],
        system_prompt: Optional[str] = None,
        knowledge_base: Optional[List[Dict[str, str]]] = None,
    ):
        """Initializes SmolAgentsAnalyst objects.

        Args:
            db_uri: The database URI.
            llm: The language model to use. Must be a BaseLanguageModel object.
            system_prompt: The system message to use for the analyst (overrides smolagent system prompt).
            knowledge_base: Optional list of documents to initialize the retriever tool. This should be a list of dicts with 'text' and 'source' keys.
        """
        super().__init__()

        self.db_uri = db_uri
        self.llm = llm
        self.system_prompt = system_prompt
        self.knowledge_base = knowledge_base if knowledge_base else []

        self.db = None
        self.agent = None
        self.initialized = False

    def _initialize(self) -> None:
        """Initializes the analyst."""
        logger.info("SmolAgentsAnalyst | Initializing")
        self._link_db()
        self._create_agent()
        logger.info("SmolAgentsAnalyst | SUCCESS")

    def _create_agent(self) -> None:
        """Creates a smolagents database agent."""
        logger.info("_create_agent | Initializing CodeAgent and Tools")

        tools = [generate_image_caption, visual_question_answering]

        # Text-to-SQL tool
        updated_description = """Allows you to perform SQL queries on the table. Beware that this tool's output is a string representation of the execution output.
        It can use the following tables:"""
        inspector = inspect(self.db)
        for table in inspector.get_table_names():
            columns_info = [(col["name"], col["type"]) for col in inspector.get_columns(table)]

            table_description = f"Table '{table}':\n"

            table_description += "Columns:\n" + "\n".join(
                [f"  - {name}: {col_type}" for name, col_type in columns_info]
            )
            updated_description += "\n\n" + table_description
        sql_engine.description = updated_description
        tools.append(sql_engine)

        # Agentic RAG tool
        if self.knowledge_base:
            retriever_tool = RetrieverTool(get_documents_from_files(knowledge_base=self.knowledge_base))
            tools.append(retriever_tool)

        self.agent = CodeAgent(
            tools=tools,
            model=self.llm,
            max_steps=20,
            planning_interval=3,
            additional_authorized_imports=["pandas", "numpy", "scipy", "pillow", "cv2, sqlalchemy"],
        )
        logger.info("_create_db_agent | SUCCESS")

    def _link_db(self) -> None:
        """Sets up the database connection."""
        logger.info("_link_db | Setting up database connection")
        self.db: Engine = create_engine(self.db_uri)
        self.table_names: List[str] = self.db
        logger.info("_link_db | SUCCESS")

    def query_analyst(self, prompt: str, messages: List[Dict[str, str]]) -> Generator[ChatMessage, Any, None]:
        """Send a query to analyst.

        Args:
            prompt: The prompt to send to the analyst.
            messages: A history of messages.

        Returns:
            a text response from the analyst.
        """
        if not self.initialized:
            self._initialize()
            self.initialized = True
        logger.info("query_analyst | Invoking agent executor")
        new_messages = []
        for msg in stream_to_gradio(self.agent, task=prompt, reset_agent_memory=False):
            new_messages.append(msg)
            yield new_messages
        yield new_messages

    def launch(self) -> None:
        """Launches the analyst."""
        logger.info("launch | Launching SmolAgentsAnalyst")
        if not self.initialized:
            self._initialize()
            self.initialized = True
        gradio_ui = GradioUI(
            agent=self.agent,
        )
        gradio_ui.launch()
