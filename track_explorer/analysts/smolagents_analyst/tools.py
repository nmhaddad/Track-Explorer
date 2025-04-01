"""This file contains the tools that are used by the smolagents."""

import os

from smolagents import tool
from sqlalchemy import create_engine, text


@tool
def sql_engine(query: str) -> str:
    """
    "Allows you to perform SQL queries on the table. Beware that this tool's output is a string representation of the execution output.
    It can use the following tables:

    Args:
        query: The query to perform. This should be correct SQL.
    """
    output = ""
    engine = create_engine(os.getenv("TRACK_EXPLORER_DB_URI"))
    with engine.connect() as con:
        rows = con.execute(text(query))
        for row in rows:
            output += "\n" + str(row)
    return output
