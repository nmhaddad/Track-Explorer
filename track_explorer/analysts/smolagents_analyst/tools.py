"""This file contains the tools that are used by the smolagents."""

import os

from openai import OpenAI
from smolagents import tool
from sqlalchemy import create_engine, text


@tool
def sql_engine(query: str) -> str:
    """
    Allows you to perform SQL queries on the table. Beware that this tool's output is a string representation of the execution output.
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


@tool
def generate_image_caption(image_base64: str) -> str | None:
    """
    Allows you to generate a caption for an image. This is useful for describing the contents of a frame in a video or
    an image. Use this tool sparingly, as it can be slow and expensive.

    Args:
        image_base64 (str): The base64 encoded string representation of the image.
    """
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe the image.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "low",
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception:
        return None
