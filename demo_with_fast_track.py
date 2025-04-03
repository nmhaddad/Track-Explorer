"""Gradio application for TrackExplorer."""

import os

import cv2
import gradio as gr
import yaml
from dotenv import load_dotenv
from fast_track import Pipeline
from fast_track.databases import SQLDatabase
from fast_track.detectors import RFDETR
from fast_track.trackers import BYTETracker
from smolagents import LiteLLMModel

from track_explorer import SmolAgentsAnalyst

load_dotenv()


with open("configs/rf-detr.yml", "r") as f:
    pipeline_config = yaml.safe_load(f)

with open("configs/agent.yml", "r") as f:
    agent_config = yaml.safe_load(f)

os.environ["TRACK_EXPLORER_DB_URI"] = "sqlite:///gradio-demo.db"


llm = LiteLLMModel(
    model_id=agent_config["llm"]["model"],
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=agent_config["llm"]["temperature"],
)

analyst = SmolAgentsAnalyst(db_uri="sqlite:///gradio-demo.db", llm=llm, system_prompt=agent_config["system_prompt"])


def run_fast_track(
    input_video: str,
) -> str:
    """Runs a fast_track pipeline with a selected detector and tracker. Writes to SQL database.

    Args:
        input_video: path to processing video.

    Returns:
        A path to an output video.
    """
    camera = cv2.VideoCapture(input_video)
    detector = RFDETR(**pipeline_config["detector"], names=pipeline_config["names"])
    tracker = BYTETracker(**pipeline_config["tracker"], names=pipeline_config["names"])
    database = SQLDatabase(
        db_uri="sqlite:///gradio-demo.db",
        class_names=pipeline_config["names"],
        create_image_captions=pipeline_config["db"]["create_image_captions"],
    )
    with Pipeline(camera=camera, detector=detector, tracker=tracker, database=database) as p:
        outfile = p.run()
    return outfile


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # TrackExplorer Demo 🚀
        ### Upload a video and process it with a Fast-Track pipeline - then chat with an agent about what's been seen!
        """
    )
    with gr.Row():
        # Input Column
        with gr.Column():
            input_video = gr.PlayableVideo(label="Input Video", interactive=True)
            # Run
            btn = gr.Button("Run", variant="primary")

        # Output Column
        with gr.Column():
            output_video = gr.Video(label="Output Video", loop=True)

            # ChatBox
            gr.ChatInterface(
                analyst.query_analyst,
                title="Chat with TrackExplorer",
                type="messages",
            )

        inputs = [
            input_video,
        ]
        btn.click(fn=run_fast_track, inputs=inputs, outputs=[output_video])

if __name__ == "__main__":
    demo.launch()
