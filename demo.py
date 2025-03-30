"""Gradio application for trackgpt."""

import cv2
import gradio as gr
import yaml
from dotenv import load_dotenv
from fast_track import Pipeline
from fast_track.databases import SQLDatabase
from langchain_openai import ChatOpenAI

from trackgpt import LangChainAnalyst

load_dotenv()


with open("configs/rf-detr.yml", "r") as f:
    pipeline_config = yaml.safe_load(f)

with open("configs/agent.yml", "r") as f:
    agent_config = yaml.safe_load(f)


llm = ChatOpenAI(
    model=agent_config["llm"]["model"],
    temperature=agent_config["llm"]["temperature"],
    verbose=agent_config["llm"]["verbose"],
)
analyst = LangChainAnalyst(db_uri="sqlite:///gradio-demo.db", llm=llm, system_prompt=agent_config["system_prompt"])


def run_fast_track(
    input_video: str,
) -> str:
    """Runs a fast_track pipeline with a selected detector and tracker.

    Args:
        input_video: path to processing video.

    Returns:
        A path to an output video.
    """
    camera = cv2.VideoCapture(input_video)

    from fast_track.detectors import RFDETR
    from fast_track.trackers import BYTETracker

    detector = RFDETR(**pipeline_config["detector"], names=pipeline_config["names"])
    tracker = BYTETracker(**pipeline_config["tracker"], names=pipeline_config["names"])

    database = SQLDatabase(
        db_uri="sqlite:///gradio-demo.db", class_names=pipeline_config["names"], use_gpt4v_captions=True
    )
    with Pipeline(camera=camera, detector=detector, tracker=tracker, database=database) as p:
        outfile = p.run()
    return outfile


with gr.Blocks() as demo:
    gr.Markdown(
        """
                # TrackGPT Demo 🚀
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
            output_video = gr.Video(label="Output Video")

            # ChatBox
            gr.ChatInterface(
                analyst.query_analyst,
                title="Chat with TrackGPT",
                type="messages",
            )

        inputs = [
            input_video,
        ]
        btn.click(fn=run_fast_track, inputs=inputs, outputs=[output_video])

if __name__ == "__main__":
    demo.launch()
