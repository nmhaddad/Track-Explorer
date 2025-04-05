# Track-Explorer

Installable Python package for interacting with object tracking databases using large language models (LLMs).

[Try it out now with Gradio](#run-the-demo).

## Installation:

Package is installable with Python 3.10, 3.11, and 3.12

1. `git clone <repo>`
1. `cd <repo>`
1. `pip install .`

## Running:

```
import os

import yaml
from dotenv import load_dotenv
from smolagents import LiteLLMModel

from track_explorer import SmolAgentsAnalyst

load_dotenv()

os.environ["TRACK_EXPLORER_DB_URI"] = "sqlite:///gradio-demo.db"

llm = LiteLLMModel(
    model_id="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.2,
)
analyst = SmolAgentsAnalyst(db_uri="sqlite:///gradio-demo.db", llm=llm)
analyst.launch()
```

## Run the Demo

1. Follow the installation instructions above. 
1. Install Fast-Track ByteTrack `pip install fast_track[bytetrack]`.
1. Adjust `configs/agent.yml` and `configs/rd-detr.yml` with runtime configuration.
1. Finally, launch the app with `python demo.py`

## Contact:
Author: Nate Haddad - nhaddad2112[at]gmail[dot]com

## License:
[See LICENSE.txt](LICENSE)

## References:

[1] Aymeric Roucher and Albert Villanova del Moral and Thomas Wolf and Leandro von Werra and Erik Kaunismäki; `smolagents`: a smol library to build great agentic systems; 2025; [Online]. Available: https://github.com/huggingface/smolagents
