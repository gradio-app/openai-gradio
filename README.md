# `openai-gradio`

is a Python package that makes it very easy for developers to create machine learning apps that are powered by OpenAI's API.

# Installation

1. Clone this repo: `git clone https://github.com/AK391/openai-gradio.git`
2. Navigate into the folder that you cloned this repo into: `cd openai-gradio`
3. Install this package: `pip install -e .`

<!-- ```bash
pip install openai-gradio
``` -->

That's it! 

# Basic Usage

Just like if you were to use the `openai` API, you should first save your OpenAI API key to this environment variable:

```
export OPENAI_API_KEY=<your token>
```

Then in a Python file, write:

```python
import gradio as gr
import openai_gradio

gr.load(
    name='gpt-4-turbo',
    src=openai_gradio.registry,
).launch()
```

Run the Python file, and you should see a Gradio Interface connected to the model on OpenAI!

![ChatInterface](chatinterface.png)

# Customization 

Once you can create a Gradio UI from an OpenAI endpoint, you can customize it by setting your own input and output components, or any other arguments to `gr.Interface`. For example, the screenshot below was generated with:

```py
import gradio as gr
import openai_gradio

gr.load(
    name='gpt-4-turbo',
    src=openai_gradio.registry,
    title='OpenAI-Gradio Integration',
    description="Chat with GPT-4-turbo model.",
    examples=["Explain quantum gravity to a 5-year old.", "How many R are there in the word Strawberry?"]
).launch()
```
![ChatInterface with customizations](chatinterface_with_customization.png)

# Composition

Or use your loaded Interface within larger Gradio Web UIs, e.g.

```python
import gradio as gr
import openai_gradio

with gr.Blocks() as demo:
    with gr.Tab("GPT-4-turbo"):
        gr.load('gpt-4-turbo', src=openai_gradio.registry)
    with gr.Tab("GPT-3.5-turbo"):
        gr.load('gpt-3.5-turbo', src=openai_gradio.registry)

demo.launch()
```

# Under the Hood

The `openai-gradio` Python library has two dependencies: `openai` and `gradio`. It defines a "registry" function `openai_gradio.registry`, which takes in a model name and returns a Gradio app.

# Supported Models in OpenAI

| Model | Context Length | Output Length | Dtype / Precision |
|-------|----------------|---------------|-------|
| gpt-4-turbo | 128000 | Varies | - |
| gpt-3.5-turbo | 4096 | Varies | - |
| gpt-3.5-turbo-16k | 16384 | Varies | - |
| gpt-4 | 8192 | Varies | - |
| gpt-4-32k | 32768 | Varies | - |

-------

Note: if you are getting a 401 authentication error, then the OpenAI API Client is not able to get the API token from the environment variable. This happened to me as well, in which case save it in your Python session, like this:

```py
import os

os.environ["OPENAI_API_KEY"] = ...
```