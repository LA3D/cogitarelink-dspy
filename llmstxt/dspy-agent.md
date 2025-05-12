Title: Advanced Tool Use - DSPy

URL Source: https://dspy.ai/tutorials/tool_use/

Markdown Content:
Tutorial: Advanced Tool Use[¶](https://dspy.ai/tutorials/tool_use/#tutorial-advanced-tool-use)
----------------------------------------------------------------------------------------------

Let's walk through a quick example of building and prompt-optimizing a DSPy agent for advanced tool use. We'll do this for the challenging task [ToolHop](https://arxiv.org/abs/2501.02506) but with an even stricter evaluation criteria.

Install the latest DSPy via `pip install -U dspy` and follow along. You will also need to `pip install func_timeout`.

Recommended: Set up MLflow Tracing to understand what's happening under the hood.

### MLflow DSPy Integration[¶](https://dspy.ai/tutorials/tool_use/#mlflow-dspy-integration)

[MLflow](https://mlflow.org/) is an LLMOps tool that natively integrates with DSPy and offer explainability and experiment tracking. In this tutorial, you can use MLflow to visualize prompts and optimization progress as traces to understand the DSPy's behavior better. You can set up MLflow easily by following the four steps below.

1.  Install MLflow

%pip install mlflow\>\=2.20

2.  Start MLflow UI in a separate terminal

mlflow ui \--port 5000

3.  Connect the notebook to MLflow

import mlflow

mlflow.set\_tracking\_uri("http://localhost:5000")
mlflow.set\_experiment("DSPy")

4.  Enabling tracing.

mlflow.dspy.autolog()

To learn more about the integration, visit [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) as well.

In this tutorial, we'll demonstrate the new experimental `dspy.SIMBA` prompt optimizer, which tends to be powerful for larger LLMs and harder tasks. Using this, we'll improve our agent from 35% accuracy to 60%.

In \[1\]:

import dspy
import ujson
import random

gpt4o \= dspy.LM("openai/gpt-4o", temperature\=0.7)
dspy.configure(lm\=gpt4o)

import dspy import ujson import random gpt4o = dspy.LM("openai/gpt-4o", temperature=0.7) dspy.configure(lm=gpt4o)

Let's now download the data.

In \[2\]:

from dspy.utils import download

download("https://huggingface.co/datasets/bytedance-research/ToolHop/resolve/main/data/ToolHop.json")

data \= ujson.load(open("ToolHop.json"))
random.Random(0).shuffle(data)

from dspy.utils import download download("https://huggingface.co/datasets/bytedance-research/ToolHop/resolve/main/data/ToolHop.json") data = ujson.load(open("ToolHop.json")) random.Random(0).shuffle(data)

Downloading 'ToolHop.json'...

Then let's prepare a cleaned set of examples. The ToolHop task is interesting in that the agent gets a _unique set_ of tools (functions) to use separately for each request. Thus, it needs to learn how to use _any_ such tools effectively in practice.

In \[3\]:

import re
import inspect

examples \= \[\]
fns2code \= {}

def finish(answer: str):
    """Conclude the trajectory and return the final answer."""
    return answer

for datapoint in data:
    func\_dict \= {}
    for func\_code in datapoint\["functions"\]:
        cleaned\_code \= func\_code.rsplit("\\n\\n\# Example usage", 1)\[0\]
        fn\_name \= re.search(r"^\\s\*def\\s+(\[a-zA-Z0-9\_\]+)\\s\*\\(", cleaned\_code)
        fn\_name \= fn\_name.group(1) if fn\_name else None

        if not fn\_name:
            continue

        local\_vars \= {}
        exec(cleaned\_code, {}, local\_vars)
        fn\_obj \= local\_vars.get(fn\_name)

        if callable(fn\_obj):
            func\_dict\[fn\_name\] \= fn\_obj
            assert fn\_obj not in fns2code, f"Duplicate function found: {fn\_name}"
            fns2code\[fn\_obj\] \= (fn\_name, cleaned\_code)

    func\_dict\["finish"\] \= finish

    example \= dspy.Example(question\=datapoint\["question"\], answer\=datapoint\["answer"\], functions\=func\_dict)
    examples.append(example.with\_inputs("question", "functions"))

trainset, devset, testset \= examples\[:100\], examples\[100:400\], examples\[400:\]

import re import inspect examples = \[\] fns2code = {} def finish(answer: str): """Conclude the trajectory and return the final answer.""" return answer for datapoint in data: func\_dict = {} for func\_code in datapoint\["functions"\]: cleaned\_code = func\_code.rsplit("\\n\\n# Example usage", 1)\[0\] fn\_name = re.search(r"^\\s\*def\\s+(\[a-zA-Z0-9\_\]+)\\s\*\\(", cleaned\_code) fn\_name = fn\_name.group(1) if fn\_name else None if not fn\_name: continue local\_vars = {} exec(cleaned\_code, {}, local\_vars) fn\_obj = local\_vars.get(fn\_name) if callable(fn\_obj): func\_dict\[fn\_name\] = fn\_obj assert fn\_obj not in fns2code, f"Duplicate function found: {fn\_name}" fns2code\[fn\_obj\] = (fn\_name, cleaned\_code) func\_dict\["finish"\] = finish example = dspy.Example(question=datapoint\["question"\], answer=datapoint\["answer"\], functions=func\_dict) examples.append(example.with\_inputs("question", "functions")) trainset, devset, testset = examples\[:100\], examples\[100:400\], examples\[400:\]

And let's define some helpers for the task. Here, we will define the `metric`, which will be (much) stricter than in the original paper: we'll expect the prediction to match exactly (after normalization) with the ground truth. We'll also be strict in a second way: we'll only allow the agent to take 5 steps in total, to allow for efficient deployment.

In \[4\]:

from func\_timeout import func\_set\_timeout

def wrap\_function\_with\_timeout(fn):
    @func\_set\_timeout(10)
    def wrapper(\*args, \*\*kwargs):
        try:
            return {"return\_value": fn(\*args, \*\*kwargs), "errors": None}
        except Exception as e:
            return {"return\_value": None, "errors": str(e)}

    return wrapper

def fn\_metadata(func):
    signature \= inspect.signature(func)
    docstring \= inspect.getdoc(func) or "No docstring."
    return dict(function\_name\=func.\_\_name\_\_, arguments\=str(signature), docstring\=docstring)

def metric(example, pred, trace\=None):
    gold \= str(example.answer).rstrip(".0").replace(",", "").lower()
    pred \= str(pred.answer).rstrip(".0").replace(",", "").lower()
    return pred \== gold  \# stricter than the original paper's metric!

evaluate \= dspy.Evaluate(devset\=devset, metric\=metric, num\_threads\=24, display\_progress\=True, display\_table\=0, max\_errors\=999)

from func\_timeout import func\_set\_timeout def wrap\_function\_with\_timeout(fn): @func\_set\_timeout(10) def wrapper(\*args, \*\*kwargs): try: return {"return\_value": fn(\*args, \*\*kwargs), "errors": None} except Exception as e: return {"return\_value": None, "errors": str(e)} return wrapper def fn\_metadata(func): signature = inspect.signature(func) docstring = inspect.getdoc(func) or "No docstring." return dict(function\_name=func.\_\_name\_\_, arguments=str(signature), docstring=docstring) def metric(example, pred, trace=None): gold = str(example.answer).rstrip(".0").replace(",", "").lower() pred = str(pred.answer).rstrip(".0").replace(",", "").lower() return pred == gold # stricter than the original paper's metric! evaluate = dspy.Evaluate(devset=devset, metric=metric, num\_threads=24, display\_progress=True, display\_table=0, max\_errors=999)

Now, let's define the agent! The core of our agent will be based on a ReAct loop, in which the model sees the trajectory so far and the set of functions available to invoke, and decides the next tool to call.

To keep the final agent fast, we'll limit its `max_steps` to 5 steps. We'll also run each function call with a timeout.

In \[5\]:

class Agent(dspy.Module):
    def \_\_init\_\_(self, max\_steps\=5):
        self.max\_steps \= max\_steps
        instructions \= "For the final answer, produce short (not full sentence) answers in which you format dates as YYYY-MM-DD, names as Firstname Lastname, and numbers without leading 0s."
        signature \= dspy.Signature('question, trajectory, functions -\> next\_selected\_fn, args: dict\[str, Any\]', instructions)
        self.react \= dspy.ChainOfThought(signature)

    def forward(self, question, functions):
        tools \= {fn\_name: fn\_metadata(fn) for fn\_name, fn in functions.items()}
        trajectory \= \[\]

        for \_ in range(self.max\_steps):
            pred \= self.react(question\=question, trajectory\=trajectory, functions\=tools)
            selected\_fn \= pred.next\_selected\_fn.strip('"').strip("'")
            fn\_output \= wrap\_function\_with\_timeout(functions\[selected\_fn\])(\*\*pred.args)
            trajectory.append(dict(reasoning\=pred.reasoning, selected\_fn\=selected\_fn, args\=pred.args, \*\*fn\_output))

            if selected\_fn \== "finish":
                break

        return dspy.Prediction(answer\=fn\_output.get("return\_value", ''), trajectory\=trajectory)

class Agent(dspy.Module): def \_\_init\_\_(self, max\_steps=5): self.max\_steps = max\_steps instructions = "For the final answer, produce short (not full sentence) answers in which you format dates as YYYY-MM-DD, names as Firstname Lastname, and numbers without leading 0s." signature = dspy.Signature('question, trajectory, functions -\> next\_selected\_fn, args: dict\[str, Any\]', instructions) self.react = dspy.ChainOfThought(signature) def forward(self, question, functions): tools = {fn\_name: fn\_metadata(fn) for fn\_name, fn in functions.items()} trajectory = \[\] for \_ in range(self.max\_steps): pred = self.react(question=question, trajectory=trajectory, functions=tools) selected\_fn = pred.next\_selected\_fn.strip('"').strip("'") fn\_output = wrap\_function\_with\_timeout(functions\[selected\_fn\])(\*\*pred.args) trajectory.append(dict(reasoning=pred.reasoning, selected\_fn=selected\_fn, args=pred.args, \*\*fn\_output)) if selected\_fn == "finish": break return dspy.Prediction(answer=fn\_output.get("return\_value", ''), trajectory=trajectory)

Out of the box, let's assess our `GPT-4o`\-powered agent on the development set.

In \[6\]:

agent \= Agent()
evaluate(agent)

agent = Agent() evaluate(agent)

2025/03/23 21:46:10 INFO dspy.evaluate.evaluate: Average Metric: 105.0 / 300 (35.0%)

Out\[6\]:

35.0

Now, let's optimize the agent using `dspy.SIMBA`, which stands for **Stochastic Introspective Mini-Batch Ascent**. This prompt optimizer accepts arbitrary DSPy programs like our agent here and proceeds in a sequence of mini-batches seeking to make incremental improvements to the prompt instructions or few-shot examples.

In \[ \]:

simba \= dspy.SIMBA(metric\=metric, max\_steps\=12, max\_demos\=10)
optimized\_agent \= simba.compile(agent, trainset\=trainset, seed\=6793115)

simba = dspy.SIMBA(metric=metric, max\_steps=12, max\_demos=10) optimized\_agent = simba.compile(agent, trainset=trainset, seed=6793115)

Having completed this optimization, let's now evaluate our agent again. We see a substantial 71% relative gain, jumping to 60% accuracy.

In \[8\]:

evaluate(optimized\_agent)

evaluate(optimized\_agent)

2025/03/23 21:46:21 INFO dspy.evaluate.evaluate: Average Metric: 182.0 / 300 (60.7%)

Out\[8\]:

60.67
