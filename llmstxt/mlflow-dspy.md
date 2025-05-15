Title: MLflow DSPy Flavor | MLflow

URL Source: https://mlflow.org/docs/latest/llms/dspy/index.html

Markdown Content:
attention

The `dspy` flavor is under active development and is marked as Experimental. Public APIs are subject to change and new features may be added as the flavor evolves.

Introduction[​](https://mlflow.org/docs/latest/llms/dspy/index.html#introduction "Direct link to Introduction")
---------------------------------------------------------------------------------------------------------------

[DSPy](https://dspy-docs.vercel.app/) is a framework for algorithmically optimizing LM prompts and weights. It's designed to improve the process of prompt engineering by replacing hand-crafted prompt strings with modular components. These modules are concise, well-defined, and maintain high quality and expressive power, making prompt creation more efficient and scalable. By parameterizing these modules and treating prompting as an optimization problem, DSPy can adapt better to different language models, potentially outperforming prompts crafted by experts. This modularity also enables easier exploration of complex pipelines, allowing for fine-tuning performance based on specific tasks or nuanced metrics.

![Image 1: Overview of DSPy and MLflow integration](https://mlflow.org/docs/latest/assets/images/dspy-integration-architecture-ee130c78236e38d9728f2a818bf024e9.png)

Why use DSPy with MLflow?[​](https://mlflow.org/docs/latest/llms/dspy/index.html#why-use-dspy-with-mlflow "Direct link to Why use DSPy with MLflow?")
-----------------------------------------------------------------------------------------------------------------------------------------------------

The native integration of the DSPy library with MLflow helps users manage the development lifecycle with DSPy. The following are some of the key benefits of using DSPy with MLflow:

*   [MLflow Tracking](https://mlflow.org/docs/latest/tracking) allows you to track your DSPy program's training and execution. With the MLflow APIs, you can log a variety of artifacts and organize training runs, thereby increasing visibility into your model performance.
*   [MLflow Model](https://mlflow.org/docs/latest/model) packages your compiled DSPy program along with its dependency versions, input and output interfaces and other essential metadata. This allows you to deploy your compiled DSPy program with ease, knowing that the environment is consistent across different stages of the ML lifecycle.
*   [MLflow Evaluate](https://mlflow.org/docs/latest/llms/llm-evaluate) provides native capabilities within MLflow to evaluate GenAI applications. This capability facilitates the efficient assessment of inference results from your DSPy compiled program, ensuring robust performance analytics and facilitating quick iterations.
*   [MLflow Tracing](https://mlflow.org/docs/latest/tracing) is a powerful observability tool for monitoring and debugging what happens inside the DSPy models, helping you identify potential bottlenecks or issues quickly. With its powerful automatic logging capability, you can instrument your DSPy application without needing to add any code apart from running a single command.

Getting Started[​](https://mlflow.org/docs/latest/llms/dspy/index.html#getting-started "Direct link to Getting Started")
------------------------------------------------------------------------------------------------------------------------

In this introductory tutorial, you will learn the most fundamental components of DSPy and how to leverage the integration with MLflow to store, retrieve, and use a DSPy program.

Concepts[​](https://mlflow.org/docs/latest/llms/dspy/index.html#concepts "Direct link to Concepts")
---------------------------------------------------------------------------------------------------

#### `Module`[​](https://mlflow.org/docs/latest/llms/dspy/index.html#module "Direct link to module")

[Modules](https://dspy.ai/learn/programming/modules) are components that handle specific text transformations, like answering questions or summarizing. They replace traditional hand-written prompts and can learn from examples, making them more adaptable.

#### `Signature`[​](https://mlflow.org/docs/latest/llms/dspy/index.html#signature "Direct link to signature")

A [signature](https://dspy.ai/learn/programming/signatures) is a natural language description of a module's input and output behavior. For example, _"question -> answer"_ specifies that the module should take a question as input and return an answer.

#### `Optimizer`[​](https://mlflow.org/docs/latest/llms/dspy/index.html#optimizer "Direct link to optimizer")

A [optimizer](https://dspy.ai/learn/optimization/optimizers) improves LM pipelines by adjusting modules to meet a performance metric, either by generating better prompts or fine-tuning models.

#### `Program`[​](https://mlflow.org/docs/latest/llms/dspy/index.html#program "Direct link to program")

A program is a a set of modules connected into a pipeline to perform complex tasks. DSPy programs are flexible, allowing you to optimize and adapt them using the compiler.

Automatic Tracing[​](https://mlflow.org/docs/latest/llms/dspy/index.html#automatic-tracing "Direct link to Automatic Tracing")
------------------------------------------------------------------------------------------------------------------------------

![Image 2: DSPy Tracing via autolog](https://mlflow.org/docs/latest/assets/images/dspy-tracing-957d61580cca35522155c70e79cdbe42.gif)

[MLflow Tracing](https://mlflow.org/docs/latest/tracing) tracing is a powerful feature that allows you to monitor and debug your DSPy programs. With MLflow, you can enable auto tracing just by calling the [`mlflow.dspy.autolog()`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.dspy.html#mlflow.dspy.autolog) function in your code.

`import mlflowmlflow.dspy.autolog()`

Once enabled, MLflow will generate traces whenever your DSPy program is executed and record them in your MLflow Experiment.

Learn more about MLflow DSPy tracing capabilities [here](https://mlflow.org/docs/latest/tracing/integrations/dspy).

Tracking DSPy Program in MLflow Experiment[​](https://mlflow.org/docs/latest/llms/dspy/index.html#tracking-dspy-program-in-mlflow-experiment "Direct link to Tracking DSPy Program in MLflow Experiment")
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#### Creating a DSPy Program[​](https://mlflow.org/docs/latest/llms/dspy/index.html#creating-a-dspy-program "Direct link to Creating a DSPy Program")

The [Module](https://dspy.ai/learn/programming/modules) object is the centerpiece of the DSPy and MLflow integration. With DSPy, you can create complex agentic logic via a module or set of modules.

`pip install mlflow dspy -U`

`import dspy# Define our language modellm = dspy.LM(model="openai/gpt-4o-mini", max_tokens=250)dspy.settings.configure(lm=lm)# Define a Chain of Thought moduleclass CoT(dspy.Module):    def __init__(self):        super().__init__()        self.prog = dspy.ChainOfThought("question -> answer")    def forward(self, question):        return self.prog(question=question)dspy_model = CoT()`

tip

Typically you'd want to leverage a [compiled DSPy](https://dspy.ai/learn/optimization/optimizers#what-does-a-dspy-optimizer-tune-how-does-it-tune-them) module. MLflow will natively supports logging both compiled and uncompiled DSPy modules. Above we show an uncompiled version for simplicity, but in production you'd want to leverage an [optimizer](https://dspy.ai/learn/optimization/optimizers) and log the outputted object instead.

#### Logging the Program to MLflow[​](https://mlflow.org/docs/latest/llms/dspy/index.html#logging-the-program-to-mlflow "Direct link to Logging the Program to MLflow")

You can log the `dspy.Module` object to an MLflow run using the [`mlflow.dspy.log_model()`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.dspy.html#mlflow.dspy.log_model) function.

We will also specify a [model signature](https://mlflow.org/docs/latest/model/signatures#input-example). An MLflow model signature defines the expected schema for model inputs and outputs, ensuring consistency and correctness during model inference.

`import mlflow# Start an MLflow runwith mlflow.start_run():    # Log the model    model_info = mlflow.dspy.log_model(        dspy_model,        artifact_path="model",        input_example="what is 2 + 2?",    )`

![Image 3: MLflow artifacts for the DSPy program](https://mlflow.org/docs/latest/assets/images/dspy-artifacts-e5f816ebc8919118392f3e066edece68.png)

#### Loading the Module for inference[​](https://mlflow.org/docs/latest/llms/dspy/index.html#loading-the-module-for-inference "Direct link to Loading the Module for inference")

The saved module can be loaded back for inference using the [`mlflow.pyfunc.load_model()`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html#mlflow.pyfunc.load_model) function. This function gives an MLflow Python Model backed by the DSPy module.

`import mlflow# Load the model as an MLflow PythonModelmodel = mlflow.pyfunc.load_model(model_info.model_uri)# Predict with the objectresponse = model.predict("What kind of bear is best?")print(response)`

Output

`{    "reasoning": """The question "What kind of bear is best?" is often associated with a    humorous reference from the television show "The Office," where the character Jim    Halpert jokingly states, "Bears, beets, Battlestar Galactica." However, if we consider    the question seriously, it depends on the context. Different species of bears have    different characteristics and adaptations that make them "best" in various ways.    For example, the American black bear is known for its adaptability, while the polar bear is    the largest land carnivore and is well adapted to its Arctic environment. Ultimately, the    answer can vary based on personal preference or specific criteria such as strength,    intelligence, or adaptability.""",    "answer": """There isn\'t a definitive answer, as it depends on the context. However, many    people humorously refer to the American black bear or the polar bear when discussing    "the best" kind of bear.""",}`

To load the DSPy program itself back instead of the PyFunc-wrapped model, use the [`mlflow.dspy.load_model()`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.dspy.html#mlflow.dspy.load_model) function.

`model = mlflow.dspy.load_model(model_uri)`

Optimizer Autologging[​](https://mlflow.org/docs/latest/llms/dspy/index.html#optimizer-autologging "Direct link to Optimizer Autologging")
------------------------------------------------------------------------------------------------------------------------------------------

The MLflow DSPy flavor supports autologging for DSPy optimizers. See the [Optimizer Autologging](https://mlflow.org/docs/latest/llms/dspy/optimizer) page for details.

FAQ[​](https://mlflow.org/docs/latest/llms/dspy/index.html#faq "Direct link to FAQ")
------------------------------------------------------------------------------------

### How can I save a compiled vs. uncompiled model?[​](https://mlflow.org/docs/latest/llms/dspy/index.html#how-can-i-save-a-compiled-vs-uncompiled-model "Direct link to How can I save a compiled vs. uncompiled model?")

DSPy compiles models by updating various LLM parameters, such as prompts, hyperparameters, and model weights, to optimize training. While MLflow allows logging both compiled and uncompiled models, it's generally preferable to use a compiled model, as it is expected to perform better in practice.

### What can be serialized by MLflow?[​](https://mlflow.org/docs/latest/llms/dspy/index.html#what-can-be-serialized-by-mlflow "Direct link to What can be serialized by MLflow?")

When using [`mlflow.dspy.log_model()`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.dspy.html#mlflow.dspy.log_model) or [`mlflow.dspy.save_model()`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.dspy.html#mlflow.dspy.save_model) in MLflow, the DSPy program is serialized and saved to the tracking server as a `.pkl` file. This enables easy deployment. Under the hood, MLflow uses `cloudpickle` to serialize the DSPy object, but some DSPy artifacts are note serializable. Relevant examples are listed below.

*   API tokens. These should be managed separately and passed securely via environment variables.
*   The DSPy trace object, which is primarily used during training, not inference.

### How do I manage secrets?[​](https://mlflow.org/docs/latest/llms/dspy/index.html#how-do-i-manage-secrets "Direct link to How do I manage secrets?")

When serializing using the MLflow DSPy flavor, tokens are dropped from the settings objects. It is the user's responsibility to securely pass the required secrets to the deployment environment.

### How is the DSPy `settings` object saved?[​](https://mlflow.org/docs/latest/llms/dspy/index.html#how-is-the-dspy-settings-object-saved "Direct link to how-is-the-dspy-settings-object-saved")

To ensure program reproducibility, the service context is converted to a Python dictionary and pickled with the model artifact. Service context is a concept that has been popularized in GenAI frameworks. Put simply, it stores a configuration that is global to your project. For DSPy specifically, we can set information such as the language model, reranker, adapter, etc.

DSPy stores this service context in a `Settings` singleton class. Sensitive API access keys that are set within the `Settings` object are not persisted when logging your model. When deploying your DSPy model, you must ensure that the deployment environment has these keys set so that your DSPy model can make remote calls to services that require access keys.