DSPy's biggest strength is also the reason it can admittedly be hard to wrap your head around it.

It's basically say: LLMs & their methods will continue to improve but not equally in every axis, so:

- What's the smallest set of fundamental abstractions that allow you to build downstream AI software that is "future-proof" and rides the tide of progress?

- Equivalently, what are the right algorithmic problems that researchers should focus on to enable as much progress as possible for AI software?

But this is necessarily complex, in the sense that the answer has to be composed of a few things, not one concept only. (Though if you had to understand one concept only, the fundamental glue is DSPy Signatures.)

It's actually only a handful of bets, though, not too many. I've been tweeting them non-stop since late 2022, but I've never collected them in one place.

All of these have proven beyond a doubt to have been the right bets so far for 2.5 years, and I think they'll stay the right bets for the next 3 years at least.

1) Information Flow is the single most key aspect of good AI software.

As foundation models improve, the bottleneck becomes basically whether you can actually (1) ask them the right question and (2) provide them with all the necessary context to address it.

Since 2022, DSPy addressed this in two directions: (i) free-form control flow ("Compound AI Systems" / LM programs) and (ii) Signatures.

Prompts have been a massive distraction here, with people thinking they need to find the magical keyword to talk to LLMs. From 2022, DSPy put the focus on *Signatures* (back then called Templates) which force you to break down LM interactions into *structured and named* input fields and *structured and named output fields*.

Getting simply those fields right was (and has been) a lot more important than "engineering" the "right prompt". That's the point of Signatures. (We know it's hard for people to force them to define their signatures so carefully, but if you can't do that, your system is going to be bad.)

2) Interactions with LLMs should be Functional and Structured.

Again, prompts are bad. People are misled from their chat interaction with LLMs to think that LLMs should take "strings", hence the magical status of "prompts".

But actually, you should define a functional contract. What are the things you will give to the function? What is the function supposed to do with them? What is it then supposed to give you back?

This is again Signatures. It's (i) structured *inputs*, (ii) structured *outputs*, and (iii) instructions. You've got to decouple these three things, which until DSP (2022) and really until very recently with mainstream structured outputs, were just meshed together into "prompts".

This bears repeating: your programmatic LLM interactions need to be functions, not strings. Why? Because there are many concerns that are actually not part of the LLM behavior that you'd otherwise need to handle ad-hoc when working with strings:

- How do you format the *inputs* to your LLM into a string?
- How do you separate *instructions* and *inputs* (data)?
- How do you *specify* the output format (string) that your LLM should produce so you can parse it?
- How do you layer on top of this the inference strategy, like CoT or ReAct, without entirely rewriting your prompt?

Signatures solve this. They ask you to *just* specify the input fields, output fields, and task instruction. The rest are the job of Modules and Optimizers, which instantiate Signatures.

3) Inference Strategies should be Polymorphic Modules.

This sounds scary but the point is that all the cool general-purpose prompting techniques or inference-scaling strategies should be Modules, like the layers in DNN frameworks like PyTorch.

Modules are generic functions, which in this case take *any* Signature, and instantiate *its* behavior generically into a well-defined strategy.

This means that we can talk about "CoT" or "ReAct" without actually committing at all to the specific task (Signature) you want to apply them to. This is a huge deal, which again only exists in DSPy.

One key thing that Modules do is that they define *parameters*. What part(s) of the Module are fixed and which parts can be learned?

For example, in CoT, the specific string that asks the model to think step by step could be learned. Or the few-shot examples of thinking step by step should be learnable. In ReAct, demonstrations of good trajectories should be learnable.

4) Specification of your AI software behavior should be decoupled from learning paradigms.

Before DSPy, every time a new ML paradigm came by, we re-wrote our AI software. Oh, we moved from LSTMs to Transformers? Or we moved from fine-tuning BERT to ICL with GPT-3? Entirely new system.

DSPy says: if you write signatures and instantiate Modules, the Modules actually know exactly what about them can be optimized: the LM underneath, the instructions in the prompt, the demonstrations, etc.

The learning paradigms (RL, prompt optimization, program transformations that respect the signature) should be layered on top, with the same frontend / language for expressing the programmatic behavior.

This means that the *same programs* you wrote in 2023 in DSPy can now be optimized with dspy.GRPO, the way they could be optimized with dspy.MIPROv2, the way they were optimized with dspy.BootstrapFS before that.

The second half of this piece is Downstream Alignment or compile-time scaling. Basically, no matter how good LLMs get, they might not perfectly align with your downstream task, especially when your information flow requires multiple modules and multiple LLM interactions.

You need to "compile" towards a metric "late", i.e. after the system is fully defined, no matter how RLHF'ed your models are.

5) Natural Language Optimization is a powerful paradigm of learning.

We've said this for years, like with the BetterTogether optimizer paper, but you need both *fine-tuning* and *coarse-tuning* at a higher level in natural language.

The analogy I use all the time is riding a bike: it's very hard to learn to ride a bike without practice (fine-tuning), but it's extremely inefficient to learn *avoiding to ride the bike on the side walk* from rewards, you want to understand and learn this rule in natural language to adhere ASAP.

This is the source of DSPy's focus on prompt optimizers as a foundational piece here; it's often far superior in sample efficiency to doing policy gradient RL if your problem has the right information flow structure.

That's it. That's the set of core bets DSPy has made since 2022/2023 until today. Compiling Declarative AI Functions into LM Calls, with Signatures, Modules, and Optimizers.

1) Information Flow is the single most key aspect of good AI software.
2) Interactions with LLMs should be Functional and Structured.
3) Inference Strategies should be Polymorphic Modules.
4) Specification of your AI software behavior should be decoupled from learning paradigms.
5) Natural Language Optimization is a powerful paradigm of learning.
