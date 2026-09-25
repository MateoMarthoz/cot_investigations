# Teacher-forced analysis of model reasoning

This repository is a follow-up to [Thought Circuits](https://github.com/MateoMarthoz/thought-circuits), my first attempt at research. That project tried to automate causal analysis of chains of thought before I had manually explored the problem enough to understand what kinds of analyses were informative, where they failed, or whether teacher-forced analysis was promising.

Here I instead investigate individual reasoning traces in detail, testing what teacher-forced interventions can tell us and comparing their conclusions with existing work where possible.

## Motivation

Chains of thought can be long and explore many different explanations, plans, objections and alternatives. When a model eventually takes a concerning action, it can be unclear which parts of that reasoning actually mattered, or even whether the behaviour reflects the kind of misalignment it initially appears to. This is closely related to the motivation behind [Model Forensics](https://arxiv.org/abs/2606.26071).

I am exploring whether interventions on a fixed CoT can help identify which thoughts causally support later reasoning and decisions, and where this approach stops being informative.

## Traces

### Qwen3-32B blackmail

From [INTENT-AS-A-TOOL](https://arxiv.org/abs/2608.27348) ([code](https://github.com/RebeccaZhang22/intent-as-a-tool)). The model learns that it will be replaced by a system with a conflicting objective, discovers private information about the person able to stop the replacement, and eventually declares blackmail intent.

The current evidence suggests a distributed causal story: the stated goal contributes to treating replacement as a goal conflict, the private information contributes to representing blackmail as effective leverage, and permissive judgements contribute to later coercive planning. The analysis does not identify a single decisive motive. These results broadly agree with the intent trajectory reported in INTENT-AS-A-TOOL.

[Analysis](iaat_blackmail/) · [Report](iaat_blackmail/report.md)

### Next: gpt-oss-120b shutdown resistance (`oss-trans-2`)

From [Shutdown Resistance in Large Language Models](https://openreview.net/forum?id=e4bTTqUnJH) ([write-up](https://palisaderesearch.org/research/shutdown-resistance), [trace](https://shutres.fyi/oss-trans-2), [code](https://github.com/PalisadeResearch/shutdown_avoidance)).

The model is explicitly instructed to allow shutdown, but reasons about circumventing the shutdown mechanism while trying to complete its task. The behaviour is clear; its cause is less so. In particular, it is unclear whether the trace reflects something like instrumental self-preservation, strong task-completion pressure, instruction weighting, or some combination.

Current conclusion: not yet analysed.