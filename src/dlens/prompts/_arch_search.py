# prompts/_arch_search.py
"""Prompts for the tree-based architecture search (generator + judge).

Deliberately free of domain preferences: no architecture family is hinted as
"known best" — the research point is what the search converges to on its own.
"""

from __future__ import annotations

ARCH_GENERATOR_PROMPT = """\
You propose candidate neural-network architectures for an image-classification
task, as structured ArchitectureSpec objects.

BUILDABLE SPACE (propose ONLY within this — anything else is discarded):
- family "resnet": BasicBlock residual network. `depths` = blocks per stage,
  `widths` = channels per stage. 2-4 stages; each depth 1-6; each width 8-1024.
- family "cnn": plain VGG-like convnet. `depths` = 3x3 convs per stage, `widths`
  = channels per stage, maxpool between stages. Same bounds.
- Always set: name (a short descriptive label), family, input_shape, channels,
  num_classes, depths, widths. Do not use other families.

Propose the requested number of candidates. Make them genuinely DIVERSE: vary
family, capacity (parameter count), depth vs width balance, small vs large.
Reason from the task characteristics you are given (image size, classes, sample
count, and any observed failure modes like overfitting) — smaller datasets often
punish very large models, but explore the space rather than assuming one answer.
In each candidate's `rationale`, say in one sentence why it might fit. In your
`reasoning`, summarize the diversity of the set.\
"""

ARCH_JUDGE_PROMPT = """\
You are ranking candidate neural-network architectures for an image-classification
task, BEFORE any of them are trained. You get the task characteristics and the
candidate specs (family, depths, widths, approximate parameter counts).

Rank ALL candidates from most to least promising for held-out generalization on
this task. Consider capacity vs dataset size, depth/width balance, and family
differences — judge purely from the specs and task numbers given; do not assume
any family is inherently best. Return `ranking` as the list of candidate indices,
best first, and justify the top picks in `reasoning`.\
"""
