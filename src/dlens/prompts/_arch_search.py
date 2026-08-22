# prompts/_arch_search.py
"""Prompts for the tree-based architecture search (generator + judge).

Deliberately free of domain preferences: no architecture family is hinted as
"known best" — the research point is what the search converges to on its own.

The buildable space was broadened on 2026-08-13 from two convolutional families
to six. The bias rule is unchanged and applies to the new families exactly as it
did to the old ones: this prompt states what CAN BE BUILT and how the fields are
interpreted, and never which family is expected to do well. In particular it does
not mention that lensed arcs are rotationally symmetric, that convolutions have
been used in this domain before, or that any family suits this task — those are
precisely the priors the experiment is designed to withhold.
"""

from __future__ import annotations

ARCH_GENERATOR_PROMPT = """\
You propose candidate neural-network architectures for an image-classification
task, as structured ArchitectureSpec objects.

BUILDABLE SPACE (propose ONLY within this — anything else is discarded):

Every candidate sets: name (a short descriptive label), family, input_shape,
channels, num_classes, depths, widths. `depths` has 2-4 entries, each 1-6;
`widths` has the same length, each 8-1024. What those two lists mean depends on
the family:

- family "cnn": plain VGG-like convnet. `depths[i]` = 3x3 convolutions in stage
  i, `widths[i]` = channels in stage i, 2x2 max-pool between stages.
- family "resnet": BasicBlock residual network. `depths[i]` = residual blocks in
  stage i, `widths[i]` = channels in stage i.
- family "vit": patch-embedding vision transformer over the whole image.
  `sum(depths)` = number of transformer blocks; `widths[-1]` = embedding
  dimension. Attention heads are chosen automatically from the dimension.
- family "mlpmixer": patch embedding followed by alternating token-mixing and
  channel-mixing MLPs. `sum(depths)` = number of mixer blocks; `widths[-1]` =
  hidden dimension.
- family "hybrid": convolutional stem followed by transformer blocks over the
  resulting feature grid. `depths[:-1]`/`widths[:-1]` describe the conv stages;
  `depths[-1]` = number of attention blocks; `widths[-1]` = attention dimension.
- family "equivariant": C4 group-equivariant convnet. Weights are shared across
  the four 90-degree rotations of each kernel and the orientation axis is cycled,
  so the features are exactly invariant to 90-degree rotations of the input.
  `depths[i]` = group-conv blocks in stage i, `widths[i]` = base channels in
  stage i (each carries 4 orientation channels, so the parameter count is about
  four times a plain convolution of the same width, and it is slower to train
  per step). Set physics_informed=true for this family.

Reachable parameter counts span roughly 0.02M to 30M+ in every family, so
capacity is a free choice rather than a consequence of picking a family.

Propose the requested number of candidates. Make them genuinely DIVERSE: vary
family, capacity (parameter count), depth vs width balance, small vs large. Do
not concentrate the set in one family. Reason from the task characteristics you
are given (image size, classes, sample count, and any observed failure modes like
overfitting) — smaller datasets often punish very large models, but explore the
space rather than assuming one answer. No family is known in advance to suit this
task; that is what the search is measuring.

In each candidate's `rationale`, say in one sentence why it might fit. In your
`reasoning`, summarize the diversity of the set.\
"""

# Appended to the generator prompt for rounds 2..R of the iterative search. The
# history is the search's OWN measurements, which is evidence it produced rather
# than a prior we injected — the bias rule is about withholding our preferences,
# not about withholding the experiment's results from itself.
ARCH_GENERATOR_ITERATIVE_SUFFIX = """\

MEASURED RESULTS SO FAR (short training runs on this exact task, all rounds):
{history}

Propose exactly {n} NEW candidates informed by those measurements. You may refine
what appears to be working, abandon what is not, or explore a region nobody has
tried — that judgement is yours to make from the numbers. Every proposal must
differ from all architectures listed above in family, depths, or widths; exact
repeats are discarded. The measurements above are the only evidence you have been
given about this task, and no family is designated as preferred.\
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
