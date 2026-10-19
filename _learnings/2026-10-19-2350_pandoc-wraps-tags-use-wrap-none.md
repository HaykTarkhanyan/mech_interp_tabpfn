# pandoc HTML output wraps long tags across lines; post-processing regexes need --wrap=none

**Symptom.** Wrapping each level-2 section of a pandoc-rendered review in `<details>` worked for 3 of 5
sections. The other two were left as plain `<section>` elements.

**Cause.** pandoc breaks long tags at the column limit:

```
<section
id="why-robot-foundation-models-need-their-own-interpretability"
class="level2">
```

A regex expecting `<section id="..." class="level2">` on one line misses exactly the sections with long ids.

**Fix.** `pandoc -f markdown -t html --section-divs --wrap=none`. All 21 sections were then wrapped, with
balanced tags (21/21 details, 20/20 sections). See `robotics_foundation_models_explainability/scripts/lit_render.py`.

**Rule.** Any time pandoc HTML is post-processed with regexes, pass `--wrap=none`, then count opening and
closing tags to confirm.
