# PRISMA "included from the search" must be matched against the screened shortlist, not the whole corpus

**Symptom.** The synthetic-data review's PRISMA box said "top 700 -> 63 included from the search". The
independent review found that 9 of those 63 never went through manual screening.

**Cause.** `lit_references.py` set `found_by_search` by title-matching against `screened_all.json`, which holds
all 3,067 records, including those that failed the automated screen. Recount
(`synthetic_data/scripts/lit_prisma.py`, 2026-10-20):

```
in the manually screened top-700 shortlist   54
passed the screen, ranked below the cut       3   candes2018, hooker2021, knaus2020
in the corpus, failed the automated screen    6   aas2021, advani2019, freiesleben2024, krishna2024, nauta2023, neal2020
not retrieved at all                         18
```

Two of the six screened-out works are central to the review (Advani et al., RealCause). A second, smaller
mislabel: 32 records dropped for an empty title were counted as duplicates (483 true duplicates, not 515).

**Fix.** `lit_prisma.py` now routes each reference against the shortlist, then the passing records, then the
corpus, and reports "54 included through screening" plus "27 added outside screening". `lit_screen.py` counts
untitled drops separately.

**Consequences.** "Retrieved" and "screened in" are different PRISMA quantities; derive each from the file
that represents that stage. Report the automated screen's misses as a limitation.
