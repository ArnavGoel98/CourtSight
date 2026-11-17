# Corrections

Every error found in CourtSight's published figures, newest first, with what changed. The figures quoted are the
ones that moved; everything else was rerun and is unchanged. The details of each check are in the
[audit](README.md#auditing-our-own-results) and the code is in this repository's history.

## November 2026: where each district's records end

- **The error.** The records were collected district by district, but every case was followed until its *state's*
  collection date. In 158 districts (a quarter of all cases) the records end three months or more before their
  state's date: in Gujarat most districts stop in February 2019 while a few run to December 2019. Following a
  district past its end counts months in which no decision could be recorded, so its cases looked slower than they
  were. In the last month before the end, recorded decisions were 26% of the same month a year earlier with state
  dates, and 85% with district dates.
- **What changed.**
  - Ordinary civil suits: median 30 → 29 months; still pending after 5 years 27% → 26%.
  - All civil cases: median 19 → 18 months. Criminal cases: 7 months, unchanged.
  - Decided cases only vs counting cases still waiting, on the same civil suits: 14 vs 29 → 13 vs 28 months.
  - 139 of 4,293 district three-year figures fell by more than 5 points.
  - Years of waiting: 13.3 → 12.7 crore case-years; the typical pending case 2.3 → 2.0 years old.
- **A warning withdrawn.** West Bengal and Odisha had been flagged as having stale records (53% and 29% of pending
  cases not listed for a hearing in the year before collection), with their delays "possibly overstated". That came
  from the same error: with each district's own date, both are 5%. The warning is gone, and their districts are back
  in the hearings analysis.
- **Hearings.** With those districts back, the link between hearing gaps and delay is stronger: 36% of criminal cases
  still pending after 3 years in the fifth of districts with the longest gaps, against 14% in the shortest-gap fifth
  (previously 31% against 14%). The typical gap is 60 days (previously 56).
- **Caseload.** "Busier courtrooms are if anything slightly faster" is withdrawn: the estimate's range now includes
  zero (−1.3 points per SD, 95% interval −2.8 to +0.2). Caseload per courtroom makes no clear difference.
- **New check.** A second, independent calculation of every national and district figure, written from the rules'
  descriptions without the analysis code, now runs in CI (`tests/test_independent.py`). It agrees on all 10 national and 4,787 district results.

## October 2026: second self-audit

- **One collection date for the whole country.** National tables merged every state's date into one (February 2019).
  Fixed by giving each state its own date (since superseded by district dates, above).
- **Estimates resting on a handful of cases.** Survival curves now stop once fewer than 20 cases are still followed;
  district figures beyond that point are left blank instead of estimated, and "10+ years" is no longer shown for
  durations the data never reaches.
- **A comparison that was not like for like.** "Decided cases only" was compared with a Kaplan–Meier figure that
  included different cases; both are now computed on exactly the same cases.
- **A chart artefact.** The audit chart's 2018 point suggested courts slowed down; 2018 filings had not had a year
  of follow-up. The chart now ends at 2017.
- **Claims that went further than the data.** "The rest are typos" (most late decision dates are plausible later
  updates; 1,412 are impossible); coverage of "eCourts" (it is Development Data Lab's copy of it); Jharkhand's records
  holding 122% of official filings, previously unmentioned; causal wording about caseload; the COVID figure presented
  as caused by COVID rather than as a departure from trend; rank ranges presented without saying they cover sampling
  noise only.

## October 2026: first self-audit

- **Districts that never recorded decisions.** The first map named Gaya, Muzaffarpur and Bhojpur among India's slowest
  courts. Their records hold almost no decisions before 2018 (Gaya: 33 in 2010–2017, 7,969 in 2018). 14 such units
  are now withheld, each with a page explaining why.
- **Digitisation (left truncation).** Cases filed before a district's records became complete were recorded mostly if
  still pending, which made early years look slow. Corrected with delayed entry: civil-suit median 32 → 29 months, and
  one district moved 278 places in the civil-suit ranking.
- **Units that are not districts.** Two special courts in Maharashtra were missed by the first pass and counted as
  districts.
- **The "speed-up".** An earlier explanation (records filling in) stopped holding once the missing-decision fix was
  made, and the text was changed to say the cause is open.

## Results never published as findings

- **Judge handovers.** A 10% drop in disposals after a judge changes looked real, but placebo dates with no change
  show the same drop. Reported as no reliable estimate.
- **Forecasting skill.** The district model first looked 30% better than its benchmark; the benchmark flattered it.
  Against strong simple forecasts it ties on the 2019 official figures, and is reported as a tie.
