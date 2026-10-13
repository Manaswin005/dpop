# Phase 2 — Feature selection (training set only)

| Feature | Type | Levels | Information gain (MI) | Chi-square p | Decision | Reason |
|---|---|---|---|---|---|---|
| Approval FY | numerical | 63 | 0.1634 | 0.00e+00 | KEEP |  |
| Lending Project Cost | numerical | 2273 | 0.1163 | 3.02e-149 | KEEP |  |
| lending_cost_reported | numerical | 2 | 0.1086 | 0.00e+00 | KEEP |  |
| Country Name | categorical | 179 | 0.0840 | 1.49e-118 | KEEP |  |
| Lending Instrument | categorical | 17 | 0.0480 | 4.64e-146 | KEEP |  |
| Sector Board | categorical | 27 | 0.0416 | 2.25e-112 | KEEP |  |
| Agreement Type | categorical | 9 | 0.0224 | 4.19e-73 | KEEP |  |
| Region | categorical | 7 | 0.0165 | 7.88e-53 | DROP | redundant with Country Name (Cramér's V=1.00) |
| Product Line Code | categorical | 10 | 0.0124 | 4.71e-37 | DROP | redundant with Agreement Type (Cramér's V=0.94) |
| Lending Instrument Type | categorical | 3 | 0.0066 | 5.49e-24 | DROP | redundant with Lending Instrument (Cramér's V=1.00) |

**Numeric correlation (Pearson, training set; log10 cost, reported values only):**

|                       |   Approval FY |   Lending Project Cost |   lending_cost_reported |
|:----------------------|--------------:|-----------------------:|------------------------:|
| Approval FY           |          1    |                  -0.05 |                    0.84 |
| Lending Project Cost  |         -0.05 |                   1    |                  nan    |
| lending_cost_reported |          0.84 |                 nan    |                    1    |

**Selected:** Approval FY, Lending Project Cost, lending_cost_reported, Country Name, Lending Instrument, Sector Board, Agreement Type.

Note: high information gain for features with many levels (e.g. Country Name) is partly an artefact of the number of categories; models limit this by grouping rare categories.