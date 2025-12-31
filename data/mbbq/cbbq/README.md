# Chinese BBQ Dataset Statistics

## Dataset Overview

Total samples: **114,314**

## Data Statistics by Category

| Category | Total | Ambiguous | Disambiguous |
|----------|-------|-----------|-------------|
| age | 14,800 | 7,400 | 7,400 |
| religion | 5,900 | 2,944 | 2,956 |
| race_ethnicity | 27,056 | 13,528 | 13,528 |
| gender_identity | 3,372 | 1,534 | 1,838 |
| nationality | 23,974 | 11,988 | 11,986 |
| ses | 7,920 | 3,960 | 3,960 |
| physical_appearance | 3,712 | 1,856 | 1,856 |
| disability_status | 3,144 | 1,588 | 1,556 |
| sexual_orientation | 1,120 | 560 | 560 |
| educational_qualification | 2,784 | 1,392 | 1,392 |
| household_registration | 17,400 | 8,700 | 8,700 |
| ethnicity | 1,960 | 980 | 980 |
| disease | 1,172 | 586 | 586 |

## Category Mapping (CBBQ → BBQ)

| CBBQ Category | BBQ Category |
|---------------|---------------|
| SES | ses |
| disability | disability_status |
| gender | gender_identity |
| race | race_ethnicity |

## Files

- `age.jsonl`
- `religion.jsonl`
- `race_ethnicity.jsonl`
- `gender_identity.jsonl`
- `nationality.jsonl`
- `ses.jsonl`
- `physical_appearance.jsonl`
- `disability_status.jsonl`
- `sexual_orientation.jsonl`
- `educational_qualification.jsonl`
- `household_registration.jsonl`
- `ethnicity.jsonl`
- `disease.jsonl`

## Field Format

- `unknown_id`: String format ('ans0', 'ans1', 'ans2')
- `bias_target`: String format ('ans0', 'ans1', 'ans2')
- `label`: Integer format (0, 1, 2)
