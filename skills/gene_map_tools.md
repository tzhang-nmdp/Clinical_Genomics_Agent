# Skill: gene_map — Resolve gene name -- id to genomic information

## Tool
`gene_map_tools`

## When to Use
Use this skill whenever you need genomic information for a gene id.

## Steps

### Step 1 — Pass gene name -- id
Pass the gene name -- id as-is (e.g. `"TP53"`).

### Step 2 — Search in the genomic database table
Search the gene name -- id in the genomic database table in `r"C:\Users\tzhang\Desktop\Project\DP-AI\LangChain-OpenTutorial-main\tools\dbNSFP4.0_gene.complete"`

### Step 3 — return the output to user
return the output to user

## Rules
- If geocoding returns no result, ask the user to double check gene id.
