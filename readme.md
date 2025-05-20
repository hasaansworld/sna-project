# Novelty Estimation in Citation Network Research

This project explores novelty metrics associated with highly cited papers in the field of "Social Media in Emergency Management." It analyzes how citation patterns, reference list characteristics, and novelty indicators correlate with a paper's citation impact.

## Project Overview

The project comprises several analytical components:
- Retrieving and analyzing papers from OpenAlex based on citation counts
- Analyzing reference lists and their citation statistics
- Examining correlation between reference list metrics and paper citations
- Investigating the relationship between topic diversity and citation impact
- Implementing and analyzing four novelty metrics using the Novelpy package

## Repository Structure

```
.
├── main.py                 # Main script for steps 1-4 (OpenAlex analysis)
├── create_dataset.py       # Creates dataset for Novelpy analysis
├── analysis.py             # Runs novelty indicators on created dataset
├── plot.py                 # Visualizes results from analysis.py
└── Data/                   # Directory containing datasets
    └── docs/               # Paper-specific datasets
        └── [paper_id]/     # Dataset for specific paper
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/citation-network-novelty.git
cd citation-network-novelty
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Install Novelpy package:
```bash
pip install novelpy
```

## Usage

### 1. Main Analysis (Steps 1-4)

The `main.py` script performs the following operations:
- Retrieves the 6 papers from OpenAlex (2 most cited, 2 least cited, 2 middle range)
- Analyzes reference lists and calculates citation statistics
- Computes correlations between paper citations and reference list metrics
- Calculates topic diversity and its correlation with citations

```bash
python main.py
```

### 2. Novelty Analysis (Step 5)

This process consists of three scripts that should be run sequentially:

#### a. Create dataset for a specific paper

```bash
python create_dataset.py
```
*Note: Edit the main function at the end of the script to specify the paper's OpenAlex ID*

This script:
- Creates a dataset for Novelpy analysis based on the paper's two main concepts
- Saves the dataset in `Data/docs/[paper_id]/`

#### b. Run novelty analysis

```bash
python analysis.py --paper_id W123456789 --start_year 2010 --end_year 2020
```

This script:
- Calculates four novelty indicators (Atypicality, Commonness, Bridging, Novelty)
- Uses the Uzzi2013, Lee2015, Foster2015, and Wang2017 methods from Novelpy
- Analyzes for each year in the specified range

#### c. Visualize results

```bash
python plot.py --start_year 2010 --end_year 2020
```

This script:
- Plots the most recent results from the analysis
- *Important: Use the same year range as in the analysis step*

## Methodology Details

### Paper Selection
The project analyzes six papers in the field of "Social Media in Emergency Management":
- Two papers with the highest citation counts
- Two papers with the lowest citation counts
- Two papers with citation counts in the middle range

### Reference List Analysis
For each paper, we analyze:
- Mean citation score of references
- Maximum citation score among references
- Minimum citation score among references
- Correlation between these metrics and the paper's own citation count

### Topic Diversity Analysis
- Extracts topics for each reference paper using OpenAlex
- Performs union operation to find total distinct topics
- Calculates correlation between topic diversity and citation count

### Novelty Metrics
Using Novelpy, we calculate four indicators for each paper over a 10-year period:
1. **Atypicality**: Measures how unusual the paper's combination of references is
2. **Commonness**: Evaluates how conventional the paper's reference patterns are
3. **Bridging**: Assesses the paper's ability to connect previously disconnected knowledge areas
4. **Novelty**: Quantifies the overall innovative nature of the paper

## Requirements

- Python 3.8+
- requests
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- novelpy

## Limitations and Considerations

- Analysis is limited to papers available in the OpenAlex database
- Citation counts may vary depending on when the data is accessed
- Novelty metrics are influenced by the quality of the created dataset
- The focal year selection can significantly impact novelty calculations

## Contributing

If you wish to contribute to this project, please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with a clear description of your changes

## License

[Specify your license here]

## Acknowledgments

- OpenAlex for providing the citation data API
- Novelpy developers for implementing the novelty metrics
- [Any other acknowledgments]
