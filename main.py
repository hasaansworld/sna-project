import requests
import pandas as pd
import json
import time
import re

# Function to get papers from OpenAlex with pagination
def get_papers(query, min_cited_by_count, max_cited_by_count, per_page=25):
    url = "https://api.openalex.org/works"
    params = {
        "filter": f"title.search:\"{query}\",cited_by_count:>{min_cited_by_count},cited_by_count:<{max_cited_by_count}",
        "sort": "cited_by_count:desc",
        "per-page": per_page
    }
    response = requests.get(url, params=params)
    return response.json()

# Function to get detailed paper info including references
def get_paper_details(paper_id):
    url = f"https://api.openalex.org/works/{paper_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching details for {paper_id}: {response.status_code}")
        return {}

# Function to extract PMID from identifiers
def extract_pmid(identifiers):
    if not identifiers:
        return "N/A"
    
    pmid = identifiers.get('pmid')
    if pmid:
        return pmid.split('/')[-1]
    
    return "N/A"

# Function to clean and extract ISSN
def clean_issn(issn):
    if not issn:
        return None
    
    # Check if it's a URL with DOI
    if issn.startswith('http'):
        # Try to extract ISSN-like pattern (8 digits with hyphen)
        issn_match = re.search(r'\d{4}-\d{4}', issn)
        if issn_match:
            return issn_match.group(0)
        else:
            # For DOIs, we'll return None and try to find an ISSN elsewhere
            return None
    
    # If it's already just an ISSN (with or without hyphen)
    issn_pattern = re.match(r'^(\d{4})-?(\d{4})$', issn)
    if issn_pattern:
        # Return standardized format with hyphen
        return f"{issn_pattern.group(1)}-{issn_pattern.group(2)}"
    
    return None

# Function to safely extract reference information
def extract_reference_info(ref_id):
    try:
        ref_details = get_paper_details(ref_id)
        if not ref_details:
            return None
        
        # Try multiple paths to find ISSN
        issn = None
        
        # Try primary_location.source.issn_l
        if ref_details.get('primary_location') and ref_details['primary_location'].get('source'):
            issn = clean_issn(ref_details['primary_location']['source'].get('issn_l'))
        
        # If not found, try host_venue.issn_l
        if not issn and ref_details.get('host_venue'):
            issn = clean_issn(ref_details['host_venue'].get('issn_l'))
            
        # Try host_venue.issn
        if not issn and ref_details.get('host_venue') and ref_details['host_venue'].get('issn'):
            for possible_issn in ref_details['host_venue']['issn']:
                cleaned = clean_issn(possible_issn)
                if cleaned:
                    issn = cleaned
                    break
        
        # Get year
        year = ref_details.get('publication_year')
        
        if issn and year:
            return {"item": issn, "year": year}
        return None
    except Exception as e:
        print(f"Error processing reference {ref_id}: {e}")
        return None

# Get papers with different citation counts
high_cited = get_papers("Social Media in Emergency Management", 100, 1000, 2)
medium_cited = get_papers("Social Media in Emergency Management", 20, 99, 2)
low_cited = get_papers("Social Media in Emergency Management", 1, 19, 2)

# Process each set of papers
all_papers = []

paper_sets = [
    ("High cited", high_cited),
    ("Medium cited", medium_cited),
    ("Low cited", low_cited)
]

for set_name, paper_set in paper_sets:
    print(f"\n{set_name} papers:")

    if 'results' not in paper_set:
        print(paper_set)  # Print error message if any
        continue
    
    for paper in paper_set['results']:
        paper_id = paper.get('id')
        title = paper.get('display_name', 'No title')
        cited_by_count = paper.get('cited_by_count', 'N/A')
        year = paper.get('publication_year', 'N/A')
        

        print(f"Processing: {title}")
        print(paper_id)
        
        # Get detailed information including references
        paper_details = get_paper_details(paper_id)
        
        # Extract PMID
        identifiers = paper_details.get('ids', {})
        
        paper_id = paper_details['id'].split('/')[-1]
        
        paper_info = {
            "title": title,
            "cited_by_count": cited_by_count,
            "year": year,
            "pmid": paper_id,
        }
        
        all_papers.append(paper_info)
        
        # Print summary
        print(f"- {title} (cited by: {cited_by_count}, year: {year}, PMID: {paper_id})")

# Save all papers to a JSON file
with open('papers_with_references.json', 'w') as f:
    json.dump(all_papers, f, indent=2)

print("\nAll papers saved to papers_with_references.json")