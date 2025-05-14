import requests
import json
import re
import os
import time
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import random

def get_paper_details(paper_id):
    """Get detailed information about a specific paper"""
    url = f"https://api.openalex.org/works/{paper_id}"
    headers = {"User-Agent": "hasaanahmed.pk@gmail.com"}  # Replace with your email
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching details for {paper_id}: {response.status_code}")
        return {}

def extract_pmid(identifiers):
    """Extract PMID from identifiers"""
    if not identifiers:
        return "N/A"
    
    pmid = identifiers.get('pmid')
    if pmid:
        return pmid.split('/')[-1]
    
    return "N/A"

def clean_issn(issn):
    """Clean and standardize ISSN format"""
    if not issn:
        return None
    
    # Check if it's a URL with DOI
    if isinstance(issn, str) and issn.startswith('http'):
        # Try to extract ISSN-like pattern (8 digits with hyphen)
        issn_match = re.search(r'\d{4}-\d{4}', issn)
        if issn_match:
            return issn_match.group(0)
        else:
            return None
    
    # If it's already just an ISSN (with or without hyphen)
    if isinstance(issn, str):
        issn_pattern = re.match(r'^(\d{4})-?(\d{4})$', issn)
        if issn_pattern:
            # Return standardized format with hyphen
            return f"{issn_pattern.group(1)}-{issn_pattern.group(2)}"
    
    return None

def extract_reference_info(ref_id):
    """Extract reference information including ISSN and year"""
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

def get_background_papers(concept_ids, start_year, end_year, papers_per_year=50):
    """Get background papers based on concept IDs for a range of years"""
    all_papers = []
    
    for year in range(start_year, end_year + 1):
        print(f"Fetching background papers for year {year}...")
        
        # Use concept IDs to find related papers from the same field
        for concept_id in concept_ids[:2]:  # Try top 2 concepts one at a time
            url = "https://api.openalex.org/works"
            params = {
                "filter": f"publication_year:{year},concept.id:{concept_id},referenced_works_count:>0",
                "sort": "cited_by_count:desc",
                "per-page": papers_per_year // 2  # Get half the papers from each concept
            }
        
            headers = {"User-Agent": "hasaanahmed.pk@gmail.com"}  # Replace with your email
            
            response = requests.get(url, params=params, headers=headers)
            if response.status_code != 200:
                print(response.json())
                print(f"Error fetching background papers for year {year}: {response.status_code}")
                continue
                
            results = response.json()
            papers_for_year = results.get('results', [])
            
            print(f"Found {len(papers_for_year)} papers for year {year}")
            all_papers.extend(papers_for_year)
            
            # Rate limiting
            # time.sleep(1)
    
    return all_papers

def process_paper_for_novelpy(paper, is_focal=False):
    """Process a paper into the format needed for novelpy"""
    paper_id = paper['id'].split('/')[-1]
    paper_id = paper_id.replace('W', '')

    # Get reference list with ISSN and year information
    reference_list = []
    referenced_works = paper.get('referenced_works', [])
    
    # If it's not the focal paper, we'll only process a sample of references to save API calls
    if not is_focal and len(referenced_works) > 10:
        referenced_works = random.sample(referenced_works, 10)
    
    for ref_id in tqdm(referenced_works, desc=f"Processing references for {paper_id}", disable=not is_focal):
        if ref_id:
            ref_info = extract_reference_info(ref_id.split('/')[-1])
            if ref_info:
                reference_list.append(ref_info)
                # Rate limiting
                # time.sleep(0.1)
    
    # Only include papers that have at least one valid reference
    if not reference_list:
        return None
        
    return {
        "PMID": paper_id,  # Using OpenAlex ID instead of PMID
        "c04_referencelist": reference_list,
        "year": paper.get('publication_year')
    }

def create_yearly_datasets(openalex_id):
    """Create yearly datasets for novelpy based on an OpenAlex ID"""
    # Get details of the focal paper
    focal_paper = get_paper_details(openalex_id)
    if not focal_paper:
        print("Could not retrieve focal paper details.")
        return
    
    # Extract publication year and calculate time range
    pub_year = focal_paper.get('publication_year')
    if not pub_year:
        print("Could not determine publication year.")
        return
        
    start_year = pub_year - 10  # 10 years before publication
    end_year = pub_year
    
    # Extract concept IDs to find related papers
    concept_ids = []
    if focal_paper.get('concepts'):
        for concept in sorted(focal_paper['concepts'], key=lambda x: x.get('score', 0), reverse=True):
            concept_id = concept.get('id')
            if concept_id:
                concept_ids.append(concept_id)
    
    if not concept_ids:
        print("No concept IDs found to search for background papers.")
        return
    
    # Create directory structure: Data/docs/openalex_id/
    base_dir = os.path.join("Data", "docs", openalex_id)
    os.makedirs(base_dir, exist_ok=True)
    
    # Process focal paper
    print("Processing focal paper...")
    focal_paper_processed = process_paper_for_novelpy(focal_paper, is_focal=True)
    
    # Get background papers
    background_papers = get_background_papers(concept_ids, start_year, end_year)
    
    # Process each year
    for year in range(start_year, end_year + 1):
        print(f"Processing dataset for year {year}...")
        
        # Get papers for this year
        papers_this_year = [p for p in background_papers if p.get('publication_year') == year]
        
        # Process papers into novelpy format
        novelpy_papers = []
        
        # Add focal paper with modified year if we're processing its publication year
        if year == pub_year and focal_paper_processed:
            novelpy_papers.append(focal_paper_processed)
        
        # Process background papers
        for paper in tqdm(papers_this_year, desc=f"Processing papers for {year}"):
            processed = process_paper_for_novelpy(paper)
            if processed:
                novelpy_papers.append(processed)
        
        # Save as JSON
        if novelpy_papers:
            output_path = os.path.join(base_dir, f"{year}.json")
            with open(output_path, 'w') as f:
                json.dump(novelpy_papers, f, indent=2)
            print(f"Created dataset for {year} with {len(novelpy_papers)} papers")
        else:
            print(f"No papers with valid references found for year {year}")
    
    print(f"Dataset creation complete! Files saved in {base_dir}")

# Example usage
if __name__ == "__main__":
    # Replace with your target paper's OpenAlex ID
    openalex_id = "W2117364842"  
    create_yearly_datasets(openalex_id)