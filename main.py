import requests
import json
import time
import numpy as np
import pandas as pd
from math import floor, ceil, isnan 
from datetime import datetime
from scipy.stats import pearsonr

API_BASE_URL = "https://api.openalex.org"
HEADERS = {'mailto': 'aqibilyas.pk@gmail.com'} 
SEARCH_QUERY = "Social Media in Emergency management"
API_DELAY_SECONDS = 0.2  # Delay between API calls to be polite

# --- Helper Function for API Calls ---
def make_api_request(endpoint, params=None):
    """Makes a request to the OpenAlex API and returns the JSON response."""
    url = f"{API_BASE_URL}/{endpoint}"
    try:
        params = params or {}
        if 'mailto' not in params and HEADERS.get('mailto'):
            params['mailto'] = HEADERS['mailto']

        # print(f"Debug: Requesting URL: {url} with params: {params}") # Uncomment for debugging
        response = requests.get(url, params=params, timeout=30) # Increased timeout
        response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
        time.sleep(API_DELAY_SECONDS)  # Add delay AFTER successful request
        if 'application/json' in response.headers.get('Content-Type', ''):
            return response.json()
        else:
            print(f"Warning: Non-JSON response received from {url}. Content: {response.text[:200]}")
            return None
    except requests.exceptions.Timeout:
        print(f"Timeout error in API request to {url} with params: {params}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error in API request to {url} with params: {params}: {e}")
        return None
    except json.JSONDecodeError:
        print(f"JSON Decode Error for {url}. Response: {response.text[:200]}")
        return None


# --- Helper to Extract OpenAlex ID ---
def extract_oa_id(url_or_id):
    """Extracts the pure OpenAlex ID (e.g., W12345) from a full URL or if already an ID."""
    if isinstance(url_or_id, str):
        if url_or_id.startswith("https://openalex.org/"):
            # Handles https://openalex.org/W123 or https://openalex.org/works/W123
            return url_or_id.split("/")[-1]
        # Checks if it's already a W-ID like W1234567890
        elif url_or_id.startswith("W") and url_or_id[1:].isdigit():
            return url_or_id
    print(f"Warning: Could not extract valid OpenAlex ID from input: {url_or_id}")
    return None

# --- Helper to handle NaN for JSON serialization ---
def replace_nan_with_none(obj):
    """Recursively replaces NaN values with None in nested dicts/lists."""
    if isinstance(obj, dict):
        return {k: replace_nan_with_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan_with_none(elem) for elem in obj]
    elif isinstance(obj, float) and isnan(obj): # Standard library isnan
        return None
    # Check for numpy float types specifically if numpy is heavily used for data creation
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)) and np.isnan(obj):
        return None
    return obj

# --- Step 1: Retrieve Target Papers (Set P) ---
def perform_step1_paper_retrieval_filtered(query_title):
    """
    Retrieves 2 high, 2 medium, and 2 low cited papers using specific filters,
    matching the user's requested param structure for filters.
    """
    print(f"Step 1: Searching for papers related to '{query_title}' using filtered categories...")
    set_P_info = {}
    processed_ids = set() # To avoid adding duplicate papers

    # Define categories with min_exclusive and max_exclusive citation counts
    # min_exclusive: corresponds to cited_by_count:>{value}
    # max_exclusive: corresponds to cited_by_count:<{value}
    categories = {
        "High Cited": {'min_exclusive': 100, 'max_exclusive': None, 'sort': 'cited_by_count:desc'}, # >100 citations
        "Medium Cited": {'min_exclusive': 19, 'max_exclusive': 100, 'sort': 'cited_by_count:desc'}, # 20-99 citations (i.e., >19 AND <100)
        "Low Cited": {'min_exclusive': 0, 'max_exclusive': 20, 'sort': 'cited_by_count:desc'}     # 1-19 citations (i.e., >0 AND <20)
    }

    for category_name, cat_params in categories.items():
        print(f"\nFetching {category_name} papers...")
        
        filter_parts = [f'title.search:"{query_title}"']
        min_exclusive_val = cat_params.get('min_exclusive')
        max_exclusive_val = cat_params.get('max_exclusive')

        if min_exclusive_val is not None:
            filter_parts.append(f'cited_by_count:>{min_exclusive_val}')
        if max_exclusive_val is not None:
            filter_parts.append(f'cited_by_count:<{max_exclusive_val}')
        
        current_filter_string = ",".join(filter_parts)
        
        params = {
            'filter': current_filter_string,
            'sort': cat_params['sort'],
            'per_page': 2, # We want 2 papers from each category. OpenAlex uses 'per_page'
            'select': 'id,title,publication_year,cited_by_count' # Keep select for efficiency
        }
        
        data = make_api_request('works', params=params)

        if data and 'results' in data and data['results']:
            for paper in data['results']:
                paper_id_url = paper.get('id')
                paper_id = extract_oa_id(paper_id_url)

                if not paper_id:
                    print(f"  Skipping paper, could not extract valid ID from URL: {paper_id_url}")
                    continue
                
                if paper_id in processed_ids:
                    print(f"  Skipping duplicate paper ID: {paper_id}")
                    continue

                title = paper.get('title', 'N/A')
                citations = paper.get('cited_by_count', 0) # Default to 0 if missing
                pub_year = paper.get('publication_year', 'N/A')

                print(f"  + {category_name}: ID: {paper_id}, Year: {pub_year}, Citations: {citations}, Title: {title[:70]}...")
                set_P_info[paper_id] = {
                    'title': title,
                    'citation_count': int(citations) if citations is not None else 0,
                    'publication_year': pub_year,
                    'openalex_id_url': paper_id_url, # Store the full URL for later use if needed
                    'category_step1': category_name
                }
                processed_ids.add(paper_id)
        else:
            print(f"  Could not retrieve papers for {category_name} or no results found.")
            if data and 'meta' in data:
                 print(f"  Meta info for {category_name}: {data['meta']}")


    print(f"\nStep 1 Finished. Total unique papers selected for Set P: {len(set_P_info)}")
    if len(set_P_info) < 2 and len(set_P_info) > 0 : # Check if at least one paper was found
        print("Warning: Fewer than 2 unique papers selected for Set P. Correlation analysis might be limited or skipped.")
    elif len(set_P_info) == 0:
        print("CRITICAL: No papers were selected for Set P. Cannot proceed with analysis.")
        return {} # Return empty dict if no papers

    # Save Set P to a JSON file
    output_filename_step1 = 'step1_selected_papers_set_p.json'
    with open(output_filename_step1, 'w') as f:
        json.dump(replace_nan_with_none(set_P_info), f, indent=4)
    print(f"Set P information saved to '{output_filename_step1}'")

    return set_P_info


# --- Step 2: Analyze References for Citation Metrics ---
def analyze_reference_citations(paper_id_url):
    """Fetches references for a paper and calculates stats on their citation counts."""
    paper_id = extract_oa_id(paper_id_url)
    if not paper_id:
        print(f"Error: Invalid paper ID URL provided to analyze_reference_citations: {paper_id_url}")
        return {'mean': np.nan, 'max': np.nan, 'min': np.nan, 'count': 0, 'retrieved_refs_for_stats': 0}

    print(f"\nStep 2: Analyzing reference citations for paper ID: {paper_id}")
    # Fetch referenced_works first
    paper_data = make_api_request(f"works/{paper_id}", params={'select': 'id,referenced_works'})


    if not paper_data:
        print(f"Could not retrieve reference list for paper {paper_id}.")
        return {'mean': np.nan, 'max': np.nan, 'min': np.nan, 'count': 0, 'retrieved_refs_for_stats': 0}

    referenced_works_urls = paper_data.get('referenced_works', [])
    if not referenced_works_urls:
        print(f"Paper {paper_id} has no listed references in OpenAlex.")
        return {'mean': 0, 'max': 0, 'min': 0, 'count': 0, 'retrieved_refs_for_stats': 0}

    print(f"Found {len(referenced_works_urls)} reference URLs. Fetching their citation counts...")

    reference_citations = []
    processed_ref_ids = set() # To avoid processing the same reference ID multiple times if listed
    fetch_limit = 1500 # Limit API calls per paper's reference list
    retrieved_ref_data_count = 0


    for i, ref_url in enumerate(referenced_works_urls):
        if i >= fetch_limit:
            print(f"  Warning: Reached reference fetch limit ({fetch_limit}) for paper {paper_id}.")
            break

        ref_id_extracted = extract_oa_id(ref_url) # ref_url is like https://openalex.org/W...
        if not ref_id_extracted or ref_id_extracted in processed_ref_ids:
            # print(f"  Skipping ref: {ref_id_extracted} (invalid or duplicate for this paper's list)")
            continue
        processed_ref_ids.add(ref_id_extracted)

        # Now fetch details for this specific reference ID
        ref_data = make_api_request(f"works/{ref_id_extracted}", params={'select': 'id,cited_by_count'})
        if ref_data and isinstance(ref_data.get('cited_by_count'), int):
            reference_citations.append(ref_data['cited_by_count'])
            retrieved_ref_data_count +=1
        # else:
            # print(f"  Warning: Could not get valid citation count for reference {ref_id_extracted} (URL: {ref_url})")

    print(f"Successfully retrieved citation counts for {len(reference_citations)} distinct references for paper {paper_id} (attempted {len(processed_ref_ids)} unique reference IDs).")

    if not reference_citations: # No valid citation counts found for any reference
        return {'mean': 0, 'max': 0, 'min': 0, 'count': 0, 'retrieved_refs_for_stats': retrieved_ref_data_count}

    mean_citations = np.mean(reference_citations) if reference_citations else 0
    max_citations = np.max(reference_citations) if reference_citations else 0
    min_citations = np.min(reference_citations) if reference_citations else 0

    print(f"Reference Citation Stats for {paper_id}: Mean={mean_citations:.2f}, Max={max_citations}, Min={min_citations}")

    return {
        'mean': mean_citations,
        'max': max_citations,
        'min': min_citations,
        'count': len(referenced_works_urls), # Total references listed
        'retrieved_refs_for_stats': retrieved_ref_data_count # References for which stats were actually obtained
    }

# --- Step 3: Calculate Correlations with Reference Citation Metrics ---
def calculate_citation_correlations(set_P_info_with_ref_stats, filename="correlation_results_citations.csv"):
    """Calculates Pearson correlations between paper citations and ref citation stats."""
    print("\nStep 3: Calculating correlations with reference citation metrics...")
    
    if not set_P_info_with_ref_stats or len(set_P_info_with_ref_stats) < 2:
        print("Insufficient data (less than 2 papers with ref stats) for citation correlations.")
        return {'corr_mean': np.nan, 'p_mean': np.nan,
                'corr_max': np.nan, 'p_max': np.nan,
                'corr_min': np.nan, 'p_min': np.nan,
                'n_corr': 0}

    df = pd.DataFrame(set_P_info_with_ref_stats)
    
    # Ensure necessary columns exist and are numeric
    required_cols = ['citation_count', 'ref_mean_citations', 'ref_max_citations', 'ref_min_citations']
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' missing from DataFrame for citation correlation. Filling with NaN.")
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df_cleaned = df.dropna(subset=required_cols)
    valid_papers_count = len(df_cleaned)

    print(f"\nUsing data from {valid_papers_count} papers for citation correlation.")
    if valid_papers_count > 0:
        print("Data used for citation correlation:")
        print(df_cleaned[['openalex_id', 'citation_count'] + required_cols[1:]].to_string())


    results = {'corr_mean': np.nan, 'p_mean': np.nan,
               'corr_max': np.nan, 'p_max': np.nan,
               'corr_min': np.nan, 'p_min': np.nan,
               'n_corr': valid_papers_count}

    if valid_papers_count < 2:
        print("Cannot calculate citation correlations with less than 2 valid data points after cleaning.")
        df.to_csv(filename, index=False) # Save even if no correlation
        print(f"Full citation stats data saved to '{filename}'")
        return results

    try:
        np_paper_citations = df_cleaned['citation_count'].values
        np_ref_mean = df_cleaned['ref_mean_citations'].values
        np_ref_max = df_cleaned['ref_max_citations'].values
        np_ref_min = df_cleaned['ref_min_citations'].values

        # Check for zero variance which makes Pearson correlation undefined or NaN
        if np.std(np_paper_citations) > 0 and np.std(np_ref_mean) > 0:
            results['corr_mean'], results['p_mean'] = pearsonr(np_paper_citations, np_ref_mean)
        else: print("Warning: Zero variance in paper citations or ref mean citations. Mean correlation undefined.")
        
        if np.std(np_paper_citations) > 0 and np.std(np_ref_max) > 0:
            results['corr_max'], results['p_max'] = pearsonr(np_paper_citations, np_ref_max)
        else: print("Warning: Zero variance in paper citations or ref max citations. Max correlation undefined.")

        if np.std(np_paper_citations) > 0 and np.std(np_ref_min) > 0:
            results['corr_min'], results['p_min'] = pearsonr(np_paper_citations, np_ref_min)
        else: print("Warning: Zero variance in paper citations or ref min citations. Min correlation undefined.")

    except ValueError as e:
        print(f"Error calculating Pearson correlation (citations): {e}. Check input data.")
    
    df.to_csv(filename, index=False) # Save the full data used for this step
    print(f"Full citation stats data saved to '{filename}'")
    print("\nPearson Correlation Results (Citation Metrics):")
    print(f"Paper Citations vs. Ref Mean Citations: Corr={results['corr_mean']:.4f}, P-value={results['p_mean']:.4f} (N={valid_papers_count})")
    print(f"Paper Citations vs. Ref Max Citations:  Corr={results['corr_max']:.4f}, P-value={results['p_max']:.4f} (N={valid_papers_count})")
    print(f"Paper Citations vs. Ref Min Citations:  Corr={results['corr_min']:.4f}, P-value={results['p_min']:.4f} (N={valid_papers_count})")

    return results


# --- Step 4: Topic Analysis ---
def analyze_reference_topics(paper_id_url):
    """Fetches references for a paper and counts distinct topics (using topic IDs) among them."""
    paper_id = extract_oa_id(paper_id_url)
    if not paper_id:
        print(f"Error: Invalid paper ID URL provided to analyze_reference_topics: {paper_id_url}")
        return np.nan # Return NaN to indicate failure clearly

    print(f"\nStep 4a: Analyzing reference topics for paper ID: {paper_id}")
    # Fetch referenced_works first
    paper_data = make_api_request(f"works/{paper_id}", params={'select': 'id,referenced_works'})

    if not paper_data:
        print(f"Could not retrieve reference list for paper {paper_id} (for topics).")
        return np.nan

    referenced_works_urls = paper_data.get('referenced_works', [])
    if not referenced_works_urls:
        print(f"Paper {paper_id} has no listed references in OpenAlex (for topics).")
        return 0 # 0 distinct topics if no references

    print(f"Found {len(referenced_works_urls)} reference URLs. Fetching their topics...")

    all_topic_ids = set() # Store unique topic IDs
    processed_ref_ids = set()
    retrieved_ref_data_count = 0
    fetch_limit = 1500 # Limit API calls per paper's reference list

    for i, ref_url in enumerate(referenced_works_urls):
        if i >= fetch_limit:
            print(f"  Warning: Reached reference fetch limit ({fetch_limit}) for paper {paper_id} (topics).")
            break
        
        ref_id_extracted = extract_oa_id(ref_url)
        if not ref_id_extracted or ref_id_extracted in processed_ref_ids:
            continue
        processed_ref_ids.add(ref_id_extracted)

        # Fetch topics for this specific reference ID
        # OpenAlex API uses 'concepts' more broadly now, 'topics' might be legacy or specific.
        # Let's try 'concepts' as it's more standard in recent OpenAlex.
        ref_data = make_api_request(f"works/{ref_id_extracted}", params={'select': 'id,concepts'})
        if ref_data:
            retrieved_ref_data_count +=1
            concepts = ref_data.get('concepts', []) # 'concepts' is the field for topics/keywords
            if isinstance(concepts, list):
                for concept in concepts:
                    if isinstance(concept, dict) and concept.get('id'):
                        all_topic_ids.add(concept['id']) # Use concept ID for uniqueness
            # else:
                # print(f"  Warning: Unexpected format for concepts field for reference {ref_id_extracted}: {concepts}")
        # else:
             # if ref_data is not None: # API call was made, but no data or concepts field
                # print(f"  Warning: Could not get concept data or 'concepts' field missing for reference {ref_id_extracted}")


    distinct_topic_count = len(all_topic_ids)
    print(f"Found {distinct_topic_count} distinct topics (concepts) across {retrieved_ref_data_count} analyzed distinct references for paper {paper_id} (attempted {len(processed_ref_ids)} unique reference IDs).")
    return distinct_topic_count

def calculate_topic_correlation(set_P_info_with_topic_counts, filename="correlation_results_topics.csv"):
    """Calculates Pearson correlation between paper citations and ref topic diversity."""
    print("\nStep 4b: Calculating correlation with reference topic diversity...")

    if not set_P_info_with_topic_counts or len(set_P_info_with_topic_counts) < 2:
        print("Insufficient data (less than 2 papers with topic counts) for topic correlations.")
        return {'corr_topics': np.nan, 'p_topics': np.nan, 'n_corr': 0}

    df = pd.DataFrame(set_P_info_with_topic_counts)

    required_cols = ['citation_count', 'ref_distinct_topic_count']
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' missing from DataFrame for topic correlation. Filling with NaN.")
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df_cleaned = df.dropna(subset=required_cols)
    valid_papers_count = len(df_cleaned)
    
    print(f"\nUsing data from {valid_papers_count} papers for topic correlation.")
    if valid_papers_count > 0:
        print("Data used for topic correlation:")
        print(df_cleaned[['openalex_id', 'citation_count', 'ref_distinct_topic_count']].to_string())


    results = {'corr_topics': np.nan, 'p_topics': np.nan, 'n_corr': valid_papers_count}

    if valid_papers_count < 2:
        print("Cannot calculate topic correlations with less than 2 valid data points after cleaning.")
        df.to_csv(filename, index=False) # Save even if no correlation
        print(f"Full topic count data saved to '{filename}'")
        return results

    try:
        np_paper_citations = df_cleaned['citation_count'].values
        np_ref_topics = df_cleaned['ref_distinct_topic_count'].values

        if np.std(np_paper_citations) > 0 and np.std(np_ref_topics) > 0:
            results['corr_topics'], results['p_topics'] = pearsonr(np_paper_citations, np_ref_topics)
        else:
            print("Warning: Zero variance in paper citations or ref topic counts. Topic correlation undefined.")
    except ValueError as e:
        print(f"Error calculating Pearson correlation (topics): {e}. Check input data.")

    df.to_csv(filename, index=False) # Save the full data used for this step
    print(f"Full topic count data saved to '{filename}'")
    print("\nPearson Correlation Results (Topic Diversity):")
    print(f"Paper Citations vs. Ref Distinct Topic Count: Corr={results['corr_topics']:.4f}, P-value={results['p_topics']:.4f} (N={valid_papers_count})")

    return results

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting OpenAlex Paper Analysis Script...")

    # Step 1: Get the target set of 6 papers (Set P)
    # This function now uses specific filters for high/medium/low cited.
    set_P_papers_info = perform_step1_paper_retrieval_filtered(SEARCH_QUERY)

    if not set_P_papers_info or len(set_P_papers_info) == 0: # Check if set_P_papers_info is empty
        print("Halting script because Step 1 did not yield any papers.")
        exit()

    # Prepare list of dictionaries for DataFrame conversion later
    papers_for_analysis = []
    for pid, info in set_P_papers_info.items():
        papers_for_analysis.append({
            'openalex_id': pid,
            'openalex_id_url': info['openalex_id_url'],
            'title': info['title'],
            'citation_count': info['citation_count'],
            'publication_year': info['publication_year'],
            'category_step1': info['category_step1']
        })

    # Steps 2 & 3: Analyze reference citations and calculate correlations
    print("\n" + "="*20 + " Processing for Citation Statistics (Steps 2 & 3) " + "="*20)
    papers_with_ref_citation_stats = []
    for paper_info in papers_for_analysis:
        temp_info = paper_info.copy()
        ref_stats = analyze_reference_citations(paper_info['openalex_id_url']) # Pass the full URL
        if ref_stats: # ref_stats will not be None, it returns a dict with NaN or values
            temp_info['ref_mean_citations'] = ref_stats['mean']
            temp_info['ref_max_citations'] = ref_stats['max']
            temp_info['ref_min_citations'] = ref_stats['min']
            temp_info['ref_total_listed'] = ref_stats['count']
            temp_info['ref_retrieved_for_stats'] = ref_stats['retrieved_refs_for_stats']
        else: # Should not happen with current analyze_reference_citations which returns dict
            temp_info.update({'ref_mean_citations': np.nan, 'ref_max_citations': np.nan,
                              'ref_min_citations': np.nan, 'ref_total_listed': 0,
                              'ref_retrieved_for_stats': 0})
        papers_with_ref_citation_stats.append(temp_info)
    
    # Perform Step 3 correlations
    citation_correlation_results = calculate_citation_correlations(papers_with_ref_citation_stats)


    # Step 4: Analyze reference topics and calculate correlations
    print("\n" + "="*20 + " Processing for Topic Statistics (Step 4) " + "="*20)
    papers_with_ref_topic_counts = []
    for paper_info in papers_for_analysis: # Use the same initial list
        temp_info = paper_info.copy()
        topic_count = analyze_reference_topics(paper_info['openalex_id_url']) # Pass the full URL
        temp_info['ref_distinct_topic_count'] = topic_count # topic_count can be NaN or int
        papers_with_ref_topic_counts.append(temp_info)

    # Perform Step 4b correlations
    topic_correlation_results = calculate_topic_correlation(papers_with_ref_topic_counts)

    print("\n" + "="*20 + " Script Finished " + "="*20)
    print("Summary of Citation Correlation:")
    print(f"  N={citation_correlation_results.get('n_corr',0)}")
    print(f"  Mean: Corr={citation_correlation_results.get('corr_mean', np.nan):.4f}, P={citation_correlation_results.get('p_mean', np.nan):.4f}")
    print(f"  Max:  Corr={citation_correlation_results.get('corr_max', np.nan):.4f}, P={citation_correlation_results.get('p_max', np.nan):.4f}")
    print(f"  Min:  Corr={citation_correlation_results.get('corr_min', np.nan):.4f}, P={citation_correlation_results.get('p_min', np.nan):.4f}")
    print("\nSummary of Topic Correlation:")
    print(f"  N={topic_correlation_results.get('n_corr',0)}")
    print(f"  Topics: Corr={topic_correlation_results.get('corr_topics', np.nan):.4f}, P={topic_correlation_results.get('p_topics', np.nan):.4f}")

