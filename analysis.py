import novelpy
import tqdm
import matplotlib.pyplot as plt

openalex_id = "W2464236921"
starting_year = 2006
ending_year = 2016

ref_cooc = novelpy.utils.cooc_utils.create_cooc(
                 collection_name = openalex_id,
                 year_var="year",
                 var = "c04_referencelist",
                 sub_var = "item",
                 time_window = range(starting_year, ending_year),
                 weighted_network = True, self_loop = True)

ref_cooc.main()

for focal_year in tqdm.tqdm(range(starting_year,ending_year), desc = "Computing indicator for window of time"):
    Uzzi = novelpy.indicators.Uzzi2013(collection_name = openalex_id,
                                            id_variable = 'PMID',
                                            year_variable = 'year',
                                            variable = "c04_referencelist",
                                            sub_variable = "item",
                                            focal_year = focal_year,
                                            density = True)
    Uzzi.get_indicator()
    if focal_year != starting_year:
        Foster = novelpy.indicators.Foster2015(collection_name = openalex_id,
                                            id_variable = 'PMID',
                                            year_variable = 'year',
                                            variable = "c04_referencelist",
                                            sub_variable = "item",
                                            focal_year = focal_year,
                                            starting_year = starting_year,
                                            community_algorithm = "Louvain",
                                            density = True)
        Foster.get_indicator()
    Lee = novelpy.indicators.Lee2015(collection_name = openalex_id,
                                           id_variable = 'PMID',
                                           year_variable = 'year',
                                           variable = "c04_referencelist",
                                           sub_variable = "item",
                                           focal_year = focal_year,
                                           density = True)
    Lee.get_indicator()
    

ref_cooc = novelpy.utils.cooc_utils.create_cooc(
                 collection_name = openalex_id,
                 year_var="year",
                 var = "c04_referencelist",
                 sub_var = "item",
                 time_window = range(starting_year, ending_year),
                 weighted_network = False, self_loop = False)

ref_cooc.main()

for focal_year in tqdm.tqdm(range(starting_year,ending_year-6), desc = "Computing indicator for window of time"):
    Wang = novelpy.indicators.Wang2017(collection_name = openalex_id,
                                           id_variable = 'PMID',
                                           year_variable = 'year',
                                           variable = "c04_referencelist",
                                           sub_variable = "item",
                                           focal_year = focal_year+3,
                                           starting_year = starting_year,
                                           time_window_cooc = 3,
                                           n_reutilisation = 1,
                                           density = True)
    Wang.get_indicator()
