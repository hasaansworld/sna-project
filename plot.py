import matplotlib.pyplot as plt
import novelpy

starting_year = 2006
ending_year = 2016
trend = novelpy.utils.novelty_trend(year_range = range(starting_year+1,ending_year-1,1),
              variables = ["c04_referencelist"],
              id_variable = "PMID",
              indicators = ["uzzi", "foster", "lee"],
              time_window_cooc = [3],
              n_reutilisation = [1]
            )

fig = trend.get_plot_trend()

# Save the figure to a file
plt.savefig('trend_plot.png', dpi=300, bbox_inches='tight')

# If you also want to display the plot, add:
plt.show()


trend = novelpy.utils.novelty_trend(year_range = range(starting_year+3,ending_year-4,1),
              variables = ["c04_referencelist"],
              id_variable = "PMID",
              indicators = ["wang"],
              time_window_cooc = [3],
              n_reutilisation = [1],
              keep_item_percentile = [50],
            )

fig = trend.get_plot_trend()

# Save the figure to a file
plt.savefig('trend_plot2.png', dpi=300, bbox_inches='tight')

# If you also want to display the plot, add:
plt.show()