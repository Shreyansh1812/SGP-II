## YYYY-MM-DD - [Optimize Backtester Loop]
**Learning:** Pandas `.loc` assignment and column extraction (`row.Close` or `data.iloc[i]`) inside a loop over a DataFrame (`itertuples()`) is a massive performance bottleneck, resulting in O(N^2) or extreme overhead.
**Action:** Always pre-extract Series values to numpy arrays (`.values`) or lists before iterating, loop using simple indexing `for i in range(len(data))`, and reconstruct the `pd.Series` after the loop. This can yield ~10x-15x performance improvements.
