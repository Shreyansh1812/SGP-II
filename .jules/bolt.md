## 2024-07-29 - [Optimizing DataFrame row iteration]
**Learning:** Iterating over a DataFrame using `itertuples()` and updating external Series with `.loc[]` inside the loop is very slow, causing O(N^2) or high constant factor performance hits.
**Action:** When needing to build up series in a loop where vectorization is difficult (like state-dependent backtesting), build standard Python lists inside the loop and convert to Series/DataFrame once at the end. Also extract values as NumPy arrays (`.values`) before the loop rather than looking up by `.loc[date]` or `.iloc[i]` on every iteration.
