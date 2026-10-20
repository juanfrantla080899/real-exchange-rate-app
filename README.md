Real Exchange Rate Explorer
A Streamlit app for International Finance that plots the real exchange rate (RER) between any two IMF countries (or the Euro area) and benchmarks it against its sample average to judge whether a currency is overvalued or undervalued.
What it does
Choose Country 1 (base currency) and Country 2 (quote currency) from IMF countries and the Euro area (`G163`). Quick presets and a swap button are in the sidebar. Any other IMF code can be typed via "Other".
Downloads monthly CPI and monthly average exchange rates from the IMF SDMX 3.0 API, from January 1976 where available.
Computes, for every month:
Symbol	Definition
E1, E2	IMF rates: local currency units per USD
S(t)	E2 / E1 = units of Country 2 currency per 1 unit of Country 1 currency
RER(t)	S(t) × P1(t) / P2(t)
RER*	Average of RER(t) over valid months in the selected sample
Real Index(t)	100 × RER(t) / RER*
S_PPP(t)	RER* × P2(t) / P1(t), the nominal rate consistent with RER*
Interactive charts (Plotly): Real, Nominal, or Both. "Both" shows the two as separate graphs, each with its own axis. The real chart shades periods above 100 in red (overvalued) and below 100 in green (undervalued). Nominal S is shown in actual currency units, never normalized. Options: log scale and a PPP-implied rate line on the nominal chart.
Sample window slider: RER*, the index, the readout and the CSV are recalculated whenever the dates change.
Valuation readout: e.g. "In Aug 2026, the MXN is 12.4% overvalued against the USD relative to its Jan 1990 to Aug 2026 average." Index above 100 means Country 1's currency is real appreciated (overvalued) relative to the benchmark.
CSV download of every month in the sample: raw IMF inputs, S, RER, RER*, Real Index, % deviation, S_PPP, a valid-month flag and a possible-break flag.
Data quality checks: coverage table per series, a warning for one-month moves in S above 50% (possible redenomination or legacy currency change), and a note when a euro member's sample starts before it adopted the euro.
Installation (Python 3.12)
```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
The app opens at http://localhost:8501.
Deploying (optional)
Push `app.py`, `requirements.txt` and `README.md` to a GitHub repository, then create an app at share.streamlit.io pointing to `app.py`. Choose Python 3.12 under Advanced settings.
Data source details
Base URL: `https://api.imf.org/external/sdmx/3.0`
Request pattern: `{base}/data/dataflow/IMF.STA/{DATAFLOW}/+/{KEY}`
CPI: dataflow `CPI`, key `{ISO3}.CPI._T.IX.M`
Exchange rate: dataflow `ER`, key `{ISO3}.XDC_USD.PA_RT.M`
Euro area code: `G163`. The US rate against the dollar is set to 1 and not requested.
The app requests SDMX CSV first and falls back to SDMX JSON and XML, so it tolerates whichever format the API returns. Responses are cached for 24 hours; use Clear cached IMF data in the sidebar to force a refresh. Failed requests are retried with backoff.
If the IMF changes identifiers, you can override them without editing code through environment variables: `IMF_SDMX_BASE`, `IMF_AGENCY`, `IMF_CPI_DATAFLOW`, `IMF_ER_DATAFLOW`. If the API ever requires a subscription key, set `IMF_API_KEY` (environment variable or `.streamlit/secrets.toml`); it is sent as `Ocp-Apim-Subscription-Key`.
Handling gaps and limited coverage
A month is valid only if both CPIs and both exchange rates exist and are positive. Only valid months enter RER*. Nothing is interpolated; invalid months appear as gaps in the charts and are flagged in the CSV.
The slider covers the span from the first to the last valid month for the pair.
If a series has no IMF data at all, the app says which one and shows the requested URLs.
Euro members' exchange rates before adoption may be in legacy currencies (DEM, FRF, ...). The Euro area aggregate has limited pre-1999 coverage.
Countries that redenominated (e.g. Mexico 1993, Turkey 2005) or had large devaluations can show jumps in S(t); these months are flagged for review.
Interpretation notes
CPI base years cancel out in the Real Index, so rebasing does not affect it.
RER* is a relative PPP benchmark tied to the chosen window. Different windows give different conclusions, and structural forces (e.g. Balassa-Samuelson effects) can shift the equilibrium RER over time.
Files
`app.py`: the Streamlit application (data access, calculations, charts, UI)
`requirements.txt`: dependencies
`README.md`: this file
