import os.path
import pandas as pd
import pandas.errors
from scipy import stats

path = "RESULTS_DEV"

aggr_cols = ['OM_P@R95', 'OM_P@R95_threshold', 'OM_F1', 'OM_PRE', 'OM_REC/SENS', 'OM_SPEC', 'OM_corrected_frac']
results = []
for fn in os.listdir(path):
    if "pre_at_rec" in fn:
        continue

    _, dataset, _, approach, _ = fn.split("_")
    full_path = os.path.join(path, fn)
    try:
        df = pd.read_csv(full_path)[['overseer_mistakes', 'ecg_count', *aggr_cols]]
    except pandas.errors.ParserError:
        continue

    for record in df.to_dict('records'):
        results.append({'dataset': dataset, 'approach': approach} | record)

res = pd.DataFrame.from_records(results)
res.to_csv(f'merged_{path}.csv')
print(res)


def confidence(x):
    scale = stats.sem(x)
    if scale == 0:
        return f"{x.mean():.2f} ({len(x)}x)"
    else:
        interval = stats.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=scale)
        return f"({interval[0]:.2f}, {interval[1]:.2f})"


res = res.loc[(0 < res['overseer_mistakes']) & (res['overseer_mistakes'] < res['ecg_count'])]
res.to_csv(f'merged_filtered_{path}.csv')
mean_and_confidence = ['median', confidence]
# mean_and_confidence = [confidence]

grp = res.groupby(['dataset', 'approach', 'overseer_mistakes']).agg(
    {k: mean_and_confidence for k in aggr_cols})

print(grp)
grp.to_csv(f'merged_{path}_grp.csv')

grp = res.groupby('approach').agg(
    {k: mean_and_confidence for k in aggr_cols})

print(grp)
grp.to_csv(f'merged_{path}_grp_by_approach.csv')
