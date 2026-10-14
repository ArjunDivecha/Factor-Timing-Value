# Patch a scratch copy of the Value repo: --alphas eligible|all  --fill eligible|all
import sys, re
d, alphas, fill = sys.argv[1], sys.argv[2], sys.argv[3]
ELIG = '''
import pandas as _pd
_c = _pd.read_excel('Step Factor Categories.xlsx'); _m = dict(zip(_c['Factor Name'], _c['Max']))
def _eligible(cols): return [c for c in cols if _m.get(c, 1.0) > 0]
'''
if alphas == "eligible":
    p = f"{d}/Step Six Create Country alphas from Factor alphas.py"; s = open(p).read()
    a = '    factor_names = [f for f in factor_file_factors if f in exposure_factors]\n'
    assert s.count(a) == 1; s = s.replace(a, a + "    factor_names = _eligible(factor_names)\n    print('VARIANT alphas eligible-only:', len(factor_names))\n")
    s = s.replace("import pandas as pd", "import pandas as pd" + ELIG, 1); open(p, "w").write(s)
if fill == "eligible":
    p = f"{d}/Step Four Create Monthly Top20 Returns FAST.py"; s = open(p).read()
    n = s.count("row.fillna(row.mean())"); assert n == 2, n
    s = s.replace("row.fillna(row.mean())", "row.fillna(row[_eligible(row.index)].mean())")
    s = s.replace("import pandas as pd", "import pandas as pd" + ELIG, 1); open(p, "w").write(s)
print("patched", d, alphas, fill)
