"""Quick test for sport_pnl context."""
import sys
sys.path.insert(0, "src")
from web.app import _games_context
ctx = _games_context()
sp = ctx.get("sport_pnl", [])
print("n_sport_count:", ctx.get("n_sport_count", 0))
for s in sp:
    print(f"  {s['league']}: {s['wins']}-{s['losses']} profit=${s['profit']}")
print("All good!")
