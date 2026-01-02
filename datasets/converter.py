from pathlib import Path
from dv import AedatFile
import csv

BASE = Path(__file__).resolve().parent
in_file  = BASE / "ottawa_8.aedat4"
out_file = BASE / "ottawa_8.csv"

with AedatFile(str(in_file)) as f, open(out_file, "w", newline="") as g:
    writer = csv.writer(g)
    writer.writerow(["t", "x", "y", "p"])

    events = f["events"]

    for e in events:
        writer.writerow([
            int(e.timestamp),
            int(e.x),
            int(e.y),
            int(e.polarity),
        ])


print("Done:", out_file)