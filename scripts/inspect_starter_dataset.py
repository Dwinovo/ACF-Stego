from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.tools import starter_dataset


def main() -> None:
    ds = starter_dataset.load_starters_train_split()
    print(f"rows={len(ds)}")
    print(f"columns={list(getattr(ds, 'column_names', []))}")
    print(f"features={getattr(ds, 'features', {})}")

    rows = starter_dataset.build_absolute_id_starters(ds)
    print("preview(abs_id, prompt):")
    for abs_id, prompt in rows[:5]:
        print(f"- {abs_id}: {prompt}")


if __name__ == "__main__":
    main()
