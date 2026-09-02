#!/usr/bin/env python
from __future__ import annotations

from serve_cross_encoder import run_server


def main(argv: list[str] | None = None) -> None:
    run_server(
        argv,
        role="activity",
        default_model="models/modernbert-activity-aligner",
        default_port=8083,
    )


if __name__ == "__main__":
    main()
