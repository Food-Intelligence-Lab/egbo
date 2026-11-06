# Real-world dairy deployment

`food_bo.py` is the BoTorch optimizer used by the closed-loop campaigns, and `app.py` is the Streamlit interface. The released `data/` directory contains recipe-level ingredient quantities together with sensory, utility, and human-comparison traces; the proprietary partner catalog and serialized project state are excluded. Confirm permission to redistribute the recipe-level data and experimental photographs before publishing the repository.

Install the exact release dependencies from this directory:

```bash
python3 -m pip install -r requirements.txt
```

Regenerate the data-driven deployment figures:

```bash
python3 plot_results.py --out-dir figures
```

Run the interface on the included example ingredient catalog:

```bash
python3 -m streamlit run app.py
```

Run the test suite:

```bash
python3 -m pytest -q
```

The exact photographs used as Figures 3 and 18-27 are in `paper_assets/`. They are experimental records, not generated outputs.

The included example catalog is sufficient to exercise the interface and
optimizer. It is not the proprietary catalog used during the physical
campaigns, so the original closed-loop experiments cannot be rerun without
access to the corresponding laboratory and ingredients. The released data do
support exact regeneration of the paper's data-driven deployment plots.
