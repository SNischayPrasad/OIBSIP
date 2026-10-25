# Project website

A static site that showcases the three Data Science projects with live, in-browser demos. The home page
(`index.html`) links to a separate page for each project:

- [`iris.html`](iris.html) (live at `/iris`): Iris Flower Classification, KNN (k = 5) on standardised features
- [`spam.html`](spam.html) (live at `/spam`): Email Spam Detection, TF-IDF + Multinomial Naive Bayes
- [`sales.html`](sales.html) (live at `/sales`): Sales Prediction, degree-2 polynomial regression

`vercel.json` turns on clean URLs, and old `/#iris`-style links redirect to the matching page.

`build_models.py` retrains each model with the notebooks' settings and exports the parameters to
`models/*.json`; `app.js` runs the predictions in JavaScript, so the site needs no backend.

```bash
python website/build_models.py      # from the repo root, regenerates models/*.json
cd website && python -m http.server # then open http://localhost:8000
```
