# Project website

A static site that showcases the three Data Science projects with live, in-browser demos:

- **Iris Flower Classification**: KNN (k = 5) on standardised features
- **Email Spam Detection**: TF-IDF + Multinomial Naive Bayes
- **Sales Prediction**: degree-2 polynomial regression

`build_models.py` retrains each model with the notebooks' settings and exports the parameters to
`models/*.json`; `app.js` runs the predictions in JavaScript, so the site needs no backend.

```bash
python website/build_models.py      # from the repo root, regenerates models/*.json
cd website && python -m http.server # then open http://localhost:8000
```
