# B2B Procurement Copilot

A product-oriented AI/ML prototype for B2B procurement and product discovery.

## What it does

Users describe a business requirement in natural language, for example:

> Need 25 laptops for our sales team, 16GB RAM, under ₹60,000 for Excel and video calls.

The application extracts useful constraints, searches a structured product catalogue, and produces an explainable ranked shortlist.

## Recommendation architecture

1. **Requirement extraction** — lightweight NLP/rule-based parsing identifies category, budget, quantity, RAM, storage and use case.
2. **TF-IDF representation** — converts the requirement and product information into numerical vectors.
3. **Cosine similarity** — measures semantic/textual relevance.
4. **Business constraints** — budget, category, RAM, storage and use-case fit are incorporated into the score.
5. **Explainable ranking** — the UI shows why a product matched.

This hybrid approach is intentionally simple and explainable: semantic matching handles natural language while business rules enforce procurement constraints.

## Tech stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- TF-IDF
- Cosine similarity
- Regex/NLP-based requirement extraction

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```text
IndiaMART-AI-Product-Assistant/
├── app.py
├── products.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Product thinking

The prototype is designed around a B2B buyer journey rather than only a model demo:

**Requirement → Understand intent → Apply constraints → Rank products → Explain recommendation → Compare options**

## Future improvements

- Connect to a real product catalogue/API
- Learn ranking weights from historical buyer interactions
- Add supplier quality, delivery time and inventory availability
- Add user feedback to improve ranking
- Add multilingual Indian-language query support
- Deploy as a production API + web application
