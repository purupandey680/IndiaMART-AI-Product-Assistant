import re
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="B2B Procurement Copilot",
    page_icon="🤖",
    layout="wide",
)

DATA_FILE = Path(__file__).parent / "products.csv"


@st.cache_data
def load_products():
    df = pd.read_csv(DATA_FILE)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    return df


products = load_products()


def extract_requirements(query: str) -> dict:
    q = query.lower()

    req = {
        "budget": None,
        "quantity": None,
        "ram_gb": None,
        "storage_gb": None,
        "screen_in": None,
        "category": None,
        "use_case": [],
    }

    # Budget: ₹60,000 / 60000 / under 60k / below 60,000
    budget_patterns = [
        r"(?:under|below|less than|upto|up to|max(?:imum)?|within)\s*(?:₹|rs\.?\s*)?([\d,]+)\s*k\b",
        r"(?:under|below|less than|upto|up to|max(?:imum)?|within)\s*(?:₹|rs\.?\s*)?([\d,]+(?:,\d{3})*)",
        r"(?:₹|rs\.?\s*)([\d,]+(?:,\d{3})*)",
    ]
    for pattern in budget_patterns:
        m = re.search(pattern, q)
        if m:
            raw = m.group(1).replace(",", "")
            value = float(raw)
            if "k" in m.group(0):
                value *= 1000
            req["budget"] = value
            break

    # Quantity: 25 laptops / quantity 25 / 25 units
    quantity_patterns = [
        r"\b(\d+)\s*(?:units?|pieces?|laptops?|desktops?|monitors?|chairs?|printers?|routers?|headsets?|webcams?)\b",
        r"(?:quantity|qty)\s*[:=]?\s*(\d+)\b",
    ]
    for pattern in quantity_patterns:
        m = re.search(pattern, q)
        if m:
            req["quantity"] = int(m.group(1))
            break

    # RAM
    m = re.search(r"\b(\d+)\s*gb\s*(?:ram|memory)\b|\b(\d+)\s*gb\b", q)
    if m:
        req["ram_gb"] = int(next(x for x in m.groups() if x))

    # Storage
    m = re.search(r"\b(\d+)\s*(?:gb|tb)\s*(?:ssd|storage|hdd)?\b", q)
    if m and not re.search(r"\b\d+\s*gb\s*(?:ram|memory)\b", m.group(0)):
        value = int(m.group(1))
        req["storage_gb"] = value * 1024 if "tb" in m.group(0) else value

    # Screen size
    m = re.search(r"\b(\d{2}(?:\.\d)?)\s*(?:inch|inches|in)\b", q)
    if m:
        req["screen_in"] = float(m.group(1))

    categories = {
        "laptop": ["laptop", "notebook"],
        "desktop": ["desktop", "pc", "workstation"],
        "monitor": ["monitor", "display", "screen"],
        "webcam": ["webcam", "camera", "video meeting"],
        "headset": ["headset", "headphones"],
        "printer": ["printer", "printing"],
        "router": ["router", "network router", "wifi router"],
        "projector": ["projector", "projection"],
        "office chair": ["office chair", "ergonomic chair", "chair"],
        "tablet": ["tablet", "ipad"],
    }
    for category, terms in categories.items():
        if any(term in q for term in terms):
            req["category"] = category
            break

    use_cases = {
        "office productivity": ["office", "productivity", "excel", "work"],
        "video conferencing": ["video call", "video meeting", "conference", "zoom", "teams", "meetings"],
        "sales": ["sales", "field sales", "sales team"],
        "design": ["design", "creative", "photoshop", "video editing", "editing"],
        "customer support": ["customer support", "call center", "contact center"],
        "remote work": ["remote", "work from home", "wfh"],
        "gaming": ["gaming", "gaming performance"],
    }
    for label, terms in use_cases.items():
        if any(term in q for term in terms):
            req["use_case"].append(label)

    return req


def build_search_text(row):
    return " ".join([
        str(row["name"]),
        str(row["category"]),
        str(row["brand"]),
        str(row["description"]),
        str(row["use_cases"]),
        str(row["specifications"]),
    ])


@st.cache_resource
def build_vectorizer(corpus):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


products["search_text"] = products.apply(build_search_text, axis=1)
vectorizer, product_matrix = build_vectorizer(tuple(products["search_text"]))


def recommend(query, top_n=5):
    req = extract_requirements(query)
    query_vector = vectorizer.transform([query])
    semantic_scores = cosine_similarity(query_vector, product_matrix).flatten()

    ranked = products.copy()
    ranked["semantic_score"] = semantic_scores

    # Hard/soft business constraints.
    ranked["constraint_score"] = 0.0

    if req["budget"] is not None:
        ranked["budget_fit"] = (ranked["price"] <= req["budget"]).astype(float)
        # Small bonus for products comfortably inside the budget.
        ranked["budget_score"] = (
            (ranked["budget_fit"] * 0.7)
            + ((ranked["price"] <= req["budget"] * 0.85).astype(float) * 0.3)
        )
        ranked["constraint_score"] += ranked["budget_score"] * 0.30
    else:
        ranked["budget_score"] = 0.0

    if req["category"]:
        category_match = ranked["category"].str.lower().eq(req["category"].lower())
        ranked["constraint_score"] += category_match.astype(float) * 0.30

    if req["ram_gb"] is not None:
        ranked["ram_fit"] = (
            ranked["ram_gb"].fillna(0) >= req["ram_gb"]
        ).astype(float)
        ranked["constraint_score"] += ranked["ram_fit"] * 0.15
    else:
        ranked["ram_fit"] = 0.0

    if req["storage_gb"] is not None:
        ranked["storage_fit"] = (
            ranked["storage_gb"].fillna(0) >= req["storage_gb"]
        ).astype(float)
        ranked["constraint_score"] += ranked["storage_fit"] * 0.10
    else:
        ranked["storage_fit"] = 0.0

    if req["use_case"]:
        def use_case_match(value):
            text = str(value).lower()
            return float(any(x in text for x in req["use_case"]))

        ranked["use_case_fit"] = ranked["use_cases"].apply(use_case_match)
        ranked["constraint_score"] += ranked["use_case_fit"] * 0.15
    else:
        ranked["use_case_fit"] = 0.0

    # 60% semantic relevance + 40% business constraints.
    ranked["match_score"] = (
        ranked["semantic_score"] * 0.60
        + ranked["constraint_score"] * 0.40
    )

    # If a requested budget exists, strongly prefer products inside it.
    if req["budget"] is not None:
        ranked.loc[ranked["price"] > req["budget"], "match_score"] *= 0.55

    return ranked.sort_values(
        ["match_score", "rating"],
        ascending=[False, False],
    ).head(top_n), req


def explain_match(row, req):
    reasons = []

    if req["category"] and str(row["category"]).lower() == req["category"].lower():
        reasons.append(f"matches the {req['category']} category")

    if req["budget"] is not None:
        if row["price"] <= req["budget"]:
            reasons.append("fits your budget")
        else:
            reasons.append("exceeds the stated budget")

    if req["ram_gb"] is not None and row["ram_gb"] >= req["ram_gb"]:
        reasons.append(f"meets the {req['ram_gb']} GB RAM requirement")

    if req["storage_gb"] is not None and row["storage_gb"] >= req["storage_gb"]:
        reasons.append(f"meets the {req['storage_gb']} GB storage requirement")

    for use_case in req["use_case"]:
        if use_case.lower() in str(row["use_cases"]).lower():
            reasons.append(f"suits {use_case}")

    if not reasons:
        reasons.append("has strong semantic similarity to the requirement")

    return reasons


st.markdown(
    """
    <style>
    .main-title {font-size: 2.3rem; font-weight: 750; margin-bottom: 0.2rem;}
    .subtitle {color: #6b7280; font-size: 1.05rem; margin-bottom: 1.5rem;}
    .metric-card {padding: 1rem; border-radius: 12px; border: 1px solid #e5e7eb;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🤖 B2B Procurement Copilot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Turn a business requirement into ranked, explainable product recommendations.</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Products in catalogue", len(products))
with col2:
    st.metric("Categories", products["category"].nunique())
with col3:
    st.metric("Recommendation model", "Hybrid NLP + Rules")

st.divider()

examples = [
    "Need 25 laptops for our sales team, 16GB RAM, under ₹60,000 for Excel and video calls",
    "Need a webcam for corporate video meetings",
    "Need office chairs for a customer support team with ergonomic support",
]

st.subheader("🔎 Describe your requirement")

example = st.selectbox("Try an example", ["Custom requirement"] + examples)
default_text = "" if example == "Custom requirement" else example

query = st.text_area(
    "Business requirement",
    value=default_text,
    height=100,
    placeholder="Example: Need 20 business laptops with 16GB RAM under ₹60,000 for sales employees...",
)

top_n = st.slider("Number of recommendations", 3, 8, 5)

if st.button("🚀 Find Best Products", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a product requirement.")
    else:
        results, req = recommend(query, top_n)

        st.session_state["results"] = results
        st.session_state["requirements"] = req
        st.session_state["query"] = query


if "results" in st.session_state:
    results = st.session_state["results"]
    req = st.session_state["requirements"]

    st.divider()
    st.subheader("🧠 Requirement understood")

    req_cols = st.columns(5)
    items = [
        ("Category", req["category"] or "Any"),
        ("Budget", f"₹{req['budget']:,.0f}" if req["budget"] else "Not specified"),
        ("Quantity", str(req["quantity"]) if req["quantity"] else "Not specified"),
        ("RAM", f"{req['ram_gb']} GB" if req["ram_gb"] else "Not specified"),
        ("Use case", ", ".join(req["use_case"]) if req["use_case"] else "General"),
    ]

    for col, (label, value) in zip(req_cols, items):
        with col:
            st.metric(label, value)

    st.subheader("🏆 Recommended products")

    for rank, (_, product) in enumerate(results.iterrows(), start=1):
        score = min(100, round(product["match_score"] * 100))
        with st.container(border=True):
            left, middle, right = st.columns([5, 2, 1])

            with left:
                st.markdown(f"### #{rank} — {product['name']}")
                st.write(f"**{product['brand']} · {product['category']}**")
                st.write(product["description"])
                reasons = explain_match(product, req)
                st.write("**Why it matches:** " + "; ".join(reasons) + ".")

            with middle:
                st.metric("Match score", f"{score}%")
                st.write(f"**Price:** ₹{product['price']:,.0f}")
                st.write(f"**Rating:** ⭐ {product['rating']}/5")

            with right:
                st.write("**Key specs**")
                st.write(product["specifications"])

    st.divider()
    st.subheader("📊 Compare top recommendations")

    compare = results[
        ["name", "category", "price", "rating", "ram_gb", "storage_gb", "match_score"]
    ].copy()
    compare["match_score"] = (compare["match_score"] * 100).round(0).astype(int).astype(str) + "%"
    compare["price"] = compare["price"].map(lambda x: f"₹{x:,.0f}")
    compare["ram_gb"] = compare["ram_gb"].map(lambda x: f"{int(x)} GB")
    compare["storage_gb"] = compare["storage_gb"].map(lambda x: f"{int(x)} GB")
    compare = compare.rename(columns={
        "name": "Product",
        "category": "Category",
        "price": "Price",
        "rating": "Rating",
        "ram_gb": "RAM",
        "storage_gb": "Storage",
        "match_score": "Match",
    })
    st.dataframe(compare, use_container_width=True, hide_index=True)

    st.caption(
        "Model note: recommendations combine TF-IDF/cosine semantic similarity "
        "with business constraints such as category, budget, RAM, storage and use case."
    )

with st.expander("ℹ️ How the recommendation engine works"):
    st.markdown(
        """
        **1. Requirement extraction:** simple NLP/rule-based parsing identifies useful
        business constraints such as category, budget, quantity, RAM, storage and use case.

        **2. Text representation:** TF-IDF converts the requirement and product information
        into numerical vectors.

        **3. Semantic matching:** cosine similarity measures how closely the requirement
        matches each product.

        **4. Business scoring:** category, budget and specification fit are added as
        constraints.

        **5. Ranking:** the hybrid score produces an explainable shortlist instead of
        relying only on keyword matching.
        """
    )
