from openai import OpenAI
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Sample B2B product catalogue
products = pd.DataFrame([
    {
        "name": "Dell Latitude 3540",
        "category": "Laptop",
        "price": 57999,
        "description": "Business laptop with 16GB RAM, Intel Core i5 processor and 512GB SSD for corporate office employees."
    },
    {
        "name": "HP ProBook 440 G10",
        "category": "Laptop",
        "price": 59499,
        "description": "Professional business laptop with 16GB RAM, Intel Core i5 processor and 512GB SSD for office productivity."
    },
    {
        "name": "Lenovo ThinkPad E14",
        "category": "Laptop",
        "price": 55999,
        "description": "Reliable business laptop with 16GB RAM, Intel Core i5 processor and 512GB SSD for professional users."
    },
    {
        "name": "ASUS ExpertBook B1",
        "category": "Laptop",
        "price": 48999,
        "description": "Affordable business laptop with 8GB RAM, Intel Core i5 processor and 512GB SSD for small businesses."
    },

    {
        "name": "Canon Laser Printer LBP",
        "category": "Printer",
        "price": 18999,
        "description": "Monochrome laser printer with high-speed printing for small and medium offices."
    },
    {
        "name": "Epson EcoTank L3250",
        "category": "Printer",
        "price": 22999,
        "description": "Colour ink tank printer with low running costs for offices, schools and businesses."
    },
    {
        "name": "HP LaserJet Pro",
        "category": "Printer",
        "price": 24999,
        "description": "Fast wireless laser printer designed for business documents and office environments."
    },

    {
        "name": "Logitech Business Webcam",
        "category": "Webcam",
        "price": 6999,
        "description": "Full HD business webcam for video meetings, remote work, conferences and corporate communication."
    },
    {
        "name": "HP 320 FHD Webcam",
        "category": "Webcam",
        "price": 4999,
        "description": "Full HD webcam with built-in microphone for online meetings and professional video calls."
    },
    {
        "name": "Lenovo 300 FHD Webcam",
        "category": "Webcam",
        "price": 4499,
        "description": "Affordable Full HD webcam for remote employees, video conferencing and corporate communication."
    },

    {
        "name": "Dell P2422H Monitor",
        "category": "Monitor",
        "price": 16999,
        "description": "24-inch Full HD business monitor designed for office productivity and professional work."
    },
    {
        "name": "LG 24MP60G Monitor",
        "category": "Monitor",
        "price": 13999,
        "description": "24-inch Full HD monitor suitable for offices, employees and everyday business productivity."
    },
    {
        "name": "Samsung Business Monitor",
        "category": "Monitor",
        "price": 17999,
        "description": "24-inch Full HD monitor designed for corporate workstations and professional office environments."
    },

    {
        "name": "TP-Link 24-Port Gigabit Switch",
        "category": "Networking",
        "price": 8499,
        "description": "24-port Gigabit network switch for connecting computers, printers and devices in offices."
    },
    {
        "name": "Cisco Business 350 Switch",
        "category": "Networking",
        "price": 32999,
        "description": "Managed Gigabit switch designed for enterprise and medium-sized business networks."
    },
    {
        "name": "TP-Link WiFi 6 Router",
        "category": "Networking",
        "price": 7999,
        "description": "High-speed WiFi 6 router suitable for offices, small businesses and reliable wireless connectivity."
    },

    {
        "name": "Epson CO-W01 Projector",
        "category": "Projector",
        "price": 34999,
        "description": "Business projector for conference rooms, presentations, training sessions and corporate meetings."
    },
    {
        "name": "BenQ Business Projector",
        "category": "Projector",
        "price": 42999,
        "description": "Bright business projector designed for meeting rooms, presentations and professional corporate use."
    }
])


# Page configuration
st.set_page_config(
    page_title="B2B Product Recommendation Assistant",
    page_icon="🤖"
)

st.title("🤖 B2B Product Recommendation Assistant")

st.write(
    "Describe what you need in normal language and "
    "the system will recommend relevant products."
)


# User query
query = st.text_input(
    "What are you looking for?",
    placeholder="Example: I need office laptops with 16GB RAM"
)


# Recommendation engine
if query:

    # Identify the product category from the user's query
    query_lower = query.lower()

    category_keywords = {
        "Laptop": ["laptop", "notebook", "computer"],
        "Printer": ["printer", "printing"],
        "Webcam": ["webcam", "camera", "video meeting"]
    }

    detected_category = None

    for category, keywords in category_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            detected_category = category
            break

    # Filter products when a category is detected
    if detected_category:
        filtered_products = products[
            products["category"] == detected_category
        ].copy()
    else:
        filtered_products = products.copy()

    # Combine product information
    product_text = (
        filtered_products["name"] + " " +
        filtered_products["category"] + " " +
        filtered_products["description"]
    )

    # Convert text into numerical representations
    vectorizer = TfidfVectorizer(stop_words="english")

    product_vectors = vectorizer.fit_transform(product_text)

    query_vector = vectorizer.transform([query])

    # Compare user query with products
    similarity_scores = cosine_similarity(
        query_vector,
        product_vectors
    ).flatten()

    filtered_products["match_score"] = similarity_scores

    # Get the most relevant products
    recommendations = filtered_products.sort_values(
        "match_score",
        ascending=False
    ).head(3)

    # Display recommendations
    st.subheader("Recommended Products")

    for _, product in recommendations.iterrows():

        score = round(product["match_score"] * 100)

        st.markdown(
            f"### {product['name']}"
        )

        st.write(
            f"**Category:** {product['category']}  \n"
            f"**Price:** ₹{product['price']:,}  \n"
            f"**Match Score:** {score}%"
        )

        st.write(product["description"])

        st.divider()
