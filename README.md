# B2B Product Recommendation Assistant

An AI/ML-powered B2B product recommendation assistant built with Python and Streamlit.

## Overview

The application allows users to describe their business product requirements in natural language and receive relevant product recommendations from a sample B2B product catalogue.

For example:

> "I need laptops for office employees with 16GB RAM."

The system analyzes the requirement and ranks products based on their similarity to the user's request.

## AI/ML Approach

- **NLP:** Processes natural-language product requirements.
- **TF-IDF Vectorization:** Converts user requirements and product descriptions into numerical representations.
- **Cosine Similarity:** Calculates similarity between the requirement and available products.
- **Ranking:** Products are ranked based on their calculated match score.

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- NLP
- TF-IDF
- Cosine Similarity

## Key Features

- Natural-language product search
- AI/ML-based product matching
- Explainable match scores
- Product category and pricing information
- Simple business-friendly user interface

## How to Run

Install the required dependencies:

```bash
pip install streamlit pandas scikit-learn openai
streamlit run app.py
