"""Product Return Prediction — Streamlit UI.

Talks to the FastAPI backend at API_URL (env var, defaults to localhost:8001).
Works locally, on Streamlit Cloud, and on EKS (set API_URL to the internal
service DNS: http://product-return-api-svc:8000).
"""

import os
import json
import requests
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8001").rstrip("/")

st.set_page_config(
    page_title="Product Return Risk",
    page_icon="📦",
    layout="wide",
)

# ── helpers ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_model_info():
    try:
        r = requests.get(f"{API_URL}/model-info", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def call_predict(payload: dict):
    r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


# ── sidebar — model info ──────────────────────────────────────────────────────

with st.sidebar:
    st.title("📦 Return Risk Predictor")
    st.caption("Powered by Logistic Regression + MLOps pipeline")

    info = fetch_model_info()
    if "error" in info:
        st.error(f"API unreachable: {info['error']}")
    else:
        st.subheader("Model metrics")
        metrics = {m["name"]: m["value"] for m in info.get("metrics", [])}
        col1, col2 = st.columns(2)
        col1.metric("F1", f"{metrics.get('f1', 0):.3f}")
        col2.metric("Recall", f"{metrics.get('recall', 0):.3f}")
        col1.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        col2.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")

        st.caption(f"Threshold: {info.get('threshold', 0.3)}")
        st.caption(f"Model: {info.get('model_name', '—')}")

        fi = info.get("feature_importance", [])
        if fi:
            st.subheader("Top features")
            fi_df = pd.DataFrame(fi).head(10)
            st.bar_chart(fi_df.set_index("feature")["importance"])


# ── tabs ──────────────────────────────────────────────────────────────────────

tab_predict, tab_bulk, tab_curve = st.tabs(
    ["🔍 Single prediction", "📋 Bulk predictions", "📈 Cost curve"]
)

# ─── tab 1: single prediction ────────────────────────────────────────────────

with tab_predict:
    st.header("Predict return risk for one order")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Product**")
            product_category = st.selectbox(
                "Category",
                ["Electronics", "Clothing", "Home & Kitchen", "Sports", "Books", "Toys"],
            )
            sub_category = st.text_input("Sub-category", value="Smartphones")
            brand = st.text_input("Brand", value="BrandA")
            product_price = st.number_input("Price (₹)", min_value=1.0, value=499.99)
            discount_percent = st.slider("Discount %", 0, 100, 15)
            product_rating = st.slider("Product rating", 0.0, 5.0, 4.2, 0.1)
            review_count = st.number_input("Review count", min_value=0, value=120)
            fragile_item = st.checkbox("Fragile item")
            warranty_available = st.checkbox("Warranty available", value=True)

        with c2:
            st.markdown("**Return history rates**")
            product_return_rate = st.slider("Product return rate", 0.0, 1.0, 0.12, 0.01)
            category_return_rate = st.slider("Category return rate", 0.0, 1.0, 0.18, 0.01)
            brand_return_rate = st.slider("Brand return rate", 0.0, 1.0, 0.10, 0.01)
            defect_rate = st.slider("Defect rate", 0.0, 1.0, 0.03, 0.01)
            seller_rating = st.slider("Seller rating", 0.0, 5.0, 4.5, 0.1)
            seller_return_rate = st.slider("Seller return rate", 0.0, 1.0, 0.08, 0.01)

        with c3:
            st.markdown("**Order details**")
            fulfillment_type = st.selectbox(
                "Fulfillment",
                ["Marketplace Fulfilled", "Seller Fulfilled", "Express"],
            )
            payment_method = st.selectbox(
                "Payment",
                ["Credit Card", "Debit Card", "UPI", "Cash On Delivery", "Net Banking"],
            )
            quantity = st.number_input("Quantity", min_value=1, value=1)
            shipping_distance_km = st.number_input("Shipping distance (km)", min_value=0.0, value=250.0)
            delayed_delivery = st.checkbox("Delayed delivery")
            wishlist_before_purchase = st.checkbox("Was in wishlist")
            product_page_views = st.number_input("Page views", min_value=0, value=35)
            customer_support_calls = st.number_input("Support calls", min_value=0, value=1)
            chat_interactions = st.number_input("Chat interactions", min_value=0, value=0)

        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if submitted:
        payload = {
            "product_category": product_category,
            "sub_category": sub_category,
            "brand": brand,
            "product_price": product_price,
            "discount_percent": float(discount_percent),
            "product_rating": product_rating,
            "review_count": float(review_count),
            "fragile_item": int(fragile_item),
            "warranty_available": int(warranty_available),
            "product_return_rate": product_return_rate,
            "category_return_rate": category_return_rate,
            "brand_return_rate": brand_return_rate,
            "defect_rate": defect_rate,
            "seller_rating": seller_rating,
            "seller_return_rate": seller_return_rate,
            "fulfillment_type": fulfillment_type,
            "payment_method": payment_method,
            "quantity": quantity,
            "shipping_distance_km": shipping_distance_km,
            "delayed_delivery": int(delayed_delivery),
            "wishlist_before_purchase": int(wishlist_before_purchase),
            "product_page_views": int(product_page_views),
            "customer_support_calls": int(customer_support_calls),
            "chat_interactions": int(chat_interactions),
        }

        try:
            result = call_predict(payload)
            prob = result["return_probability"]
            risk = result["risk_level"]

            color = {"Low": "green", "Medium": "orange", "High": "red"}[risk]
            r1, r2, r3 = st.columns(3)
            r1.metric("Return probability", f"{prob:.1%}")
            r2.metric("Risk level", risk)
            r3.metric("Will be returned", "Yes" if result["predicted_returned"] else "No")

            st.progress(prob, text=f"Return probability: {prob:.1%}")
            if risk == "High":
                st.error("High risk — consider review before dispatch.")
            elif risk == "Medium":
                st.warning("Medium risk — monitor this order.")
            else:
                st.success("Low risk — safe to ship.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ─── tab 2: bulk predictions ─────────────────────────────────────────────────

with tab_bulk:
    st.header("Batch predictions from CSV")
    st.markdown(
        "Upload a CSV with the same columns as the single prediction form. "
        "The app calls the API for each row and shows results."
    )

    sample_cols = [
        "product_category", "sub_category", "brand", "product_price",
        "discount_percent", "product_rating", "review_count", "fragile_item",
        "warranty_available", "product_return_rate", "category_return_rate",
        "brand_return_rate", "defect_rate", "seller_rating", "seller_return_rate",
        "fulfillment_type", "payment_method", "quantity", "shipping_distance_km",
        "delayed_delivery", "wishlist_before_purchase", "product_page_views",
        "customer_support_calls", "chat_interactions",
    ]

    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        missing = [c for c in sample_cols if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            st.write(f"Loaded {len(df)} rows. Running predictions…")
            results = []
            progress = st.progress(0)
            for i, row in df.iterrows():
                try:
                    res = call_predict(row[sample_cols].to_dict())
                    results.append(res)
                except Exception as e:
                    results.append({"return_probability": None, "predicted_returned": None, "risk_level": "Error"})
                progress.progress((i + 1) / len(df))

            out = df.copy()
            out["return_probability"] = [r["return_probability"] for r in results]
            out["predicted_returned"] = [r["predicted_returned"] for r in results]
            out["risk_level"] = [r["risk_level"] for r in results]

            high = (out["risk_level"] == "High").sum()
            med = (out["risk_level"] == "Medium").sum()
            low = (out["risk_level"] == "Low").sum()

            m1, m2, m3 = st.columns(3)
            m1.metric("High risk", high)
            m2.metric("Medium risk", med)
            m3.metric("Low risk", low)

            st.dataframe(
                out[["product_category", "brand", "return_probability", "risk_level", "predicted_returned"]],
                use_container_width=True,
            )

            csv_out = out.to_csv(index=False).encode()
            st.download_button("Download results CSV", csv_out, "predictions.csv", "text/csv")

    else:
        st.info("Upload a CSV to start. Columns needed: " + ", ".join(sample_cols[:6]) + " …")


# ─── tab 3: cost curve ───────────────────────────────────────────────────────

with tab_curve:
    st.header("Threshold cost curve")
    st.markdown(
        "Cost model: **missed return (FN) = 5× the cost of a false alarm (FP)**. "
        "The optimal threshold minimises total cost."
    )

    curve_path = os.path.join(os.path.dirname(__file__), "..", "artifacts", "threshold_cost_curve.csv")
    try:
        curve = pd.read_csv(curve_path)
        opt = curve.loc[curve["cost"].idxmin()]

        st.metric("Optimal threshold", f"{opt['threshold']:.2f}", help="Minimises FN×5 + FP cost")

        import altair as alt
        base = alt.Chart(curve).mark_line(color="#4C78A8").encode(
            x=alt.X("threshold:Q", title="Decision threshold"),
            y=alt.Y("cost:Q", title="Total cost"),
            tooltip=["threshold", "cost", "false_pos", "false_neg"],
        )
        rule = alt.Chart(pd.DataFrame({"threshold": [opt["threshold"]]})).mark_rule(
            color="red", strokeDash=[4, 4]
        ).encode(x="threshold:Q")
        st.altair_chart(base + rule, use_container_width=True)

        c1, c2 = st.columns(2)
        fp_chart = alt.Chart(curve).mark_line(color="orange").encode(
            x="threshold:Q", y=alt.Y("false_pos:Q", title="False positives"),
        )
        fn_chart = alt.Chart(curve).mark_line(color="steelblue").encode(
            x="threshold:Q", y=alt.Y("false_neg:Q", title="False negatives"),
        )
        c1.altair_chart(fp_chart + rule, use_container_width=True)
        c2.altair_chart(fn_chart + rule, use_container_width=True)

    except FileNotFoundError:
        st.warning("threshold_cost_curve.csv not found — run the DVC pipeline first.")
