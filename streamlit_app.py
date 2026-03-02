"""
Streamlit demo REIC - Phân loại intent (ý định người dùng).
Chạy: uv run streamlit run streamlit_app.py   (hoặc: source .venv/bin/activate && streamlit run streamlit_app.py)
"""

from pathlib import Path

import streamlit as st

# Project root
ROOT = Path(__file__).resolve().parent


def load_pipeline(backend: str, top_k: int, use_llm: bool, use_local_llm: bool):
    from reic.pipeline import ReicPipeline

    ontology_path = ROOT / "data" / "ontology.json"
    return ReicPipeline(
        ontology_path,
        top_k=top_k,
        backend=backend,
        use_llm=use_llm,
        use_local_llm=use_local_llm,
    )


def main():
    st.set_page_config(
        page_title="REIC Demo",
        page_icon="🎯",
        layout="wide",
    )
    st.title("🎯 REIC – Phân loại Intent")
    st.caption("Retrieval-Enhanced Intent Classification | Query → Retrieve → Rerank → Intent")

    # Sidebar
    with st.sidebar:
        st.header("Cấu hình")
        backend = st.selectbox(
            "Retrieval backend",
            ["tfidf", "dense"],
            index=0,
            help="tfidf: nhanh, không tải model. dense: sentence-transformers, chất lượng cao hơn.",
        )
        top_k = st.slider("Top-k candidates", 2, 10, 5)
        use_llm = st.checkbox("Dùng OpenAI reranker", False, help="Cần OPENAI_API_KEY")
        use_local_llm = st.checkbox("Dùng Local LLM (1.5B)", False, help="Qwen2-1.5B, constrained decoding")
        if use_local_llm and use_llm:
            st.warning("Chỉ nên bật một: OpenAI hoặc Local LLM.")
        st.divider()
    # Cache pipeline
    key = f"{backend}_{top_k}_{use_llm}_{use_local_llm}"
    if key not in st.session_state:
        with st.spinner("Đang tải pipeline..."):
            st.session_state[key] = load_pipeline(backend, top_k, use_llm, use_local_llm)
    pipeline = st.session_state[key]

    # Input
    query = st.text_area(
        "Nhập câu hỏi của khách hàng",
        placeholder="VD: Update my shipping address / Tôi muốn trả hàng / Where is my package?",
        height=80,
    )

    if not query.strip():
        st.info("Nhập câu hỏi và nhấn **Phân loại**.")
        return

    if st.button("Phân loại", type="primary"):
        with st.spinner("Đang xử lý..."):
            result = pipeline.predict(query.strip())

        # Kết quả chính
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Intent dự đoán", result.predicted_intent or "—")
        with col2:
            st.metric("Confidence P(t̂|q,E)", f"{result.confidence:.2%}" if result.predicted_intent else "—")
        with col3:
            st.metric("Vertical", result.vertical)

        if result.path:
            st.markdown(f"**Path:** `{' → '.join(result.path)}`")

        # P(tj|q,E)
        if result.intent_probabilities:
            st.subheader("P(tj|q,E)")
            import pandas as pd
            df = (
                pd.DataFrame(
                    list(result.intent_probabilities.items()),
                    columns=["Intent", "Xác suất"],
                )
                .sort_values("Xác suất", ascending=False)
            )
            st.bar_chart(df.set_index("Intent"))

        # Top candidates (retrieval)
        if result.candidates:
            with st.expander("Top candidates (retrieval)"):
                for c in result.candidates:
                    st.markdown(f"- **{c.name}** (`{c.intent_id}`) — score: {c.score:.3f}")
                    st.caption(f"Example: {c.example[:80]}..." if len(c.example) > 80 else f"Example: {c.example}")


if __name__ == "__main__":
    main()
