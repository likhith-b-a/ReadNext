import streamlit as st
import utils.util as util
import utils.util_model as recommender

def advanced_filters(df, key_prefix=""):
    with st.expander("Advanced Filters"):
        col1, col2 = st.columns(2)

        exclude_cat = st.multiselect(
            "Exclude Categories",
            options=sorted(df['Category'].unique()),
            key=f"{key_prefix}_exclude_cat"
        )

        min_year, max_year = st.slider(
            "Publication Year Range",
            min_value=int(df['year_of_publication'].min()),
            max_value=int(df['year_of_publication'].max()),
            value=(
                int(df['year_of_publication'].min()),
                int(df['year_of_publication'].max())
            ),
            key=f"{key_prefix}_year_range"
        )

        top_n = st.slider(
            "Number of Recommendations" if key_prefix == "" else "Number of Books",
            min_value=5, max_value=20, value=10,
            key=f"{key_prefix}_top_n"
        )

    return exclude_cat, min_year, max_year, top_n

def get_suggestion(df, column, label, key_prefix=""):
    user_input = st.text_input(label, key=f"{key_prefix}_input")
    final_input = user_input

    if user_input:
        matches = df[df[column].str.contains(user_input, case=False, na=False)]
        if not matches.empty:
            suggestions = matches[column].unique()[:5]
            selected = st.selectbox(
                "Did you mean:", 
                ["Select a suggestion"] + list(suggestions), 
                key=f"{key_prefix}_select"
            )
            if selected != "Select a suggestion":
                final_input = selected

    return final_input

def run_recommendation(
    input_query, query_type, top_n, exclude_cat, min_year, max_year,
    display_function, model_data, books_df, input_label=None
):
    button_key = f"{query_type}_button"
    button_label = "Get Recommendations" if query_type == "title" else (
        "Find Books" if query_type == "author" else "Search"
    )

    if st.button(button_label, key=button_key):
        if input_query:
            with st.spinner(f"Finding books based on {input_label or query_type}..."):
                recs = util.recommend_books(
                    model_data,
                    query=input_query,
                    query_type=query_type,
                    top_n=top_n,
                    exclude_categories=exclude_cat if exclude_cat else None,
                    year_range=(min_year, max_year)
                )

                if recs is not None and not recs.empty:
                    explained_recs = (
                        recommender.explain_recommendations(recs, input_query, books_df)
                        if query_type == "title"
                        else recommender.explain_recommendations(recs)
                    )

                    st.success(f"Found {len(explained_recs)} recommendations for '{input_query}'")

                    for _, book in explained_recs.iterrows():
                        display_function(book)

                    with st.expander("📊 Visualization"):
                        util.visualize_recommendations(explained_recs, query_type)
                else:
                    st.warning(f"No results found for '{input_query}'. Try different input or filters.")
        else:
            st.warning(f"Please enter a {input_label or query_type}.")
