import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import utils.util as util
import utils.util_streamlit as helper
import utils.util_model as recommender

st.set_page_config(
    page_title=" ReadNext - Recommendation System",
    page_icon=":material/menu_book:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Card-only CSS variables. Everything else in the app is themed natively via
# .streamlit/config.toml's [theme.light]/[theme.dark]; book cards stay custom
# HTML (see utils/util.py) since they need arbitrary category-colored badges
# and a cover-image layout native st.badge/st.container can't fully replicate.
CARD_THEME_VARS = {
    "light": {
        "--primary-color": "#7c3aed",
        "--secondary-color": "#9333ea",
        "--card-bg": "#ffffff",
        "--border-color": "#e2e8f0",
        "--shadow-color": "rgba(15, 23, 42, 0.08)",
        "--badge-bg": "rgba(124, 58, 237, 0.08)",
        "--badge-text": "#6d28d9",
        "--light-text": "#475569",
    },
    "dark": {
        "--primary-color": "#8b5cf6",
        "--secondary-color": "#a855f7",
        "--card-bg": "#1e293b",
        "--border-color": "#334155",
        "--shadow-color": "rgba(0, 0, 0, 0.5)",
        "--badge-bg": "rgba(139, 92, 246, 0.18)",
        "--badge-text": "#c4b5fd",
        "--light-text": "#cbd5e1",
    },
}

def inject_card_css():
    st.html("./styles/styles.css")
    theme_type = st.context.theme.type or "light"
    var_rules = "; ".join(f"{k}: {v}" for k, v in CARD_THEME_VARS[theme_type].items())
    st.html(f"<style>:root {{ {var_rules}; }}</style>")

inject_card_css()

def main():

    model_path = "./model/model.pkl"
    model_data = recommender.load_model(model_path)

    st.title(":material/menu_book: ReadNext: Book recommendations")

    if model_data:
        books_df = model_data['books_df']

        #sidebar
        st.sidebar.subheader("Dataset statistics")
        st.sidebar.metric("Total Books", f"{len(books_df):,}", border=True)
        st.sidebar.metric("Total Authors", f"{books_df['book_author'].nunique():,}", border=True)
        st.sidebar.metric("Categories", f"{books_df['Category'].nunique():,}", border=True)
        st.sidebar.metric("Years", f"{int(books_df['year_of_publication'].min())}–{int(books_df['year_of_publication'].max())}", border=True)

        with st.sidebar:
            st.subheader(":material/casino: Suggest a random book")
            if st.button("Suggest a random book", icon=":material/casino:"):
                book = books_df.sample(1).iloc[0]
                util.display_random_book(book)

        # User options
        tabs = st.tabs([
            ":material/menu_book: Search by title",
            ":material/edit: Search by author",
            ":material/search: Search by keywords",
            ":material/bar_chart: Explore data",
        ])

        # Title search
        with tabs[0]:
            st.header("Find similar books")
            st.write("Enter a book title to find similar books you might enjoy.")

            input_title = helper.get_suggestion(books_df, "book_title", "Book title", key_prefix="title")
            exclude_cat, min_year, max_year, top_n = helper.advanced_filters(books_df, key_prefix="title")

            helper.run_recommendation(
                input_query=input_title,
                query_type="title",
                top_n=top_n,
                exclude_cat=exclude_cat,
                min_year=min_year,
                max_year=max_year,
                display_function=util.display_book_card_with_image,
                model_data=model_data,
                books_df=books_df,
                input_label="book title"
            )

        # Author search
        with tabs[1]:
            st.header("Find books by author")
            st.write("Enter an author's name to discover their books.")

            input_author = helper.get_suggestion(books_df, "book_author", "Author name", key_prefix="author")
            exclude_cat, min_year, max_year, top_n = helper.advanced_filters(books_df, key_prefix="author")

            helper.run_recommendation(
                input_query=input_author,
                query_type="author",
                top_n=top_n,
                exclude_cat=exclude_cat,
                min_year=min_year,
                max_year=max_year,
                display_function=util.display_book_card_with_image_for_author,
                model_data=model_data,
                books_df=books_df,
                input_label="author name"
            )

        # Keyword search
        with tabs[2]:
            st.header("Search by keywords")
            st.write("Enter keywords to find related books.")

            input_keywords = st.text_input("Keywords (e.g., mystery detective crime)", key="keywords_input")
            exclude_cat, min_year, max_year, top_n = helper.advanced_filters(books_df, key_prefix="keywords")

            helper.run_recommendation(
                input_query=input_keywords,
                query_type="keywords",
                top_n=top_n,
                exclude_cat=exclude_cat,
                min_year=min_year,
                max_year=max_year,
                display_function=util.display_book_card_with_image,
                model_data=model_data,
                books_df=books_df,
                input_label="keywords"
            )

        with tabs[3]:  # Data exploration
            st.header("Explore dataset")
            st.write("Explore the book dataset and gain insights.")

            # Data exploration options
            explore_option = st.selectbox("Select visualization:",
                                       ["Category distribution", "Publication year distribution",
                                        "Authors with most books", "Popular books per year"])

            if explore_option == "Category distribution":
                fig, ax = plt.subplots(figsize=(10, 8))
                category_counts = books_df['Category'].value_counts().head(20)
                sns.barplot(x=category_counts.values, y=category_counts.index, hue=category_counts.index, palette='viridis', legend=False, ax=ax)
                ax.set_title('Top 20 book categories')
                ax.set_xlabel('Number of books')
                plt.tight_layout()
                st.pyplot(fig)

                # Show table of categories
                st.subheader("All categories")
                all_categories = books_df['Category'].value_counts().reset_index()
                all_categories.columns = ['Category', 'Count']
                st.dataframe(all_categories)

            elif explore_option == "Publication year distribution":
                fig, ax = plt.subplots(figsize=(12, 6))
                years = books_df['year_of_publication'].dropna()
                years = years[(years > 1900) & (years < 2023)]  # Filter out erroneous years
                sns.histplot(years, bins=30, kde=True, ax=ax)
                ax.set_title('Book publication years')
                ax.set_xlabel('Year')
                ax.set_ylabel('Number of books')
                plt.tight_layout()
                st.pyplot(fig)

                # Show publication year stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Earliest year", int(years.min()), border=True)
                with col2:
                    st.metric("Latest year", int(years.max()), border=True)
                with col3:
                    st.metric("Median year", int(years.median()), border=True)

            elif explore_option == "Authors with most books":
                top_authors = books_df['book_author'].value_counts().head(20)
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.barplot(x=top_authors.values, y=top_authors.index, hue=top_authors.index, palette='coolwarm', legend=False, ax=ax)
                ax.set_title('Authors with most books')
                ax.set_xlabel('Number of books')
                plt.tight_layout()
                st.pyplot(fig)

                # Show table of top authors
                st.subheader("Top authors")
                top_authors_df = books_df['book_author'].value_counts().reset_index()
                top_authors_df.columns = ['Author', 'Number of Books']
                st.dataframe(top_authors_df.head(50))

            elif explore_option == "Popular books per year":
                # Group books by year and count
                books_per_year = books_df.groupby('year_of_publication').size().reset_index()
                books_per_year.columns = ['Year', 'Number of Books']
                books_per_year = books_per_year[(books_per_year['Year'] > 1900) & (books_per_year['Year'] < 2023)]

                fig, ax = plt.subplots(figsize=(12, 6))
                sns.lineplot(x='Year', y='Number of Books', data=books_per_year, ax=ax)
                ax.set_title('Number of books published per year')
                ax.set_xlabel('Year')
                ax.set_ylabel('Number of books')
                plt.tight_layout()
                st.pyplot(fig)

                # Show years with most books
                st.subheader("Years with most publications")
                st.dataframe(books_per_year.sort_values('Number of Books', ascending=False).head(20))

    else:
        st.error("Failed to load the recommendation model. Please check the model path or upload a valid model file.")

if __name__ == "__main__":
    main()
