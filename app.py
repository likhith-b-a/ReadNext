import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import utils.util as util
import utils.util_streamlit as helper
import utils.util_model as recommender

st.set_page_config(
    page_title=" ReadNext - Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Explicit variable sets for the Dark/Light toggle.
THEME_VARS = {
    "Light": {
        "--primary-color": "#7c3aed",
        "--secondary-color": "#9333ea",
        "--accent-color": "#0ea5e9",
        "--text-color": "#0f172a",
        "--light-text": "#475569",
        "--background-color": "#f8fafc",
        "--card-bg": "#ffffff",
        "--border-color": "#e2e8f0",
        "--shadow-color": "rgba(15, 23, 42, 0.08)",
        "--badge-bg": "rgba(124, 58, 237, 0.08)",
        "--badge-text": "#6d28d9",
    },
    "Dark": {
        "--primary-color": "#8b5cf6",
        "--secondary-color": "#a855f7",
        "--accent-color": "#38bdf8",
        "--text-color": "#f8fafc",
        "--light-text": "#cbd5e1",
        "--background-color": "#0f172a",
        "--card-bg": "#1e293b",
        "--border-color": "#334155",
        "--shadow-color": "rgba(0, 0, 0, 0.5)",
        "--badge-bg": "rgba(139, 92, 246, 0.18)",
        "--badge-text": "#c4b5fd",
    },
}

# App-wide containers Streamlit themes itself; we force these to our
# variables too so the toggle affects the whole app, not just the sidebar.
APP_CONTAINER_SELECTORS = (
    ".stApp",
    '[data-testid="stAppViewContainer"]',
    '[data-testid="stMain"]',
    '[data-testid="stHeader"]',
    '[data-testid="stBottomBlockContainer"]',
)

if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True

def inject_css():
    with open("./styles/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    theme = "Dark" if st.session_state.get("dark_mode", True) else "Light"
    var_rules = "; ".join(f"{k}: {v} !important" for k, v in THEME_VARS[theme].items())
    containers = ", ".join(APP_CONTAINER_SELECTORS)
    override_css = f"""
    :root {{ {var_rules}; }}
    {containers} {{
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
    }}
    """
    st.markdown(f"<style>{override_css}</style>", unsafe_allow_html=True)

inject_css()

def main():
    
    model_path = "./model/model.pkl"
    model_data = recommender.load_model(model_path)
    
    st.markdown("<h1 class='main-header'>📚 ReadNext: Book Recommendations</h1>", unsafe_allow_html=True)
    
    if model_data:
        books_df = model_data['books_df']

        #sidebar
        st.sidebar.toggle("🌙 Dark Mode", key="dark_mode")
        st.sidebar.divider()

        st.sidebar.markdown("### Dataset Statistics")
        st.sidebar.metric("Total Books", f"{len(books_df):,}")
        st.sidebar.metric("Total Authors", f"{books_df['book_author'].nunique():,}")
        st.sidebar.metric("Categories", f"{books_df['Category'].nunique():,}")
        st.sidebar.metric("Years", f"{int(books_df['year_of_publication'].min())}–{int(books_df['year_of_publication'].max())}")

        with st.sidebar:
            st.divider()
            st.markdown("### 🔄 Random Book Suggestion")
            if st.button("Suggest a Random Book"):
                book = books_df.sample(1).iloc[0]
                util.display_random_book(book)
                
        # User options
        tabs = st.tabs(["📖 Search by Title", "✍️ Search by Author", "🔍 Search by Keywords", "📊 Explore Data"])
        
        # Title search
        with tabs[0]:  
            st.markdown("<h2 class='sub-header'>Find Similar Books</h2>", unsafe_allow_html=True)
            st.write("Enter a book title to find similar books you might enjoy.")
                        
            input_title = helper.get_suggestion(books_df, "book_title", "Book Title", key_prefix="title")
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
            st.markdown("<h2 class='sub-header'>Find Books by Author</h2>", unsafe_allow_html=True)
            st.write("Enter an author's name to discover their books.")
            
            input_author = helper.get_suggestion(books_df, "book_author", "Author Name", key_prefix="author")
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
            st.markdown("<h2 class='sub-header'>Search by Keywords</h2>", unsafe_allow_html=True)
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
            st.markdown("<h2 class='sub-header'>Dataset Exploration</h2>", unsafe_allow_html=True)
            st.write("Explore the book dataset and gain insights.")
            
            # Data exploration options
            explore_option = st.selectbox("Select visualization:", 
                                       ["Category Distribution", "Publication Year Distribution", 
                                        "Authors with Most Books", "Popular Books per Year"])
            
            if explore_option == "Category Distribution":
                fig, ax = plt.subplots(figsize=(10, 8))
                category_counts = books_df['Category'].value_counts().head(20)
                sns.barplot(x=category_counts.values, y=category_counts.index, hue=category_counts.index, palette='viridis', legend=False, ax=ax)
                ax.set_title('Top 20 Book Categories')
                ax.set_xlabel('Number of Books')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show table of categories
                st.markdown("<h3 class='sub-header'>All Categories</h3>", unsafe_allow_html=True)
                all_categories = books_df['Category'].value_counts().reset_index()
                all_categories.columns = ['Category', 'Count']
                st.dataframe(all_categories)
                
            elif explore_option == "Publication Year Distribution":
                fig, ax = plt.subplots(figsize=(12, 6))
                years = books_df['year_of_publication'].dropna()
                years = years[(years > 1900) & (years < 2023)]  # Filter out erroneous years
                sns.histplot(years, bins=30, kde=True, ax=ax)
                ax.set_title('Book Publication Years')
                ax.set_xlabel('Year')
                ax.set_ylabel('Number of Books')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show publication year stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Earliest Year", int(years.min()))
                with col2:
                    st.metric("Latest Year", int(years.max()))
                with col3:
                    st.metric("Median Year", int(years.median()))
                
            elif explore_option == "Authors with Most Books":
                top_authors = books_df['book_author'].value_counts().head(20)
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.barplot(x=top_authors.values, y=top_authors.index, hue=top_authors.index, palette='coolwarm', legend=False, ax=ax)
                ax.set_title('Authors with Most Books')
                ax.set_xlabel('Number of Books')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show table of top authors
                st.markdown("<h3 class='sub-header'>Top Authors</h3>", unsafe_allow_html=True)
                top_authors_df = books_df['book_author'].value_counts().reset_index()
                top_authors_df.columns = ['Author', 'Number of Books']
                st.dataframe(top_authors_df.head(50))
                
            elif explore_option == "Popular Books per Year":
                # Group books by year and count
                books_per_year = books_df.groupby('year_of_publication').size().reset_index()
                books_per_year.columns = ['Year', 'Number of Books']
                books_per_year = books_per_year[(books_per_year['Year'] > 1900) & (books_per_year['Year'] < 2023)]
                
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.lineplot(x='Year', y='Number of Books', data=books_per_year, ax=ax)
                ax.set_title('Number of Books Published per Year')
                ax.set_xlabel('Year')
                ax.set_ylabel('Number of Books')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show years with most books
                st.markdown("<h3 class='sub-header'>Years with Most Publications</h3>", unsafe_allow_html=True)
                st.dataframe(books_per_year.sort_values('Number of Books', ascending=False).head(20))
        
    else:
        st.error("Failed to load the recommendation model. Please check the model path or upload a valid model file.")

if __name__ == "__main__":
    main()