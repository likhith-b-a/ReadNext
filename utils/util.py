import html
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import utils.util_model as recommender

NO_IMAGE_PLACEHOLDER = "https://placehold.co/150x200?text=No+Image"

@st.cache_data(show_spinner=False, ttl=86400)
def is_valid_image(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": "https://www.amazon.com/"
    }
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content))
        width, height = img.size

        return width > 1 or height > 1

    except (UnidentifiedImageError, requests.RequestException):
        return False

def recommend_books(model_data, query=None, query_type='title', top_n=10, 
                    exclude_categories=None, year_range=None, include_keywords=None):
    """ Function for Defferent kind of queries """
    
    if model_data is None or query is None:
        return None
    
    # Extract model components
    tfidf = model_data['tfidf_vectorizer']
    tfidf_matrix = model_data['tfidf_matrix']
    # cosine_sim is computed on the fly now 
    indices = model_data['indices']
    books_df = model_data['books_df']
    
    # Get base recommendations based on query type
    if query_type.lower() == 'title':
        recommendations = recommender.get_recommendations_by_title(query, tfidf_matrix, books_df, indices, top_n=top_n*2)
    
    elif query_type.lower() == 'author':
        recommendations = recommender.get_recommendations_by_author(query, books_df, top_n=top_n*2, exclude_categories=exclude_categories, year_range=year_range)
    
    elif query_type.lower() == 'keywords': 
        recommendations = recommender.search_books_by_content(query, tfidf, tfidf_matrix, books_df, top_n=top_n*2)
    
    else:
        st.error("Invalid query type. Choose 'title', 'author', or 'keywords'.")
        return None
    
    if recommendations is None or len(recommendations) == 0:
        return None
    
    # Apply category filter if specified (for title and keywords)
    if exclude_categories:
        if not isinstance(exclude_categories, list):
            exclude_categories = [exclude_categories]
        
        for category in exclude_categories:
            recommendations = recommendations[~recommendations['Category'].str.contains(category, case=False, na=False)]

    # Apply year range filter if specified (for title and keywords)
    if year_range and len(year_range) == 2 and query_type.lower() != 'author':
        min_year, max_year = year_range
        recommendations = recommendations[
            (recommendations['year_of_publication'] >= min_year) & 
            (recommendations['year_of_publication'] <= max_year)
        ]
    
    # Filter by keywords if specified
    if include_keywords:
        # Get books containing the keywords
        keyword_results = recommender.search_books_by_content(include_keywords, tfidf, tfidf_matrix, books_df, top_n=len(books_df))
        keyword_books = set(keyword_results['book_title'])
        
        # Only keep recommendations that are in the keyword results
        recommendations = recommendations[recommendations['book_title'].isin(keyword_books)]
    
    # Return top N results
    return recommendations.head(top_n)

def visualize_recommendations(recommendations, query_type):
    """Create visualizations for recommendation results"""
    if recommendations is None or len(recommendations) == 0:
        return
    
    st.markdown("<h3 style='font-size: 1.5rem; color: #1e3a8a; margin-bottom: 1rem;'>📊 Insights from Your Recommendations</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 7))
        category_counts = recommendations['Category'].value_counts()
        palette = sns.color_palette("Blues_r", len(category_counts))
        bars = sns.barplot(x=category_counts.values, y=category_counts.index, hue=category_counts.index, palette=palette, legend=False, ax=ax)
        ax.set_title('Category Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Books', fontsize=12)
        ax.set_ylabel('Category', fontsize=12)
        # Add data labels to bars
        for i, v in enumerate(category_counts.values):
            ax.text(v + 0.1, i, str(v), va='center')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 7))
        years = recommendations['year_of_publication'].astype(int)
        sns.histplot(years, bins=min(10, len(years.unique())), kde=True, ax=ax, color='#3b82f6', line_kws={'color': '#1e40af'})
        ax.set_title('Publication Year Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
    
    # Similarity/relevance scores
    if query_type == 'title' and 'similarity_score' in recommendations.columns:
        fig, ax = plt.subplots(figsize=(12, 8))
        books = recommendations['book_title'].str[:30] + '...'
        scores = recommendations['similarity_score']
        
        # Color gradient based on score
        palette = sns.color_palette("Blues", len(scores))
        bars = sns.barplot(x=scores, y=books, hue=books, palette=palette, legend=False, ax=ax)
        ax.set_title('Similarity Scores', fontsize=14, fontweight='bold')
        ax.set_xlabel('Similarity Score', fontsize=12)
        ax.set_ylabel('Book', fontsize=12)
        
        # Add value labels
        for i, v in enumerate(scores):
            ax.text(v + 0.01, i, f"{v:.2f}", va='center', fontweight='bold')
            
        plt.tight_layout()
        st.pyplot(fig)
    elif query_type == 'keywords' and 'relevance_score' in recommendations.columns:
        fig, ax = plt.subplots(figsize=(12, 8))
        books = recommendations['book_title'].str[:30] + '...'
        scores = recommendations['relevance_score']
        
        # Color gradient based on score
        palette = sns.color_palette("Blues", len(scores))
        bars = sns.barplot(x=scores, y=books, hue=books, palette=palette, legend=False, ax=ax)
        ax.set_title('Relevance Scores', fontsize=14, fontweight='bold')
        ax.set_xlabel('Relevance Score', fontsize=12)
        ax.set_ylabel('Book', fontsize=12)
        
        # Add value labels
        for i, v in enumerate(scores):
            ax.text(v + 0.01, i, f"{v:.2f}", va='center', fontweight='bold')
            
        plt.tight_layout()
        st.pyplot(fig)

def _book_cover_html(book):
    """Cover image if it validates, otherwise a consistently-sized placeholder."""
    img_url = book['img_l'] if 'img_l' in book and pd.notna(book['img_l']) else None
    if img_url and is_valid_image(img_url):
        src = html.escape(img_url, quote=True)
        return f'<img class="book-cover" src="{src}" alt="Book cover">'
    return f'<img class="book-cover" src="{NO_IMAGE_PLACEHOLDER}" alt="No cover available">'

def _render_book_card(book, show_explanation=False, centered=False):
    """Shared card renderer for title/author/keyword results and the random suggestion."""
    img_html = _book_cover_html(book)
    title = html.escape(str(book['book_title']))
    author = html.escape(str(book['book_author']))

    badges = []
    if 'Category' in book and pd.notna(book['Category']):
        badges.append(f'<span class="badge">{html.escape(str(book["Category"]))}</span>')
    if 'year_of_publication' in book and pd.notna(book['year_of_publication']):
        badges.append(f'<span class="badge badge-year">{int(book["year_of_publication"])}</span>')
    badges_html = f'<div class="badges">{"".join(badges)}</div>' if badges else ''

    rating_html = ''
    if 'average_rating' in book and pd.notna(book['average_rating']):
        rating_html = f'<div class="book-rating">★ {float(book["average_rating"]):.1f}</div>'

    explanation_html = ''
    if show_explanation and 'explanation' in book and book['explanation']:
        explanation_html = f'<div class="explanation">{html.escape(str(book["explanation"]))}</div>'

    body_html = f"""
                <div class='book-title'>{title}</div>
                <div class='book-author'>by {author}</div>
                {badges_html}
                {rating_html}
                {explanation_html}
    """

    if centered:
        return f"""
    <div class='card card-centered'>
        <div class='book-cover-wrap'>{img_html}</div>
        {body_html}
    </div>
    """

    return f"""
    <div class='card'>
        <div class='row'>
            <div class='col-1'>
                {img_html}
            </div>
            <div class='col-3'>
                {body_html}
            </div>
        </div>
    </div>
    """

def display_book_card_with_image(book):
    """Card for title/keyword search results, with the match explanation."""
    st.html(_render_book_card(book, show_explanation=True))

def display_book_card_with_image_for_author(book):
    """Card for author search results (no match explanation)."""
    st.html(_render_book_card(book, show_explanation=False))

def display_random_book(book):
    """Centered card for the sidebar's random book suggestion."""
    st.html(_render_book_card(book, show_explanation=False, centered=True))