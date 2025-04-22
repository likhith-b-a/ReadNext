import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import utils.util_model as recommender

broken_urls = []

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

        if width <= 1 and height <= 1:
            broken_urls.append(url)
            return False
        else:
            return True

    except (UnidentifiedImageError, requests.RequestException) as e:
        broken_urls.append(url)
        return False

def recommend_books(model_data, query=None, query_type='title', top_n=10, 
                    exclude_categories=None, year_range=None, include_keywords=None):
    """ Function for Defferent kind of queries """
    
    if model_data is None or query is None:
        return None
    
    # Extract model components
    tfidf = model_data['tfidf_vectorizer']
    tfidf_matrix = model_data['tfidf_matrix']
    cosine_sim = model_data['cosine_sim']
    indices = model_data['indices']
    books_df = model_data['books_df']
    
    # Get base recommendations based on query type
    if query_type.lower() == 'title':
        recommendations = recommender.get_recommendations_by_title(query, cosine_sim, books_df, indices, top_n=top_n*2)
    
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
        bars = sns.barplot(x=category_counts.values, y=category_counts.index, palette=palette, ax=ax)
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
        bars = sns.barplot(x=scores, y=books, palette=palette, ax=ax)
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
        bars = sns.barplot(x=scores, y=books, palette=palette, ax=ax)
        ax.set_title('Relevance Scores', fontsize=14, fontweight='bold')
        ax.set_xlabel('Relevance Score', fontsize=12)
        ax.set_ylabel('Book', fontsize=12)
        
        # Add value labels
        for i, v in enumerate(scores):
            ax.text(v + 0.01, i, f"{v:.2f}", va='center', fontweight='bold')
            
        plt.tight_layout()
        st.pyplot(fig)

def display_book_card_with_image(book):
    """
    Display a book card with book cover image and details
    """
    # Prepare image HTML
    if 'img_l' in book and book['img_l'] and book['img_l'] not in broken_urls and is_valid_image(book['img_l']):
        img_html = f'<img src="{book["img_l"]}" width="150">'
    else:
        img_html = '<img src="https://placehold.co/150x200?text=No+Image" height="245" width="150">'
    
    # Prepare optional fields
    category_html = f'<div>Category: {book["Category"]}</div>' if 'Category' in book else ''
    year_html = f'<div>Year: {book["year_of_publication"]}</div>' if 'year_of_publication' in book else ''
    
    # Prepare score HTML
    score_html = ''
    if 'similarity_score' in book:
        score_html = f'<div>Similarity: {book["similarity_score"]:.2f}</div>'
    elif 'relevance_score' in book:
        score_html = f'<div>Relevance: {book["relevance_score"]:.2f}</div>'
    
    # Prepare explanation HTML
    explanation_html = f'<div class="explanation">{book["explanation"]}</div>' if 'explanation' in book else ''
    
    # Combine all HTML parts
    card_content = f"""
    <div class='card'>
        <div class='row'>
            <div class='col-1'>
                {img_html}
            </div>
            <div class='col-3'>
                <div class='book-title'>{book['book_title']}</div>
                <div class='book-author'>by {book['book_author']}</div>
                {category_html}
                {year_html}
                {score_html}
                <div class='book-author'>Average Rating: {book['average_rating']}</div>
                {explanation_html}
            </div>
        </div>
    </div>
    """

    st.markdown(card_content, unsafe_allow_html=True)
    
def display_book_card_with_image_for_author(book):
    """
    Display a book card with book cover image and details
    """
    if 'img_l' in book and book['img_l'] and book['img_l'] not in broken_urls and is_valid_image(book['img_l']):
        img_html = f'<img src="{book["img_l"]}" width="150">'
    else:
        img_html = '<img src="https://placehold.co/150x200?text=No+Image" width="150">'
    
    category_html = f'<div>Category: {book["Category"]}</div>' if 'Category' in book else ''
    year_html = f'<div>Year: {book["year_of_publication"]}</div>' if 'year_of_publication' in book else ''
    
    card_content = f"""
    <div class='card'>
        <div class='row'>
            <div class='col-1'>
                {img_html}
            </div>
            <div class='col-3'>
                <div class='book-title'>{book['book_title']}</div>
                <div class='book-author'>by {book['book_author']}</div>
                <br><br><br><br>
                {category_html}
                {year_html}
                <div class='book-author'>Average Rating: {book['average_rating']}</div>
            </div>
        </div>
    </div>
    """

    st.markdown(card_content, unsafe_allow_html=True)
    
def display_random_book(book):
    """
    Display a book card with book cover image and details
    """
    if 'img_l' in book and book['img_l'] and book['img_l'] not in broken_urls and is_valid_image(book['img_l']):
        # img_html = f'<img src="{book["img_l"]}" width="150">'
        img_html = f'<div style="text-align: center;"><img src="{book["img_l"]}" width="150"></div>'

    else:
        # img_html = '<img src="https://placehold.co/150x200?text=No+Image" width="150">'
        img_html = '<div style="text-align: center;"><img src="https://placehold.co/150x200?text=No+Image" width="150"></div>'

    
    category_html = f'<div>Category: {book["Category"]}</div>' if 'Category' in book else ''
    year_html = f'<div>Year: {book["year_of_publication"]}</div>' if 'year_of_publication' in book else ''
    
    card_content = f"""
    <div class='card'>
        <div >
            {img_html}
            <br><br>
            <div class='book-title'>{book['book_title']}</div>
            <div class='book-author'>by {book['book_author']}</div>
            {category_html}
            {year_html}
            <div class='book-author'>Average Rating: {book['average_rating']}</div>
        </div>
    </div>
    """

    st.markdown(card_content, unsafe_allow_html=True)