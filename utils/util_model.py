from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import pickle

@st.cache_data
def load_model(model_path='model.pkl'):
    """Load the recommendation model from disk"""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error(f"Model file '{model_path}' not found. Please check the file path.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_recommendations_by_title(title, cosine_sim, df, indices, top_n=10):
    """
    1) Get index of Title
    2) Calculate cosine similarity
    3) Select the top_n books with similarity
    4) Add similarity score column to the dataframe of top_n books
    5) return Recommendations
    """
    try:
        idx = indices[title]
    except KeyError:
        return None
    
    # Get similarity scores
    sim_scores = list(enumerate(list(cosine_sim[idx])))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    
    book_indices = [i[0] for i in sim_scores]
    similarity_scores = [i[1] for i in sim_scores]
    
    recommendations = df.iloc[book_indices].copy()
    
    recommendations['similarity_score'] = similarity_scores
    
    return recommendations

def get_recommendations_by_author(author, df, top_n=10, exclude_categories=None, year_range=None):
    """
    1) Get books of the author
    2) Exclude the categories in exclude_categories
    3) Apply year_range
    4) Sort Based on the year_of publications(later can be made from ratings)
    """
    matching_books = df[df['book_author'].str.contains(author, case=False, na=False)]
    
    if matching_books.empty:
        return None
    
    recommendations = matching_books.sort_values('average_rating', ascending=False).head(top_n)
    return recommendations

def search_books_by_content(keywords, tfidf, tfidf_matrix, df, top_n=10):
    """
    1) Clean the Keywords provided
    2) Create TF-IDF vector for query
    3) Calculate cosine similarity
    4) Get the indices of the books which are more similar
    5) Create Dataframe of top_n similar books
    6) Add Relevence Score
    """
    
    query_vector = tfidf.transform([keywords])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    similar_indices = cosine_similarities.argsort()[-top_n:][::-1]
    similar_scores = cosine_similarities[similar_indices]
    
    recommendations = df.iloc[similar_indices].copy()
    recommendations['relevance_score'] = similar_scores
    
    recommendations = recommendations[recommendations['relevance_score'] > 0.05]
    
    return recommendations

def explain_recommendations(recommendations, original_title=None, books_df=None):
    """
    1) Check for same author
    2) Check for category
    3) Check for Similarity Score
    4) Check for Relevence Score
    """
    if recommendations is None or len(recommendations) == 0:
        return None
    
    explained_recs = recommendations.copy()
    explanations = []
    
    for _, row in recommendations.iterrows():
        explanation = []
        
        # If we have the original title
        if original_title and books_df is not None:
            original_book = books_df[books_df['book_title'] == original_title]
            if not original_book.empty:
                original_book = original_book.iloc[0]
                # Check if same author
                if row['book_author'] == original_book['book_author']:
                    explanation.append(f"Same author as '{original_title}'")
                
                # Check if same category
                if row['Category'] == original_book['Category']:
                    explanation.append(f"Same genre/category")
                else:
                    explanation.append(f"Different genre that you might enjoy")
        
        # Add similarity explanation
        if 'similarity_score' in row:
            if row['similarity_score'] > 0.55:
                explanation.append("Very similar content")
            elif row['similarity_score'] > 0.35:
                explanation.append("Moderately similar themes")
            else:
                explanation.append("Some thematic elements in common")
        elif 'relevance_score' in row:
            if row['relevance_score'] > 0.5:
                explanation.append("Highly relevant to your search")
            elif row['relevance_score'] > 0.35:
                explanation.append("Moderately relevant to your search")
            else:
                explanation.append("Somewhat relevant to your search")
        
        explanations.append(" - ".join(explanation))
    
    explained_recs['explanation'] = explanations
    
    return explained_recs
