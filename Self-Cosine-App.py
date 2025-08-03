
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
from io import StringIO
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Cosine Similarity Checker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .similarity-high {
        background-color: #d4edda;
        color: #155724;
    }
    .similarity-medium {
        background-color: #fff3cd;
        color: #856404;
    }
    .similarity-low {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

def preprocess_text(text):
    """
    Preprocess text by converting to lowercase, removing punctuation,
    and handling extra whitespace.
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[' + string.punctuation + ']', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def read_uploaded_file(uploaded_file):
    """
    Read content from uploaded file with error handling.
    """
    try:
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                content = str(uploaded_file.read(), encoding)
                uploaded_file.seek(0)  # Reset file pointer
                return content
            except UnicodeDecodeError:
                uploaded_file.seek(0)  # Reset file pointer
                continue
        
        # If all encodings fail, return error message
        return None
    except Exception as e:
        st.error(f"Error reading file {uploaded_file.name}: {str(e)}")
        return None

def calculate_similarity_matrix(queries, documents, doc_names):
    """
    Calculate cosine similarity matrix between queries and documents.
    """
    try:
        # Preprocess all texts
        processed_queries = [preprocess_text(query) for query in queries]
        processed_docs = [preprocess_text(doc) for doc in documents]
        
        # Combine all texts for vectorization
        all_texts = processed_queries + processed_docs
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        # Fit and transform all texts
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Split back into queries and documents
        query_vectors = tfidf_matrix[:len(queries)]
        doc_vectors = tfidf_matrix[len(queries):]
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(query_vectors, doc_vectors)
        
        return similarity_matrix, vectorizer
        
    except Exception as e:
        st.error(f"Error calculating similarity: {str(e)}")
        return None, None

def create_similarity_dataframe(similarity_matrix, queries, doc_names):
    """
    Create a formatted DataFrame from similarity matrix.
    """
    df = pd.DataFrame(
        similarity_matrix,
        index=[f"Query {i+1}" for i in range(len(queries))],
        columns=doc_names
    )
    return df

def color_similarity_values(val):
    """
    Color code similarity values based on their range.
    """
    if val >= 0.7:
        return 'background-color: #d4edda; color: #155724'
    elif val >= 0.4:
        return 'background-color: #fff3cd; color: #856404'
    else:
        return 'background-color: #f8d7da; color: #721c24'

def main():
    # Main header
    st.markdown('<h1 class="main-header">📊 Cosine Similarity Checker</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This application calculates cosine similarity between your queries and uploaded documents using TF-IDF vectorization.
    Upload your documents, enter your queries, and get similarity scores with visual indicators.
    """)
    
    # Sidebar for configuration
    st.sidebar.markdown("### Configuration")
    max_files = st.sidebar.slider("Maximum files to upload", 1, 20, 10)
    similarity_threshold = st.sidebar.slider("Similarity threshold for highlighting", 0.0, 1.0, 0.5, 0.1)
    
    # Create two columns for input
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="section-header">📝 Input Queries</h2>', unsafe_allow_html=True)
        queries_input = st.text_area(
            "Enter your queries (one per line):",
            height=200,
            placeholder="Enter your search queries here...\nOne query per line\nExample: What is machine learning?\nHow does AI work?"
        )
    
    with col2:
        st.markdown('<h2 class="section-header">📁 Upload Documents</h2>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            f"Choose up to {max_files} text files:",
            type=['txt'],
            accept_multiple_files=True,
            help="Upload .txt files containing the documents you want to compare against your queries."
        )
        
        if len(uploaded_files) > max_files:
            st.warning(f"Please upload no more than {max_files} files. Only the first {max_files} will be processed.")
            uploaded_files = uploaded_files[:max_files]
    
    # Process button
    if st.button("🔍 Calculate Similarities", type="primary"):
        if not queries_input.strip():
            st.error("Please enter at least one query.")
            return
        
        if not uploaded_files:
            st.error("Please upload at least one document.")
            return
        
        # Parse queries
        queries = [q.strip() for q in queries_input.strip().split('\n') if q.strip()]
        
        if not queries:
            st.error("Please enter valid queries.")
            return
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Read uploaded files
            status_text.text("Reading uploaded files...")
            documents = []
            doc_names = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                content = read_uploaded_file(uploaded_file)
                if content:
                    documents.append(content)
                    doc_names.append(uploaded_file.name)
                progress_bar.progress((i + 1) / (len(uploaded_files) + 2))
            
            if not documents:
                st.error("Could not read any of the uploaded files. Please check file formats and try again.")
                return
            
            # Calculate similarities
            status_text.text("Calculating similarities...")
            similarity_matrix, vectorizer = calculate_similarity_matrix(queries, documents, doc_names)
            progress_bar.progress((len(uploaded_files) + 1) / (len(uploaded_files) + 2))
            
            if similarity_matrix is None:
                return
            
            # Create results DataFrame
            status_text.text("Formatting results...")
            results_df = create_similarity_dataframe(similarity_matrix, queries, doc_names)
            progress_bar.progress(1.0)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            st.markdown('<h2 class="section-header">📊 Similarity Results</h2>', unsafe_allow_html=True)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Queries", len(queries))
            with col2:
                st.metric("Total Documents", len(documents))
            with col3:
                st.metric("Max Similarity", f"{results_df.values.max():.3f}")
            with col4:
                st.metric("Average Similarity", f"{results_df.values.mean():.3f}")
            
            # Display styled DataFrame
            st.dataframe(
                results_df.style.applymap(color_similarity_values).format("{:.3f}"),
                use_container_width=True
            )
            
            # Interpretation guide
            st.markdown("""
            **Similarity Score Interpretation:**
            - 🟢 **High (≥0.7)**: Very similar content
            - 🟡 **Medium (0.4-0.7)**: Moderately similar content  
            - 🔴 **Low (<0.4)**: Low similarity or different content
            """)
            
            # Top matches for each query
            st.markdown('<h3 class="section-header">🎯 Top Matches by Query</h3>', unsafe_allow_html=True)
            for i, query in enumerate(queries):
                query_results = results_df.iloc[i].sort_values(ascending=False)
                top_match = query_results.index[0]
                top_score = query_results.iloc[0]
                
                st.markdown(f"**Query {i+1}:** *{query[:100]}{'...' if len(query) > 100 else ''}*")
                st.markdown(f"Best match: **{top_match}** (Score: {top_score:.3f})")
                st.markdown("---")
            
            # Download section
            st.markdown('<h3 class="section-header">💾 Download Results</h3>', unsafe_allow_html=True)
            
            # Prepare CSV download
            csv_buffer = StringIO()
            results_df.to_csv(csv_buffer)
            csv_data = csv_buffer.getvalue()
            
            # Download button
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name=f"similarity_results_{int(time.time())}.csv",
                mime="text/csv"
            )
            
            # Additional insights
            if st.checkbox("Show Additional Insights"):
                st.markdown('<h3 class="section-header">🔍 Additional Insights</h3>', unsafe_allow_html=True)
                
                # Feature information
                if vectorizer:
                    feature_names = vectorizer.get_feature_names_out()
                    st.write(f"**Total TF-IDF Features:** {len(feature_names)}")
                    
                    # Show some example features
                    sample_features = np.random.choice(feature_names, min(20, len(feature_names)), replace=False)
                    st.write("**Sample TF-IDF Features:**")
                    st.write(", ".join(sample_features))
                
                # Similarity distribution
                flat_similarities = results_df.values.flatten()
                st.write("**Similarity Score Distribution:**")
                hist_df = pd.DataFrame({'Similarity Scores': flat_similarities})
                st.histogram(hist_df['Similarity Scores'], bins=20)
        
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
            progress_bar.empty()
            status_text.empty()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Note:** This application uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization 
    to convert text into numerical vectors, then calculates cosine similarity between query and document vectors. 
    Higher scores indicate greater similarity in content and vocabulary.
    """)

if __name__ == "__main__":
    main()

