import streamlit as st
import pandas as pd
import numpy as np
import ast
import requests
import io
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CineML Recommender", page_icon="🎬", layout="wide")

TMDB_ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI1YmY0Yzk3YWYwNDBhZWIxMmQ3OWUzMjc1Nzc3N2Y4NyIsIm5iZiI6MTc3NTAyNzA5OC4zMDQsInN1YiI6IjY5Y2NjMzlhNTdkYTc3ZDhmYjI3ZjEwMSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.dxgzeF72RqMvoNNqRAtwe4J6plhch18zq6oqR2XtHXo"  # <-- paste your key here
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
FALLBACK_POSTER = "https://placehold.co/500x750/1f1f1f/ffffff/png?text=Poster%0ANot%0AAvailable"

# --- DATA LOADING & PREPROCESSING ---
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_data.csv')
    
    # Safely parse stringified lists/dictionaries
    def safe_parse(val):
        try:
            return ast.literal_eval(val)
        except:
            return []
            
    df['genres'] = df['genres'].apply(safe_parse)
    df['directors'] = df['directors'].apply(safe_parse)
    df['cast'] = df['cast'].apply(safe_parse)
    df['reviews_list'] = df['reviews'].apply(safe_parse)
    
    # Clean rating column to extract float
    df['rating_val'] = df['rating'].str.extract(r'([\d\.]+)').astype(float)
    
    # Create a combined features column for Content-Based Filtering
    df['combined_features'] = df['synopsis'].fillna('') + " " + \
                              df['genres'].apply(lambda x: ' '.join(x)) + " " + \
                              df['directors'].apply(lambda x: ' '.join(x))
    return df

@st.cache_data
def get_user_reviews(df):
    all_reviews = []
    for idx, row in df.iterrows():
        for rev in row['reviews_list']:
            if isinstance(rev, dict) and 'username' in rev:
                all_reviews.append({
                    'movie': row['title'],
                    'username': rev['username'],
                    'likes': rev.get('likes', 0),
                    'review_text': rev.get('review_text', '')
                })
    return pd.DataFrame(all_reviews)

# Load data
df = load_data()
reviews_df = get_user_reviews(df)

# --- ML MODELS ---
@st.cache_data
def compute_cosine_sim(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim = compute_cosine_sim(df)

@st.cache_data(show_spinner=False, ttl=86400) # Caches the images for 24 hours so your app stays fast
def get_tmdb_poster(title: str,year=None) -> str:
    """
    Downloads the image on the backend masking as a real web browser.
    Returns the actual Image object or a reliable fallback URL.
    """
    try:
        params = {
            "api_key": TMDB_API_KEY,
            "query": title,
            "include_adult": False,
        }
        if year:
            params["year"] = int(year)

        resp = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params=params,
            timeout=5
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])

        if results and results[0].get("poster_path"):
            return TMDB_IMAGE_BASE + results[0]["poster_path"]

    except Exception:
        pass  # Any network/parse error → fall through to placeholder

    return FALLBACK_POSTER

def get_content_recommendations(title, cosine_sim=cosine_sim, df=df, top_n=5):
    if title not in df['title'].values:
        return pd.DataFrame()
    idx = df.index[df['title'] == title][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
   
    movie_indices = []
    seen_titles = set([title]) # Add the selected movie to 'seen' so it doesn't recommend itself
    
    # Loop through scores and only add unique titles
    for i, score in sim_scores:
        movie_title = df.iloc[i]['title']
        if movie_title not in seen_titles:
            seen_titles.add(movie_title)
            movie_indices.append(i)
        
        # Stop once we have exactly top_n unique movies
        if len(movie_indices) == top_n:
            break
            
    return df.iloc[movie_indices]

def get_collaborative_recommendations(username, reviews_df, top_n=5):
    # Find movies the user has reviewed
    user_movies = reviews_df[reviews_df['username'] == username]['movie'].tolist()
    if not user_movies:
        return []
    
    # Find other users who reviewed the same movies
    similar_users = reviews_df[reviews_df['movie'].isin(user_movies)]['username'].value_counts().index.tolist()
    
    # Get movies reviewed by similar users but not the target user
    recs = reviews_df[(reviews_df['username'].isin(similar_users)) & 
                      (~reviews_df['movie'].isin(user_movies))]
    
    # Rank by how frequently these movies were reviewed by similar users
    top_movies = recs['movie'].value_counts().head(top_n * 3).index.tolist()
    
# Filter the main dataframe, DROP DUPLICATES, and return exactly top_n
    final_recs = df[df['title'].isin(top_movies)].drop_duplicates(subset=['title'])
    
    return final_recs.head(top_n)

    
@st.dialog("🎬 Movie Details", width="large")

def show_movie_details(movie_row):
    # Create a two-column layout for the pop-up (Poster on left, details on right)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Pass the URL through our new backend fetcher
        st.image(get_tmdb_poster(row['title'], row['year'])))
        
    with col2:
        st.header(f"{movie_row['title']} ({movie_row['year']})")
        st.write(f"**⭐ Rating:** {movie_row['rating_val']} / 5")
        
        # Format lists neatly
        directors = ", ".join(movie_row['directors']) if isinstance(movie_row['directors'], list) else "Unknown"
        genres = ", ".join(movie_row['genres']) if isinstance(movie_row['genres'], list) else "Unknown"
        
        # Show only the top 5 cast members so it doesn't clutter the screen
        cast_list = movie_row['cast'][:5] if isinstance(movie_row['cast'], list) else []
        cast_str = ", ".join(cast_list) + ("..." if len(movie_row['cast']) > 5 else "")
        
        st.write(f"**🎬 Director:** {directors}")
        st.write(f"**🎭 Genres:** {genres}")
        st.write(f"**👥 Cast:** {cast_str}")
        
    st.markdown("---")
    st.subheader("Synopsis")
    st.write(movie_row['synopsis'])

# --- UI LAYOUT ---
st.title("🎬 CineML: AI-Powered Movie Analytics & Recommender")
st.markdown("Explore movie statistics, get personalized recommendations, and analyze review sentiments using Machine Learning.")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🤖 Recommender Engine", "🧠 Sentiment Analysis", "🗄️ Raw Data"])

# --- TAB 1: DASHBOARD ---
with tab1:
    st.header("Movie Dataset Statistics")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Movies", len(df))
    col2.metric("Total Reviews Extracted", len(reviews_df))
    col3.metric("Unique Users", reviews_df['username'].nunique())
    
    st.markdown("---")

# --- ROW 1: Ratings & Years ---
    st.subheader("Distribution of Movie Ratings")
    st.markdown("*(Click on any bar to see all movies with that specific rating!)*")
    
    # 1. Prepare data: Count exact number of movies per rating
    rating_counts = df['rating_val'].value_counts().reset_index()
    rating_counts.columns = ['Rating', 'Number of Movies']
    rating_counts = rating_counts.sort_values('Rating', ascending=False)
    
    # 2. Create a discrete bar chart
    fig_ratings = px.bar(
        rating_counts, 
        x='Rating', 
        y='Number of Movies', 
        
        color='Number of Movies',
        color_continuous_scale='Blues'
    )
    # Force the X-axis to treat ratings as distinct categories, not a continuous timeline
    fig_ratings.update_xaxes(type='category') 
    fig_ratings.update_traces(textposition='outside')
    fig_ratings.update_layout(margin=dict(t=20, b=20))
    
    # 3. Render chart with click-selection enabled (Requires Streamlit 1.35+)
    # This captures the user's click event on the chart
    selection = st.plotly_chart(
        fig_ratings, 
        use_container_width=True, 
        on_select="rerun", 
        key="rating_dist_chart"
    )
    
    # 4. Handle the click event logic
    selected_rating = None
    
    # Check if the user clicked a bar in the Plotly chart
    if selection and 'selection' in selection and selection['selection']['points']:
        # Extract the X value (the rating) from the clicked bar
        selected_rating = float(selection['selection']['points'][0]['x'])
    
    # Fallback/Accessibility: A dropdown just in case the user prefers not to click the chart
    fallback_rating = st.selectbox(
        "Or select a rating manually:", 
        options=['-- Select a Rating --'] + rating_counts['Rating'].tolist(),
        index=0
    )
    
    # Determine which rating to show (prioritize chart click, then dropdown)
    final_rating = selected_rating if selected_rating else (fallback_rating if fallback_rating != '-- Select a Rating --' else None)
    
    # 5. Display the matching movies below the chart
    if final_rating:
        st.markdown("---")
        st.subheader(f"🍿 Movies Rated {final_rating}")
        
        # Filter dataframe for the selected rating
        matched_movies = df[df['rating_val'] == final_rating]
        
        st.write(f"Found **{len(matched_movies)}** movies:")
        
        # Display each movie in an expander for a clean UI
        for _, row in matched_movies.iterrows():
            with st.expander(f"🎬 {row['title']} ({row['year']})"):
                det_col1, det_col2 = st.columns([1, 4])
                
                with det_col1:
                    st.image(get_tmdb_poster(movie_row['title'], movie_row['year']))
                    
                with det_col2:
                    directors = ", ".join(row['directors']) if isinstance(row['directors'], list) else "Unknown"
                    genres = ", ".join(row['genres']) if isinstance(row['genres'], list) else "Unknown"
                    
                    st.write(f"**Director:** {directors}")
                    st.write(f"**Genres:** {genres}")
                    st.write(f"**Synopsis:** {row['synopsis']}")
                    
    st.markdown("---")
    # --- ROW 2: Genres & Directors ---
    r2_col1, r2_col2 = st.columns(2)
    
    with r2_col1:
        st.subheader("Most Popular Genres")
        # Flatten the genres list
        all_genres = [genre for sublist in df['genres'] for genre in sublist]
        genre_df = pd.DataFrame(all_genres, columns=['Genre']).value_counts().reset_index(name='Count').head(10)
        fig_genres = px.bar(genre_df, x='Genre', y='Count', color='Count', color_continuous_scale='Plasma')
        st.plotly_chart(fig_genres, use_container_width=True)

    with r2_col2:
        st.subheader("Top Directors (By Movie Count)")
        all_dirs = [director for sublist in df['directors'] for director in sublist]
        dir_df = pd.DataFrame(all_dirs, columns=['Director']).value_counts().reset_index(name='Count').head(10)
        fig_dirs = px.bar(dir_df, x='Count', y='Director', orientation='h', color='Count', color_continuous_scale='Magma')
        fig_dirs.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_dirs, use_container_width=True)

    st.markdown("---")
    
    # --- ROW 3: Top Cast ---
    st.subheader("Most Frequently Cast Actors")
    all_cast = [actor for sublist in df['cast'] for actor in sublist]
    cast_df = pd.DataFrame(all_cast, columns=['Actor']).value_counts().reset_index(name='Count').head(15)
    fig_cast = px.bar(cast_df, x='Actor', y='Count', color='Count', color_continuous_scale='Cividis')
    st.plotly_chart(fig_cast, use_container_width=True)

# --- TAB 2: RECOMMENDER ENGINE ---
with tab2:
    st.header("🔍 Search & 🤖 Recommendations")
    
    # Create sub-tabs within Tab 2 for a cleaner UI
    search_tab, rec_tab = st.tabs(["Global Search", "AI Recommender"])

    # --- 1. SEARCH MECHANISM ---
    with search_tab:
        st.subheader("Filter Movies by Attributes")
        s_col1, s_col2 = st.columns(2)
        
        with s_col1:
            search_name = st.text_input("Search by Movie Name", placeholder="e.g. Inception")
            
            # Extract unique genres for the dropdown
            all_genres_list = sorted(list(set([g for sub in df['genres'] for g in sub])))
            search_genre = st.multiselect("Filter by Genre", all_genres_list)

        with s_col2:
            # Extract unique directors
            all_dirs_list = sorted(list(set([d for sub in df['directors'] for d in sub])))
            search_dir = st.selectbox("Search by Director", ["All"] + all_dirs_list)
            
            search_cast = st.text_input("Search by Cast Member", placeholder="e.g. Leonardo DiCaprio")

        # Search Logic
        filtered_df = df.copy()
        if search_name:
            filtered_df = filtered_df[filtered_df['title'].str.contains(search_name, case=False, na=False)]
        if search_genre:
            filtered_df = filtered_df[filtered_df['genres'].apply(lambda x: any(g in x for g in search_genre))]
        if search_dir != "All":
            filtered_df = filtered_df[filtered_df['directors'].apply(lambda x: search_dir in x)]
        if search_cast:
            filtered_df = filtered_df[filtered_df['cast'].apply(lambda x: any(search_cast.lower() in actor.lower() for actor in x))]

        st.markdown(f"Found **{len(filtered_df)}** movies matching your criteria.")
        
        # Display Results in a Grid
        if not filtered_df.empty:
            # Show top 12 results to keep it snappy
            results = filtered_df.head(12)
            grid_cols = st.columns(4)
            for idx, (_, row) in enumerate(results.iterrows()):
                with grid_cols[idx % 4]:
                    st.image(get_tmdb_poster(row['title'], row['year']))
                    if st.button(f"ℹ️ {row['title']}", key=f"search_res_{idx}"):
                        st.session_state.selected_movie_details = row
                        st.rerun()

    # --- 2. AI RECOMMENDER (Original Logic) ---
    with rec_tab:
        rec_type = st.radio("Choose Logic:", ["Content-Based", "Collaborative Filtering"], horizontal=True)
        
        if rec_type == "Content-Based":
            selected_movie = st.selectbox("Because you liked:", df['title'].unique(), key="rec_select")
            if st.button("Generate Recommendations"):
                st.session_state.current_recs = get_content_recommendations(selected_movie)
        else:
            top_users = reviews_df['username'].value_counts().head(50).index.tolist()
            selected_user = st.selectbox("Select User Profile:", top_users)
            if st.button("Generate for User"):
                st.session_state.user_recs = get_collaborative_recommendations(selected_user, reviews_df)

        # Rendering logic for recommendations (using the proxy)
        active_recs = st.session_state.get('current_recs' if rec_type == "Content-Based" else 'user_recs', pd.DataFrame())
        
        if not active_recs.empty:
            st.markdown("---")
            cols = st.columns(len(active_recs))
            for i, (_, row) in enumerate(active_recs.iterrows()):
                with cols[i]:
                    st.image(get_tmdb_poster(row['title'], row['year']))
                    if st.button(f"ℹ️ {row['title']}", key=f"rec_btn_{i}"):
                        st.session_state.selected_movie_details = row
                        st.rerun()

    # --- COMMON DETAIL VIEW (The Pop-up/Expander) ---
    if st.session_state.get('selected_movie_details') is not None:
        movie = st.session_state.selected_movie_details
        with st.expander(f"🎬 Details: {movie['title']}", expanded=True):
            det_col1, det_col2 = st.columns([1, 3])
            with det_col1:
                st.image(get_tmdb_poster(movie['title'], movie['year']))
            with det_col2:
                st.header(f"{movie['title']} ({movie['year']})")
                st.write(f"**Rating:** ⭐ {movie['rating_val']}")
                st.write(f"**Directors:** {', '.join(movie['directors'])}")
                st.write(f"**Genres:** {', '.join(movie['genres'])}")
                st.write(f"**Cast:** {', '.join(movie['cast'][:5])}...")
                st.info(f"**Synopsis:** {movie['synopsis']}")
                if st.button("Close Details"):
                    st.session_state.selected_movie_details = None
                    st.rerun()
# --- TAB 3: SENTIMENT ANALYSIS ---
with tab3:
    st.header("🧠 NLP Sentiment Analysis on Reviews")
    st.markdown("Analyze how the audience actually felt about a movie based on their written reviews.")
    
    # Only show movies that actually have text reviews
    movies_with_reviews = reviews_df[reviews_df['review_text'].str.strip() != '']['movie'].unique()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        movie_for_sa = st.selectbox("Select a movie to analyze:", movies_with_reviews, key="sa_movie")
        analyze_btn = st.button("Analyze Audience Mood", use_container_width=True)
    
    if analyze_btn or 'sa_results' in st.session_state:
        movie_reviews = reviews_df[(reviews_df['movie'] == movie_for_sa) & (reviews_df['review_text'].str.strip() != '')]
        
        if movie_reviews.empty:
            st.warning("No detailed text reviews available for this movie.")
        else:
            with st.spinner("Running NLP sentiment models..."):
                sentiments = []
                polarities = []
                
                for text in movie_reviews['review_text']:
                    blob = TextBlob(text)
                    polarity = blob.sentiment.polarity
                    polarities.append(polarity)
                    
                    if polarity > 0.1:
                        sentiments.append("Positive")
                    elif polarity < -0.1:
                        sentiments.append("Negative")
                    else:
                        sentiments.append("Neutral")
                
                results_df = pd.DataFrame({
                    'Username': movie_reviews['username'].values,
                    'Review': movie_reviews['review_text'].values,
                    'Sentiment': sentiments, 
                    'Polarity': polarities
                })
                
                st.markdown("---")
                # Top Level Metrics
                avg_polarity = np.mean(polarities)
                overall_mood = "Positive 😊" if avg_polarity > 0.1 else "Negative 😠" if avg_polarity < -0.1 else "Mixed / Neutral 😐"
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Overall Audience Mood", overall_mood)
                m2.metric("Average Polarity Score", f"{avg_polarity:.2f} / 1.0")
                m3.metric("Total Reviews Analyzed", len(results_df))
                
                # Charts
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    fig_pie = px.pie(results_df, names='Sentiment', title="Sentiment Distribution", 
                                     color='Sentiment', color_discrete_map={'Positive':'#2ecc71', 'Neutral':'#95a5a6', 'Negative':'#e74c3c'},
                                     hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                with chart_col2:
                    fig_hist = px.histogram(results_df, x='Polarity', nbins=15, title="Polarity Score Variance",
                                            labels={'Polarity': 'Score (-1.0 to 1.0)'}, color_discrete_sequence=['#3498db'])
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                # Interactive Data Table
                st.subheader("📝 Processed Reviews & Scores")
                st.dataframe(
                    results_df.style.applymap(
                        lambda x: 'color: #2ecc71;' if x == 'Positive' else ('color: #e74c3c;' if x == 'Negative' else 'color: #95a5a6;'),
                        subset=['Sentiment']
                    ),
                    use_container_width=True,
                    hide_index=True
                )

# --- TAB 4: RAW DATA ---
with tab4:
    st.header("Dataset Explorer")
    st.dataframe(df[['title', 'year', 'rating', 'genres', 'directors']].head(100), use_container_width=True)