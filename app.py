import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# =====================================================
# STREAMLIT CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Smart Tourist Guide Indonesia",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# DATA LOADING
# =====================================================
@st.cache_data

def load_and_preprocess_datasets():
    try:
        df_places = pd.read_csv(
            "data/indonesia tourism destination/tourism_with_id.csv"
        )

        df_ratings = pd.read_csv(
            "data/indonesia tourism destination/tourism_rating.csv"
        )

    except FileNotFoundError:
        st.error("Dataset file not found. Verify dataset path configuration.")
        st.stop()

    # Handle missing values
    df_places = df_places.fillna("")

    # Standardize coordinate column names for st.map()
    if 'Lat' in df_places.columns:
        df_places.rename(columns={'Lat': 'lat'}, inplace=True)

    if 'Long' in df_places.columns:
        df_places.rename(columns={'Long': 'lon'}, inplace=True)

    # Create combined text features for TF-IDF
    text_features = [
        'Place_Name',
        'Description',
        'Category',
        'City'
    ]

    for column in text_features:
        if column not in df_places.columns:
            df_places[column] = ""

    df_places['content'] = (
        df_places['Place_Name'].astype(str) + ' ' +
        df_places['Description'].astype(str) + ' ' +
        df_places['Category'].astype(str) + ' ' +
        df_places['City'].astype(str)
    )

    # Clean text content
    df_places['content'] = df_places['content'].str.lower()

    return df_places, df_ratings

# =====================================================
# CONTENT-BASED FILTERING MODEL
# =====================================================
@st.cache_resource

def train_content_based_model(df_places):

    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=5000
    )

    tfidf_matrix = tfidf.fit_transform(df_places['content'])

    cosine_sim_matrix = cosine_similarity(tfidf_matrix)

    return cosine_sim_matrix

# =====================================================
# COLLABORATIVE FILTERING MODEL
# =====================================================
@st.cache_resource

def train_collaborative_model(df_ratings):

    utility_matrix = df_ratings.pivot_table(
        index='User_Id',
        columns='Place_Id',
        values='Place_Ratings'
    ).fillna(0)

    max_components = min(20, utility_matrix.shape[1] - 1)

    if max_components < 2:
        max_components = 2

    svd = TruncatedSVD(
        n_components=max_components,
        random_state=42
    )

    svd.fit_transform(utility_matrix)

    item_factors = svd.components_.T

    cf_similarity_matrix = cosine_similarity(item_factors)

    mapped_place_ids = utility_matrix.columns.tolist()

    return utility_matrix, cf_similarity_matrix, mapped_place_ids

# =====================================================
# CONTENT-BASED RECOMMENDATION ENGINE
# =====================================================
def generate_cbf_recommendation(
    target_place,
    df_places,
    similarity_matrix,
    k=5
):

    if target_place not in df_places['Place_Name'].values:
        return pd.DataFrame()

    idx = df_places[df_places['Place_Name'] == target_place].index[0]

    sim_scores = list(enumerate(similarity_matrix[idx]))

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:k + 1]

    item_indices = [i[0] for i in sim_scores]
    similarity_scores = [i[1] for i in sim_scores]

    recommendations = df_places.iloc[item_indices].copy()

    recommendations['Similarity_Score'] = similarity_scores

    return recommendations

# =====================================================
# COLLABORATIVE FILTERING RECOMMENDATION ENGINE
# =====================================================
def generate_cf_recommendation(
    user_id,
    utility_matrix,
    cf_matrix,
    place_ids,
    df_places,
    k=5
):

    if user_id not in utility_matrix.index:
        return pd.DataFrame()

    user_history = utility_matrix.loc[user_id]

    visited_idx = np.where(user_history > 0)[0]

    if len(visited_idx) == 0:
        return pd.DataFrame()

    predictive_scores = np.zeros(len(place_ids))

    for idx in visited_idx:
        actual_rating = user_history.iloc[idx]
        predictive_scores += cf_matrix[idx] * actual_rating

    predictive_scores[visited_idx] = 0

    top_indices = predictive_scores.argsort()[-k:][::-1]

    recommended_ids = [place_ids[i] for i in top_indices]

    recommendations = df_places[
        df_places['Place_Id'].isin(recommended_ids)
    ].copy()

    recommendations['Predicted_Score'] = recommendations['Place_Id'].map(
        {
            place_ids[i]: predictive_scores[i]
            for i in top_indices
        }
    )

    recommendations = recommendations.sort_values(
        by='Predicted_Score',
        ascending=False
    )

    return recommendations

# =====================================================
# STREAMLIT APPLICATION
# =====================================================
def main():

    df_places, df_ratings = load_and_preprocess_datasets()

    cbf_sim_matrix = train_content_based_model(df_places)

    utility_matrix, cf_sim_matrix, place_ids = (
        train_collaborative_model(df_ratings)
    )

    st.title("🗺️ Smart Tourist Guide Indonesia")

    st.markdown(
        "Hybrid Recommendation System for personalized tourism destination discovery."
    )

    # Sidebar
    st.sidebar.header("System Configuration")

    algorithm_mode = st.sidebar.radio(
        "Recommendation Method",
        [
            "Content-Based Filtering",
            "Collaborative Filtering"
        ]
    )

    k_slider = st.sidebar.slider(
        "Top-K Recommendation",
        min_value=3,
        max_value=12,
        value=6
    )

    st.markdown("---")

    # =====================================================
    # CONTENT-BASED FILTERING PAGE
    # =====================================================
    if algorithm_mode == "Content-Based Filtering":

        st.subheader("Destination Similarity Recommendation")

        st.write(
            "Recommendation generated from semantic similarity using TF-IDF vectorization and cosine similarity."
        )

        place_dictionary = sorted(
            df_places['Place_Name'].unique().tolist()
        )

        user_query = st.selectbox(
            "Select Reference Destination",
            place_dictionary
        )

        if st.button("Generate Recommendation"):

            with st.spinner('Processing recommendation model...'):

                results = generate_cbf_recommendation(
                    user_query,
                    df_places,
                    cbf_sim_matrix,
                    k=k_slider
                )

            if not results.empty:

                st.success(
                    f"Top recommendation results related to: {user_query}"
                )

                columns_layout = st.columns(3)

                for index, (_, row) in enumerate(results.iterrows()):

                    with columns_layout[index % 3]:

                        st.info(f"**{row['Place_Name']}**")

                        st.caption(
                            f"{row['Category']} | {row['City']}"
                        )

                        st.write(
                            f"Similarity Score: {row['Similarity_Score']:.4f}"
                        )

                        if 'Price' in row:
                            st.write(f"Ticket Price: Rp {row['Price']:,}")

                # Interactive Map
                if 'lat' in results.columns and 'lon' in results.columns:

                    st.subheader("Interactive Destination Map")

                    map_dataset = pd.concat([
                        df_places[
                            df_places['Place_Name'] == user_query
                        ][['lat', 'lon']],
                        results[['lat', 'lon']]
                    ])

                    st.map(map_dataset)

            else:
                st.warning("No recommendation result found.")

    # =====================================================
    # COLLABORATIVE FILTERING PAGE
    # =====================================================
    else:

        st.subheader("User Preference Recommendation")

        st.write(
            "Recommendation generated from historical user interaction patterns using collaborative filtering."
        )

        user_dictionary = sorted(
            df_ratings['User_Id'].unique().tolist()
        )

        user_id_query = st.selectbox(
            "Select User ID",
            user_dictionary
        )

        if st.button("Run Recommendation Engine"):

            with st.spinner('Analyzing user preference pattern...'):

                results = generate_cf_recommendation(
                    user_id_query,
                    utility_matrix,
                    cf_sim_matrix,
                    place_ids,
                    df_places,
                    k=k_slider
                )

            if not results.empty:

                st.success(
                    f"Recommendation generated for User ID: {user_id_query}"
                )

                columns_layout = st.columns(3)

                for index, (_, row) in enumerate(results.iterrows()):

                    with columns_layout[index % 3]:

                        st.success(f"**{row['Place_Name']}**")

                        st.caption(
                            f"{row['Category']} | {row['City']}"
                        )

                        st.write(
                            f"Predicted Score: {row['Predicted_Score']:.4f}"
                        )

                        if 'Price' in row:
                            st.write(f"Ticket Price: Rp {row['Price']:,}")

                # Interactive Map
                if 'lat' in results.columns and 'lon' in results.columns:

                    st.subheader("Interactive Destination Map")

                    st.map(results[['lat', 'lon']])

            else:
                st.warning(
                    "User history is insufficient for collaborative filtering recommendation."
                )

# =====================================================
# APPLICATION ENTRY POINT
# =====================================================
if __name__ == '__main__':
    main()


