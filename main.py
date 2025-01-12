import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Upload dataset
@st.cache
def load_data():
    df = pd.read_csv("game_list_dataset.csv")  # Pastikan file ini sudah ada di repository Anda
    return df

# Fungsi untuk membangun model Collaborative Filtering
@st.cache
def train_model(df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['userId', 'itemId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)
    
    algo = SVD()
    algo.fit(trainset)
    
    # Evaluasi model
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    return algo, rmse

# Fungsi untuk mendapatkan rekomendasi
def recommend(algo, user_id, df):
    all_games = df['itemId'].unique()
    rated_games = df[df['userId'] == user_id]['itemId'].tolist()
    unrated_games = [game for game in all_games if game not in rated_games]

    recommendations = []
    for game in unrated_games:
        pred = algo.predict(uid=user_id, iid=game)
        recommendations.append((game, pred.est))
    
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    return recommendations[:10]

# Streamlit UI
st.title("Sistem Rekomendasi Game")
st.write("Aplikasi ini memberikan rekomendasi game berdasarkan Collaborative Filtering.")

# Load data
df = load_data()

# Sidebar untuk memilih user
user_id = st.sidebar.number_input("Masukkan User ID:", min_value=1, value=1, step=1)

# Tampilkan dataset
st.subheader("Dataset")
st.dataframe(df)

# Training model
algo, rmse = train_model(df)
st.write(f"Model dilatih dengan RMSE: {rmse:.2f}")

# Tampilkan rekomendasi
if st.button("Rekomendasi Game"):
    recommendations = recommend(algo, user_id, df)
    st.subheader("Rekomendasi untuk Anda:")
    for i, (game, rating) in enumerate(recommendations, start=1):
        st.write(f"{i}. {game} (Predicted Rating: {rating:.2f})")
