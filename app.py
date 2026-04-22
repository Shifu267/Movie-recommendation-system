# ═══════════════════════════════════════════════════════════════════════
# app.py — FINAL UPGRADED VERSION
# Run AFTER recommender.py has been run once (to generate .pkl files)
# ═══════════════════════════════════════════════════════════════════════

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# ═══════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD SAVED MODEL FILES
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "="*55)
print("       MOVIE RECOMMENDATION SYSTEM")
print("="*55)
print("Loading model files...")

try:
    final_df   = pickle.load(open('output/movies.pkl',     'rb'))
    similarity = pickle.load(open('output/similarity.pkl', 'rb'))
    kmeans     = pickle.load(open('output/kmeans.pkl',     'rb'))
    vectors    = pickle.load(open('output/vectors.pkl',    'rb'))
    print("✅ All model files loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ ERROR: {e}")
    print("Please run recommender.py first to generate model files.")
    exit()

# ═══════════════════════════════════════════════════════════════════════
# STEP 2 — HELPER: GET MOVIE DETAILS
# ═══════════════════════════════════════════════════════════════════════

def get_movie_details(movie_name):
    """
    Returns index, cluster of the given movie.
    Returns None if movie not found.
    """
    if movie_name not in final_df['title'].values:
        return None, None

    idx     = final_df[final_df['title'] == movie_name].index[0]
    cluster = final_df.iloc[idx]['cluster']
    return idx, cluster

# ═══════════════════════════════════════════════════════════════════════
# STEP 3 — MAIN RECOMMENDATION FUNCTION (UPGRADED)
# ═══════════════════════════════════════════════════════════════════════

def recommend(movie_name, top_n=5, same_cluster_only=False):
    """
    Recommends top N similar movies using:
    - Cosine Similarity (main engine)
    - K-Means Cluster info (bonus info)
    - Optional: filter recommendations to same cluster only

    Parameters:
        movie_name       : name of input movie (string)
        top_n            : how many recommendations to show (default 5)
        same_cluster_only: if True, only recommend from same cluster
    """

    # ── Check if movie exists ─────────────────────────────────────
    idx, cluster = get_movie_details(movie_name)

    if idx is None:
        print(f"\n❌ '{movie_name}' not found in dataset.")
        print("💡 Tip: Check spelling or try a slightly different name.")

        # Suggest similar names
        all_titles  = final_df['title'].str.lower().tolist()
        input_lower = movie_name.lower()
        suggestions = [t for t in all_titles if input_lower[:4] in t][:5]

        if suggestions:
            print("\n🔍 Did you mean:")
            for s in suggestions:
                # Show original casing
                original = final_df[
                    final_df['title'].str.lower() == s
                ]['title'].values[0]
                print(f"   → {original}")
        return

    # ── Get similarity scores ──────────────────────────────────────
    distances     = list(enumerate(similarity[idx]))
    sorted_movies = sorted(distances, key=lambda x: x[1], reverse=True)

    # ── Filter by same cluster if requested ───────────────────────
    if same_cluster_only:
        sorted_movies = [
            (i, score) for i, score in sorted_movies
            if final_df.iloc[i]['cluster'] == cluster
        ]

    # ── Print results ─────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Input Movie : {movie_name}")
    print(f"  Cluster     : {cluster}  (K-Means Group)")
    print(f"  Mode        : {'Same Cluster Only' if same_cluster_only else 'All Movies'}")
    print(f"{'='*55}")
    print(f"\n  Top {top_n} Similar Movies:\n")
    print(f"  {'#':<4} {'Title':<35} {'Score':<8} {'Cluster'}")
    print("  " + "-"*55)

    shown = 0
    for index, score in sorted_movies[1:]:         # skip index 0 (itself)
        if shown >= top_n:
            break
        title       = final_df.iloc[index]['title']
        rec_cluster = final_df.iloc[index]['cluster']
        tag         = "✅ Same" if rec_cluster == cluster else "🔄 Diff"
        print(f"  {shown+1:<4} {title:<35} {round(score,4):<8} {tag} ({rec_cluster})")
        shown += 1

    print("  " + "-"*55)

# ═══════════════════════════════════════════════════════════════════════
# STEP 4 — COMPARE TWO MOVIES
# ═══════════════════════════════════════════════════════════════════════

def compare_movies(movie1, movie2):
    """
    Compares two movies directly:
    - Shows their similarity score
    - Shows their clusters
    - Shows common tags/keywords
    """
    idx1, cluster1 = get_movie_details(movie1)
    idx2, cluster2 = get_movie_details(movie2)

    if idx1 is None:
        print(f"❌ '{movie1}' not found.")
        return
    if idx2 is None:
        print(f"❌ '{movie2}' not found.")
        return

    # Similarity score between the two movies
    score = round(similarity[idx1][idx2], 4)

    # Find common words in tags
    tags1 = set(final_df.iloc[idx1]['tags'].split())
    tags2 = set(final_df.iloc[idx2]['tags'].split())
    common_tags = tags1 & tags2

    # Remove very common stop words from display
    stop = {'the','a','an','in','of','and','to','is','it','on','with'}
    common_tags = [t for t in common_tags if t not in stop]

    print(f"\n{'='*55}")
    print(f"  MOVIE COMPARISON")
    print(f"{'='*55}")
    print(f"  Movie 1 : {movie1}  (Cluster {cluster1})")
    print(f"  Movie 2 : {movie2}  (Cluster {cluster2})")
    print(f"  Similarity Score : {score}")

    if score >= 0.5:
        print(f"  Verdict : 🟢 Very Similar")
    elif score >= 0.2:
        print(f"  Verdict : 🟡 Somewhat Similar")
    else:
        print(f"  Verdict : 🔴 Not Very Similar")

    if cluster1 == cluster2:
        print(f"  Cluster Match : ✅ Same cluster group")
    else:
        print(f"  Cluster Match : ❌ Different cluster groups")

    print(f"\n  Common Tags ({len(common_tags)} found):")
    print(f"  {', '.join(sorted(common_tags)[:15])}")
    print("="*55)

# ═══════════════════════════════════════════════════════════════════════
# STEP 5 — CLUSTER EXPLORER
# ═══════════════════════════════════════════════════════════════════════

def explore_cluster(movie_name, show_n=10):
    """
    Shows all movies in the same cluster as the input movie.
    Useful to understand what group a movie belongs to.
    """
    idx, cluster = get_movie_details(movie_name)

    if idx is None:
        print(f"❌ '{movie_name}' not found.")
        return

    cluster_movies = final_df[
        final_df['cluster'] == cluster
    ]['title'].values

    print(f"\n{'='*55}")
    print(f"  Cluster {cluster} — Movies in same group as '{movie_name}'")
    print(f"  Total movies in this cluster: {len(cluster_movies)}")
    print(f"{'='*55}")

    for i, title in enumerate(cluster_movies[:show_n], 1):
        marker = "👉" if title == movie_name else "  "
        print(f"  {marker} {i}. {title}")

    if len(cluster_movies) > show_n:
        print(f"\n  ... and {len(cluster_movies) - show_n} more movies in this cluster.")

# ═══════════════════════════════════════════════════════════════════════
# STEP 6 — VISUALIZE RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════

def visualize_recommendations(movie_name, top_n=5):
    """
    Creates a bar chart of top N recommended movies
    with their similarity scores.
    """
    idx, cluster = get_movie_details(movie_name)
    if idx is None:
        print(f"❌ '{movie_name}' not found.")
        return

    distances     = list(enumerate(similarity[idx]))
    sorted_movies = sorted(distances, key=lambda x: x[1], reverse=True)

    titles = []
    scores = []
    colors_list = []

    for index, score in sorted_movies[1:top_n+1]:
        title       = final_df.iloc[index]['title']
        rec_cluster = final_df.iloc[index]['cluster']
        titles.append(title)
        scores.append(round(score, 4))
        # Green if same cluster, orange if different
        colors_list.append('#2ecc71' if rec_cluster == cluster else '#e67e22')

    plt.figure(figsize=(10, 5))
    bars = plt.barh(titles[::-1], scores[::-1],
                    color=colors_list[::-1], edgecolor='black')

    # Add score labels on bars
    for bar, score in zip(bars, scores[::-1]):
        plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                 str(score), va='center', fontsize=9)

    plt.xlabel('Cosine Similarity Score')
    plt.title(f'Top {top_n} Movies Similar to "{movie_name}"\n'
              f'🟢 Same Cluster   🟠 Different Cluster',
              fontsize=12, fontweight='bold')
    plt.xlim(0, max(scores) + 0.05)
    plt.grid(True, linestyle='--', alpha=0.4, axis='x')
    plt.tight_layout()

    filename = f'output/rec_{movie_name.replace(" ","_")}.png'
    plt.savefig(filename)
    plt.show()
    print(f"✅ Chart saved → {filename}")

# ═══════════════════════════════════════════════════════════════════════
# STEP 7 — INTERACTIVE MENU
# ═══════════════════════════════════════════════════════════════════════

def show_menu():
    print("\n" + "="*55)
    print("  WHAT WOULD YOU LIKE TO DO?")
    print("="*55)
    print("  1. Get movie recommendations")
    print("  2. Get recommendations (same cluster only)")
    print("  3. Compare two movies")
    print("  4. Explore movies in a cluster")
    print("  5. Visualize recommendations as chart")
    print("  6. Exit")
    print("="*55)

def run_app():
    """
    Main interactive loop — runs the full app.
    """
    print("\n✅ System ready!")
    print(f"📦 Dataset: {len(final_df)} movies loaded")

    while True:
        show_menu()
        choice = input("\n  Enter choice (1-6): ").strip()

        # ── Option 1: Normal recommendations ──────────────────────
        if choice == '1':
            movie = input("\n  Enter movie name: ").strip()
            n     = input("  How many recommendations? (default=5): ").strip()
            n     = int(n) if n.isdigit() else 5
            recommend(movie, top_n=n)

        # ── Option 2: Same cluster recommendations ────────────────
        elif choice == '2':
            movie = input("\n  Enter movie name: ").strip()
            recommend(movie, top_n=5, same_cluster_only=True)

        # ── Option 3: Compare two movies ──────────────────────────
        elif choice == '3':
            movie1 = input("\n  Enter first movie name : ").strip()
            movie2 = input("  Enter second movie name: ").strip()
            compare_movies(movie1, movie2)

        # ── Option 4: Explore cluster ─────────────────────────────
        elif choice == '4':
            movie = input("\n  Enter movie name: ").strip()
            n     = input("  How many cluster movies to show? (default=10): ").strip()
            n     = int(n) if n.isdigit() else 10
            explore_cluster(movie, show_n=n)

        # ── Option 5: Visualize ───────────────────────────────────
        elif choice == '5':
            movie = input("\n  Enter movie name: ").strip()
            n     = input("  How many recommendations to chart? (default=5): ").strip()
            n     = int(n) if n.isdigit() else 5
            visualize_recommendations(movie, top_n=n)

        # ── Option 6: Exit ────────────────────────────────────────
        elif choice == '6':
            print("\n👋 Thank you for using the Movie Recommendation System!")
            print("   Project by: [Your Name] | BTech Capstone 2024")
            break

        else:
            print("\n⚠️  Invalid choice. Please enter a number between 1-6.")

# ═══════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_app()