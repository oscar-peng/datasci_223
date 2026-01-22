"""Generate pedagogical visualizations for NLP lecture."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Sample clinical documents
docs = [
    "patient reports chest pain",
    "patient denies chest pain",
    "patient reports headache"
]

# 1. Bag of Words Heatmap
def generate_bow_heatmap():
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()

    fig, ax = plt.subplots(figsize=(10, 4))

    # Create heatmap
    sns.heatmap(
        X.toarray(),
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=feature_names,
        yticklabels=['Doc 1: "patient reports chest pain"',
                     'Doc 2: "patient denies chest pain"',
                     'Doc 3: "patient reports headache"'],
        ax=ax,
        cbar_kws={'label': 'Word Count'}
    )

    ax.set_title('Document-Term Matrix (Bag of Words)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Vocabulary Terms', fontsize=11)
    ax.set_ylabel('Documents', fontsize=11)

    plt.tight_layout()
    plt.savefig('bow_heatmap.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved bow_heatmap.png")


# 2. TF-IDF Bar Chart
def generate_tfidf_barchart():
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()
    idf_values = vectorizer.idf_

    # Sort by IDF value
    sorted_indices = np.argsort(idf_values)[::-1]
    sorted_names = feature_names[sorted_indices]
    sorted_idf = idf_values[sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Color bars by IDF value (higher = more distinctive)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_idf)))

    bars = ax.barh(sorted_names, sorted_idf, color=colors)

    # Add value labels
    for bar, val in zip(bars, sorted_idf):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=10)

    ax.set_xlabel('IDF Weight (higher = more distinctive)', fontsize=11)
    ax.set_ylabel('Term', fontsize=11)
    ax.set_title('Inverse Document Frequency (IDF) Weights', fontsize=14, fontweight='bold')

    # Add annotation
    ax.annotate('Common words\n(low IDF)', xy=(1.0, 0), xytext=(1.2, 1),
                fontsize=9, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
    ax.annotate('Distinctive words\n(high IDF)', xy=(1.7, 4), xytext=(1.5, 3),
                fontsize=9, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    ax.set_xlim(0, 2.0)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig('tfidf_weights.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved tfidf_weights.png")


# 3. Document Similarity Scatter Plot
def generate_similarity_scatter():
    # Use more documents for better visualization
    docs_extended = [
        "patient reports chest pain and shortness of breath",
        "patient reports chest discomfort and difficulty breathing",
        "patient complains of headache and nausea",
        "patient presents with migraine symptoms",
        "chest pain radiating to left arm",
    ]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(docs_extended)

    # Reduce to 2D with PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X.toarray())

    fig, ax = plt.subplots(figsize=(10, 7))

    # Define clusters by topic
    colors = ['#e74c3c', '#e74c3c', '#3498db', '#3498db', '#e74c3c']
    markers = ['o', 's', 'o', 's', '^']

    # Plot points
    for i, (x, y) in enumerate(X_2d):
        ax.scatter(x, y, c=colors[i], s=200, marker=markers[i],
                   edgecolors='black', linewidths=1, zorder=5)

        # Shortened labels
        labels = [
            "chest pain +\nshortness of breath",
            "chest discomfort +\ndifficulty breathing",
            "headache +\nnausea",
            "migraine\nsymptoms",
            "chest pain\nradiating to arm"
        ]
        ax.annotate(labels[i], (x, y), textcoords="offset points",
                    xytext=(10, 10), fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='gray', alpha=0.8))

    # Draw similarity lines between related docs
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(X)

    # Draw lines for high similarity pairs
    for i in range(len(docs_extended)):
        for j in range(i+1, len(docs_extended)):
            sim = sim_matrix[i, j]
            if sim > 0.15:  # Only show meaningful similarities
                alpha = min(sim * 1.5, 0.8)
                ax.plot([X_2d[i, 0], X_2d[j, 0]],
                        [X_2d[i, 1], X_2d[j, 1]],
                        'k-', alpha=alpha, linewidth=sim * 3, zorder=1)
                # Add similarity label at midpoint
                mid_x = (X_2d[i, 0] + X_2d[j, 0]) / 2
                mid_y = (X_2d[i, 1] + X_2d[j, 1]) / 2
                if sim > 0.25:
                    ax.text(mid_x, mid_y, f'{sim:.2f}', fontsize=8,
                            ha='center', va='center',
                            bbox=dict(boxstyle='round', facecolor='yellow',
                                      alpha=0.7, edgecolor='none'))

    ax.set_xlabel('Principal Component 1', fontsize=11)
    ax.set_ylabel('Principal Component 2', fontsize=11)
    ax.set_title('Document Similarity in TF-IDF Vector Space\n(line thickness = cosine similarity)',
                 fontsize=14, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', edgecolor='black', label='Chest/cardiac symptoms'),
        Patch(facecolor='#3498db', edgecolor='black', label='Head/neuro symptoms'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    ax.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    plt.savefig('document_similarity.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved document_similarity.png")


if __name__ == "__main__":
    generate_bow_heatmap()
    generate_tfidf_barchart()
    generate_similarity_scatter()
    print("\nAll visualizations generated!")
