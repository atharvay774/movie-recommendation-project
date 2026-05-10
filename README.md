# 🎬 Movie Recommender System

> A content-based movie recommendation engine powered by machine learning and the TMDB 5000 movies dataset. Built with scikit-learn, Streamlit, and cosine similarity for intelligent movie discovery.

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Virtual Environment Setup](#virtual-environment-setup)
- [Dependency Installation](#dependency-installation)
- [How to Run](#how-to-run)
- [ML Workflow Explanation](#ml-workflow-explanation)
- [Recommendation Engine](#recommendation-engine)
- [Dataset Information](#dataset-information)
- [Future Improvements](#future-improvements)
- [Screenshots](#screenshots)
- [License](#license)
- [Author](#author)

---

## 📖 Overview

The **Movie Recommender System** is an intelligent recommendation engine that leverages **content-based filtering** to suggest movies to users based on their selections. By analyzing movie metadata (genres, keywords, cast, crew, and plot summaries), the system computes similarity scores using cosine similarity and recommends the top 5 most similar films.

This project demonstrates end-to-end machine learning application development, from data preprocessing and feature engineering to model deployment via an interactive web interface.

---

## ✨ Key Features

- **Content-Based Filtering**: Recommends movies based on genres, keywords, cast, crew, and plot descriptions
- **Fast Similarity Computation**: Pre-computed cosine similarity matrix for instant recommendations
- **Interactive Web UI**: Clean, user-friendly Streamlit interface
- **Movie Posters**: Fetches and displays actual movie posters from TMDB API
- **Scalable Architecture**: Pickle-based model serialization for efficient deployment
- **5000+ Movies**: Trained on comprehensive TMDB 5000 movies dataset
- **Zero Cold-Start for Known Movies**: Works immediately for any movie in the dataset

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **ML Framework** | scikit-learn |
| **Vectorization** | CountVectorizer (TF approach) |
| **Similarity Metric** | Cosine Similarity |
| **Web Framework** | Streamlit |
| **API Integration** | Requests (TMDB API) |
| **Model Storage** | Pickle |
| **Dataset** | TMDB 5000 Movies & Credits |

---

## 🏗️ Project Architecture

```
movie-recommender-system/
├── main.ipynb                      # Complete ML workflow notebook
├── streamlit-app.py                # Interactive Streamlit application
├── model/
│   ├── movie_list.pkl              # Serialized movie dataframe
│   └── similarity.pkl               # Pre-computed cosine similarity matrix
├── README.md                        # Project documentation
└── requirements.txt                 # Python dependencies (optional)
```

### Architecture Flow

```
Raw Data (TMDB CSVs)
    ↓
Data Loading & Merging
    ↓
Feature Extraction (Genres, Keywords, Cast, Crew, Overview)
    ↓
Text Preprocessing & Cleaning
    ↓
Feature Vectorization (CountVectorizer)
    ↓
Cosine Similarity Computation
    ↓
Model Serialization (Pickle)
    ↓
Streamlit Deployment
    ↓
User Interaction & Recommendations
```

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- TMDB API Key (free account at [https://www.themoviedb.org/settings/api](https://www.themoviedb.org/settings/api))

### Clone the Repository

```bash
git clone https://github.com/yourusername/movie-recommender-system.git
cd movie-recommender-system
```

---

## 🔧 Virtual Environment Setup

### On Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### On macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

Verify activation:
```bash
which python  # Should show path to venv/bin/python
```

---

## 📥 Dependency Installation

Install all required packages:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy scikit-learn streamlit requests
```

**Key Dependencies:**
- `pandas >= 1.3.0` - Data manipulation and analysis
- `numpy >= 1.21.0` - Numerical computing
- `scikit-learn >= 1.0.0` - Machine learning toolkit
- `streamlit >= 1.0.0` - Web application framework
- `requests >= 2.26.0` - HTTP library for API calls

---

## 🚀 How to Run

### 1. Run the Streamlit Application

Ensure your virtual environment is activated, then:

```bash
streamlit run streamlit-app.py
```

The application will:
- Open in your default browser at `http://localhost:8501`
- Load pre-trained models from `model/` directory
- Display a dropdown menu with 5000+ movie titles
- Show 5 recommended movies with posters when you click "Show Recommendation"

### 2. Train the Model (Optional)

To retrain the recommendation model with updated data:

```bash
jupyter notebook main.ipynb
```

Run all cells in sequence to:
- Load TMDB 5000 movies and credits datasets
- Preprocess and feature-engineer the data
- Generate cosine similarity matrix
- Export new pickle models

**Note**: You'll need the raw TMDB CSV files in `/kaggle/input/tmdb-movie-metadata/`

---

## 🧠 ML Workflow Explanation

### 1. **Data Loading & Exploration**
- Load `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`
- Merge on movie title
- Inspect dataset dimensions and structure

### 2. **Feature Selection**
Selected columns: `movie_id`, `title`, `overview`, `genres`, `keywords`, `cast`, `crew`

### 3. **Feature Extraction**
Extract structured information from JSON-formatted columns:
- **Genres**: Movie categories (Action, Drama, Comedy, etc.)
- **Keywords**: Plot keywords and themes
- **Cast**: Top 3 actors (actors field contains JSON objects)
- **Crew**: Director name (job == 'Director')
- **Overview**: Movie plot summary

### 4. **Text Preprocessing**
- Remove spaces from names (e.g., "Johnny Depp" → "JohnnyDepp")
- Tokenize overview text into words
- Handle missing values with dropna()

### 5. **Feature Combination**
Combine all extracted features into a single "tags" string:
```
tags = overview + genres + keywords + cast + crew
```

This rich combined feature set captures movie essence from multiple angles.

### 6. **Vectorization**
Use **CountVectorizer** from scikit-learn:
- Max features: 5000 (controls vocabulary size)
- Remove English stop words
- Creates a sparse TF (Term Frequency) matrix
- Shape: (5000 movies, 5000 features)

### 7. **Similarity Computation**
Calculate **cosine similarity** between all movies:
- Dense 5000×5000 similarity matrix
- Value range: [0, 1] where 1 = identical, 0 = completely different
- Computation: similarity(movie_i, movie_j) = cos(θ) between feature vectors

### 8. **Model Serialization**
Save trained models using pickle:
```python
pickle.dump(movies_df, open('model/movie_list.pkl', 'wb'))
pickle.dump(similarity_matrix, open('model/similarity.pkl', 'wb'))
```

---

## 🎯 Recommendation Engine

### How Recommendations Work

When a user selects a movie:

1. **Movie Lookup**: Find the index of the selected movie in the dataframe
2. **Similarity Retrieval**: Get the similarity scores between the selected movie and all others
3. **Ranking**: Sort movies by similarity score (descending)
4. **Top-K Selection**: Select top 5 movies (excluding the input movie itself)
5. **Poster Fetching**: Retrieve movie posters from TMDB API using `movie_id`
6. **Display**: Show recommendations with titles and posters in Streamlit UI

### Example

```
User selects: "The Lego Movie"
↓
System finds movies with similar:
  - Genres (Animation, Comedy, Adventure)
  - Keywords (building, humor, family-friendly)
  - Cast (similar actors)
↓
Top 5 recommendations:
1. The Lego Batman Movie (similarity: 0.95)
2. Toy Story (similarity: 0.87)
3. Wreck-It Ralph (similarity: 0.84)
4. Finding Nemo (similarity: 0.81)
5. Monsters vs. Aliens (similarity: 0.78)
```

---

## 📊 Dataset Information

### TMDB 5000 Movies Dataset

**Source**: Kaggle - TMDB 5000 Movie Dataset  
**Movies Included**: 5000 movies  
**Date Range**: Multiple decades of cinema

**Available Columns** (pre-processed):
- `movie_id` - TMDB unique identifier
- `title` - Movie title
- `overview` - Plot summary
- `genres` - Movie genres (JSON array)
- `keywords` - Plot keywords (JSON array)
- `cast` - Actor names (JSON array, top 3 selected)
- `crew` - Director information (JSON array, director extracted)

**After Processing**:
- Total features used: 5000 (from CountVectorizer)
- Feature type: Term frequency (TF)
- Sparsity: High (typical for text data)
- Vectorization method: Bag-of-words approach

**Dataset Statistics**:
- Clean movies after dropna(): ~4800-4900 valid entries
- Average overview length: 150-200 words
- Movies per genre: 100-800 (varied distribution)

---

## 🚀 Future Improvements

### Short-term Enhancements
- [ ] **Collaborative Filtering**: Add user-based recommendations using user ratings
- [ ] **Hybrid Model**: Combine content-based and collaborative filtering
- [ ] **Ratings Integration**: Use TMDB ratings as ranking signals
- [ ] **Improved Vectorization**: Experiment with TF-IDF, Word2Vec, or BERT embeddings
- [ ] **Caching**: Add Redis for faster API calls

### Medium-term Features
- [ ] **User Profiles**: Track user preferences and viewing history
- [ ] **Advanced Filters**: Filter recommendations by year, rating, language
- [ ] **Personalization**: Learn from user feedback to improve suggestions
- [ ] **Database Backend**: Replace pickle with PostgreSQL for scalability
- [ ] **API Endpoint**: Create FastAPI/Flask REST API for mobile apps

### Long-term Scaling
- [ ] **Deep Learning**: Implement neural collaborative filtering
- [ ] **Real-time Updates**: Stream new movies and update recommendations
- [ ] **A/B Testing**: Test different algorithms and models
- [ ] **Microservices**: Deploy as containerized services (Docker/Kubernetes)
- [ ] **Recommendation Diversity**: Balance accuracy with recommendation variety
- [ ] **Cold-start Solution**: Implement content-based fallback for new users

---

## 📸 Screenshots

### 1. Application Home Screen
![Movie Recommender Home](https://via.placeholder.com/600x400?text=Streamlit+Movie+Selection)

### 2. Movie Selection Dropdown
![Movie Dropdown](https://via.placeholder.com/600x400?text=Movie+Dropdown+List)

### 3. Recommendations Display
![Recommendations](https://via.placeholder.com/600x400?text=Top+5+Movie+Recommendations)

*Note: Replace placeholder images with actual screenshots of your application*

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

You are free to use, modify, and distribute this project for personal and commercial purposes, provided you include the original license notice.

---

## 👤 Author

**Your Name**  
- Portfolio: [your-website.com](https://your-website.com)
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [LinkedIn Profile](https://linkedin.com/in/yourprofile)

### Acknowledgments

- **Dataset**: TMDB 5000 Movies Dataset from Kaggle
- **API**: The Movie Database (TMDB) for poster images
- **Framework**: Streamlit for making web deployment simple
- **ML Tools**: scikit-learn for vectorization and similarity computation

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## ❓ FAQ

**Q: How is the recommendation calculated?**  
A: The system calculates cosine similarity between the selected movie's feature vector and all other movies' vectors. The 5 movies with the highest similarity scores are recommended.

**Q: Can I add new movies to the system?**  
A: Yes, you can retrain the model with an updated dataset using the `main.ipynb` notebook. You'll need the raw TMDB data files.

**Q: Why use content-based filtering instead of collaborative filtering?**  
A: Content-based filtering works for new movies and avoids cold-start problems. It requires only movie metadata, not user ratings.

**Q: What does the TMDB API key do?**  
A: The API key fetches movie posters in real-time. Without it, the application will fail when trying to display recommendations.

**Q: How accurate are the recommendations?**  
A: Accuracy depends on the quality of input features and the appropriateness of cosine similarity for your use case. Consider user feedback to evaluate and improve the model.

---

**Made with ❤️ for movie lovers and ML enthusiasts**
