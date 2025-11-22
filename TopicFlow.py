# =============================================================================
# UNIFIED NLP ANALYSIS: BERTopic + Word2Vec for Bibliometric and Scientometric Analysis
# =============================================================================

# -----------------------------------------------------------------------------
# INSTALLATION
# -----------------------------------------------------------------------------

!pip install pandas bertopic scikit-learn umap-learn hdbscan sentence-transformers pybibx tabulate prettytable


# -----------------------------------------------------------------------------
# PART 1: IMPORTS & INITIAL SETUP
# -----------------------------------------------------------------------------
import sys
import glob
import os
import pandas as pd
from google.colab import drive
from pybibx.base import pbx_probe
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from tabulate import tabulate
from prettytable import PrettyTable
import textwrap

# Google Drive Connection
drive.mount('/GDrive')

# Enter your folder path here
WORKING_DIR = '/GDrive/My Drive/your_folder/'
sys.path.insert(0, WORKING_DIR)

print(f" Working Directory: {WORKING_DIR}")

# -----------------------------------------------------------------------------
# PART 2: DATA LOADING (COMBINE & IMPORT BIB FILES)
# -----------------------------------------------------------------------------

def combine_and_load_bib_files(working_dir, db='wos', year_start=None, year_end=None):
    """
    Combines all .bib files in directory and loads data
    
    Args:
        working_dir: Working directory path
        db: Database type ('wos' or 'scopus')
        year_start: Optional start year filter
        year_end: Optional end year filter
    
    Returns:
        bibfile object
    """
    # Find and combine .bib files
    bib_files = glob.glob(working_dir + '*.bib')
    output_filename = 'combined_all.bib'
    output_path = working_dir + output_filename
    
    # Remove already combined file from list
    if output_path in bib_files:
        bib_files.remove(output_path)
    
    if bib_files:
        combined_content = ""
        for file_name in bib_files:
            with open(file_name, 'r', encoding='utf-8') as f:
                combined_content += f.read() + "\n\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(combined_content)
        
        print(f"✅ {len(bib_files)} files combined -> {output_filename}")
        for f in bib_files:
            print(f"  - {f.split('/')[-1]}")
    else:
        print(f"⚠️ Using existing combined file: {output_filename}")
    
    # Load data
    try:
        bibfile = pbx_probe(file_bib=output_path, db=db, del_duplicated=True)
        print(f"✅ Data Loaded: {len(bibfile.data)} articles")
        
        # Optional year filter
        if year_start and year_end:
            bibfile.filter_bib(year_str=year_start, year_end=year_end)
            print(f"   Filtered: {year_start}-{year_end}")
        
        # Display sample
        df_head = bibfile.data.head(n=5)
        print("\n📊 Sample Data:")
        print(tabulate(df_head, headers=df_head.columns, tablefmt='psql'))
        
        return bibfile
        
    except Exception as e:
        raise ValueError(f"❌ Error loading data: {e}")

# Load data
bibfile = combine_and_load_bib_files(
    WORKING_DIR, 
    db='wos',
    year_start=2015,  # Optional: set to None to skip
    year_end=2024     # Optional: set to None to skip
)

# -----------------------------------------------------------------------------
# PART 3: TOPIC ANALYZER CLASS (ENHANCED VERSION)
# -----------------------------------------------------------------------------

class TopicAnalyzer:
    """Enhanced class for academic text analysis using BERTopic"""
    
    def __init__(self, bibfile_data):
        """
        Args:
            bibfile_data: DataFrame containing bibliographic data
        """
        self.data = bibfile_data
        self.topic_model = None
        self.topics = None
        self.probs = None
        self._texts_used_for_fit = None
        self._original_indices_for_fit = None
    
    def _prepare_text_data(self, text_column='abstract', min_words=3):
        """
        Prepares and cleans text data
        
        Args:
            text_column: Text column to use ('abstract' or 'title')
            min_words: Minimum word count
        
        Returns:
            Cleaned text list, original indices
        """
        if text_column not in self.data.columns:
            raise ValueError(f"Column '{text_column}' not found!")
        
        # Filter and clean
        df_filtered = self.data[self.data[text_column].notna()].copy()
        texts = df_filtered[text_column].astype(str).tolist()
        original_indices = df_filtered.index.tolist()
        
        # Remove short texts
        texts_filtered = []
        indices_filtered = []
        for i, t in enumerate(texts):
            if len(t.split()) >= min_words:
                texts_filtered.append(t)
                indices_filtered.append(original_indices[i])
        
        removed = len(texts) - len(texts_filtered)
        if removed > 0:
            print(f"⚠️  {removed} short texts (< {min_words} words) filtered out")
        
        return texts_filtered, indices_filtered
    
    def create_base_model(self, verbose=True):
        """Creates basic BERTopic model"""
        vectorizer_model = CountVectorizer(stop_words="english")
        
        topic_model = BERTopic(
            verbose=verbose,
            vectorizer_model=vectorizer_model
        )
        
        return topic_model
    
    def create_advanced_model(self, min_cluster_size=10, top_n_words=10):
        """
        Creates optimized BERTopic model with advanced parameters
        
        Args:
            min_cluster_size: Minimum cluster size for HDBSCAN
            top_n_words: Number of top words per topic
        
        Returns:
            BERTopic model
        """
        # Advanced embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # UMAP dimensionality reduction
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        
        # HDBSCAN clustering
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size, #(10)
            min_samples=5,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        # Advanced CountVectorizer
        vectorizer_model = CountVectorizer(
            stop_words="english",
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2)
        )
        
        # BERTopic model
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            min_topic_size=min_cluster_size, #(10)
            nr_topics="auto",
            top_n_words=top_n_words,
            calculate_probabilities=True,
            verbose=True
        )
        
        return topic_model
    
    def analyze_all_data(self, use_advanced=True, text_column='abstract', 
                        min_words=3, min_cluster_size=10):
        """
        Performs topic analysis on entire dataset
        
        Args:
            use_advanced: Use advanced model (True/False)
            text_column: Text column to analyze ('abstract' or 'title')
            min_words: Minimum word count filter
            min_cluster_size: Minimum cluster size
        """
        # Prepare data
        texts, original_indices = self._prepare_text_data(text_column, min_words)
        self._texts_used_for_fit = texts
        self._original_indices_for_fit = original_indices
        
        if not texts:
            raise ValueError(f"No suitable text found in '{text_column}' column!")
        
        print(f"\n📊 Analyzing {len(texts)} {text_column}s...")
        print("=" * 80)
        
        # Model selection
        if use_advanced:
            self.topic_model = self.create_advanced_model(min_cluster_size=min_cluster_size)
        else:
            self.topic_model = self.create_base_model()
        
        # Train model
        print("\n⏳ Training model... (This may take time)")
        self.topics, self.probs = self.topic_model.fit_transform(texts)
        
        print(f"\n✅ Training completed!")
        print(f"   Total {len(set(self.topics))} topics detected (including outliers)")
        print("=" * 80)
        
        # Show results
        self.show_results()
        
        return self.topics, self.probs
    
    def analyze_by_year(self, target_year, use_advanced=True, text_column='abstract', 
                       min_words=3, min_cluster_size=5):
        """
        Analyzes data for a specific year
        
        Args:
            target_year: Year to analyze
            use_advanced: Use advanced model
            text_column: Text column to analyze
            min_words: Minimum word count
            min_cluster_size: Minimum cluster size
        """
        if 'year' not in self.data.columns:
            raise ValueError("'year' column not found!")
        
        # Filter by year
        target_year = str(target_year)
        year_data = self.data[self.data['year'] == target_year]
        
        if year_data.empty:
            raise ValueError(f"No data found for year {target_year}!")
        
        print(f"\n📊 Analyzing year {target_year} ({len(year_data)} articles)")
        print("=" * 80)
        
        # Temporarily replace data
        original_data = self.data
        self.data = year_data
        
        try:
            # Analyze
            self.analyze_all_data(
                use_advanced=use_advanced,
                text_column=text_column,
                min_words=min_words,
                min_cluster_size=min_cluster_size
            )
        finally:
            # Restore original data
            self.data = original_data
    
    def analyze_over_time(self, text_column='abstract', nr_bins=20):
        """
        Analyzes topic evolution over time
        
        Args:
            text_column: Text column to use
            nr_bins: Number of time intervals
        """
        if self.topic_model is None:
            raise ValueError("Model must be trained first! Call analyze_all_data()")
        
        if self._texts_used_for_fit is None or self._original_indices_for_fit is None:
            raise ValueError("No fitted data found!")
        
        # Get timestamps
        if 'year' not in self.data.columns:
            raise ValueError("'year' column not found!")
        
        texts = self._texts_used_for_fit
        timestamps = self.data.loc[self._original_indices_for_fit, 'year'].tolist()
        
        print(f"\n⏳ Performing time series analysis...")
        print(f"   Year range: {min(timestamps)} - {max(timestamps)}")
        
        topics_over_time = self.topic_model.topics_over_time(
            texts,
            timestamps,
            nr_bins=nr_bins
        )
        
        print(f"✅ Time series analysis completed!")
        
        return topics_over_time
    
    def find_similar_topics(self, query, top_n=5):
        """
        Finds topics similar to a query
        
        Args:
            query: Search query
            top_n: Top N similar topics
        """
        if self.topic_model is None:
            raise ValueError("Model must be trained first!")
        
        if self.topic_model.embedding_model is None:
            print("\n⚠️  WARNING: Embedding model not found!")
            print("   Load manually: analyzer.topic_model.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')")
            return None, None
        
        try:
            similar_topics, similarity = self.topic_model.find_topics(query, top_n=top_n)
            
            print(f"\n🔍 Most similar topics for '{query}':")
            print("=" * 80)
            for i in range(len(similar_topics)):
                if similar_topics[i] == -1:
                    continue
                print(f"\nTopic {similar_topics[i]}: Similarity = {similarity[i]:.3f}")
                topic_words = self.topic_model.get_topic(similar_topics[i])[:5]
                print(f"  Keywords: {', '.join([w[0] for w in topic_words])}")
            
            return similar_topics, similarity
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return None, None
    
    def get_topic_distribution_by_year(self, text_column='abstract', min_words=3, top_n=5):
        """
        Shows topic distribution for each year
        
        Args:
            text_column: Text column to analyze
            min_words: Minimum word count
            top_n: Top N topics per year to display
        """
        if self.topic_model is None:
            raise ValueError("Model must be loaded first!")
        
        # Predict topics if not available
        if self.topics is None:
            print("\n⏳ Predicting topics...")
            texts, original_indices = self._prepare_text_data(text_column, min_words)
            self.topics, self.probs = self.topic_model.transform(texts)
            self._texts_used_for_fit = texts
            self._original_indices_for_fit = original_indices
            print(f"✅ Topics predicted for {len(texts)} texts!")
        
        if 'year' not in self.data.columns:
            raise ValueError("'year' column not found!")
        
        # Get years for topics
        years_for_topics = self.data.loc[self._original_indices_for_fit, 'year'].values
        
        # Create DataFrame
        df_topics = pd.DataFrame({
            'Topic': self.topics,
            'Year': years_for_topics
        })
        
        # Topic distribution by year
        year_topic_dist = df_topics.groupby(['Year', 'Topic']).size().reset_index(name='Count')
        
        print("\n📊 Topic Distribution by Year:")
        print("=" * 80)
        for year in sorted(year_topic_dist['Year'].unique()):
            year_data = year_topic_dist[year_topic_dist['Year'] == year]
            top_topics = year_data.nlargest(top_n, 'Count')
            print(f"\n{int(year)}:")
            for _, row in top_topics.iterrows():
                if row['Topic'] != -1:
                    topic_words = self.topic_model.get_topic(int(row['Topic']))[:5]
                    keywords = ', '.join([w[0] for w in topic_words])
                    print(f"  Topic {int(row['Topic'])} ({keywords}): {int(row['Count'])} articles")
        
        return year_topic_dist
    
    def save_model(self, filepath="bertopic_model", save_embeddings=False):
        """
        Saves trained model
        
        Args:
            filepath: File path (without extension)
            save_embeddings: Also save embeddings (larger file size)
        """
        if self.topic_model is None:
            raise ValueError("No model to save!")
        
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2" if save_embeddings else None
        
        self.topic_model.save(
            filepath,
            serialization="pytorch",
            save_ctfidf=True,
            save_embedding_model=embedding_model
        )
        
        print(f"✅ Model saved: {filepath}")
        if save_embeddings:
            print(f"   ⚠️  Embedding model saved (~90MB)")
        else:
            print(f"   ⚠️  Embedding model NOT saved")
    
    def load_model(self, filepath="bertopic_model"):
        """
        Loads previously saved model
        
        Args:
            filepath: Model file path
        """
        try:
            self.topic_model = BERTopic.load(filepath)
            print(f"✅ Model loaded: {filepath}")
            print(f"  Total {len(self.topic_model.get_topics())} topics available")
            
            # Reset state
            self.topics = None
            self.probs = None
            self._texts_used_for_fit = None
            self._original_indices_for_fit = None
            
            return self.topic_model
        except Exception as e:
            raise ValueError(f"Model could not be loaded: {e}")
    
    def show_results(self, top_n_topics=None):
        """Shows analysis results"""
        if self.topic_model is None:
            raise ValueError("Model must be trained first!")
        
        print("\n" + "=" * 80)
        print("TOPIC INFORMATION")
        print("=" * 80)
        topic_info = self.topic_model.get_topic_info()
        if top_n_topics:
            topic_info = topic_info.head(top_n_topics + 1)
        print(tabulate(topic_info, headers='keys', tablefmt='psql'))
        
        print("\n" + "=" * 80)
        print("TOPIC DETAILS")
        print("=" * 80)
        topic_ids = list(self.topic_model.get_topics().keys())
        if top_n_topics:
            topic_ids = [t for t in topic_ids if t != -1][:top_n_topics]
        
        for topic_id in topic_ids:
            if topic_id != -1:
                print(f"\nTopic {topic_id}:")
                topic_words = self.topic_model.get_topic(topic_id)
                for word, score in topic_words[:10]:
                    print(f"  {word}: {score:.4f}")
    
    def export_results(self, output_path="bertopic_results.csv"):
        """
        Exports analysis results to CSV
        
        Args:
            output_path: Output file path
        """
        if self.topic_model is None or self.topics is None:
            raise ValueError("Model must be trained first!")
        
        results_df = self.data.loc[self._original_indices_for_fit].copy()
        results_df['Topic'] = self.topics
        
        # Add topic labels
        topic_labels = {}
        for topic_id in set(self.topics):
            if topic_id != -1:
                words = self.topic_model.get_topic(topic_id)[:3]
                topic_labels[topic_id] = ', '.join([w[0] for w in words])
            else:
                topic_labels[topic_id] = 'Outlier'
        
        results_df['Topic_Label'] = results_df['Topic'].map(topic_labels)
        
        results_df.to_csv(output_path, index=False)
        print(f"\n✅ Results saved: {output_path}")
        print(f"   Total {len(results_df)} records")
    
    def visualize(self, viz_type="barchart", **kwargs):
        """
        Creates visualizations
        
        Args:
            viz_type: Type ("barchart", "topics", "hierarchy", "heatmap", "topics_over_time")
            **kwargs: Visualization parameters
        """
        if self.topic_model is None:
            raise ValueError("Model must be trained first!")
        
        print(f"\n📊 Creating {viz_type} visualization...")
        
        viz_map = {
            "barchart": self.topic_model.visualize_barchart,
            "topics": self.topic_model.visualize_topics,
            "hierarchy": self.topic_model.visualize_hierarchy,
            "heatmap": self.topic_model.visualize_heatmap,
            "topics_over_time": self.topic_model.visualize_topics_over_time
        }
        
        if viz_type not in viz_map:
            raise ValueError(f"Unknown visualization type: {viz_type}")
        
        return viz_map[viz_type](**kwargs)

# Initialize Analyzer
analyzer = TopicAnalyzer(bibfile.data)
print("✅ TopicAnalyzer initialized!")

# -----------------------------------------------------------------------------
# USAGE EXAMPLES & WORKFLOWS (PART 3:)
# -----------------------------------------------------------------------------

WORKFLOW 1: Train New Model (All Data)
--------------------------------------
topics, probs = analyzer.analyze_all_data(use_advanced=True)
analyzer.save_model("my_model", save_embeddings=True)

WORKFLOW 2: Load Existing Model
--------------------------------------
analyzer.load_model("my_model")
analyzer.show_results()

WORKFLOW 3: Visualizations
--------------------------------------
fig1 = analyzer.visualize("barchart", top_n_topics=15)
fig2 = analyzer.visualize("topics")
fig3 = analyzer.visualize("hierarchy")
fig4 = analyzer.visualize("heatmap")
fig1.show()

WORKFLOW 4: Time Series Analysis
--------------------------------------
topics_over_time = analyzer.analyze_over_time(nr_bins=20)
fig = analyzer.visualize("topics_over_time", topics_over_time=topics_over_time, top_n_topics=10)
fig.show()

WORKFLOW 5: Year-by-Year Distribution
--------------------------------------
year_dist = analyzer.get_topic_distribution_by_year(top_n=5)
year_dist.to_csv('yearly_distribution.csv', index=False)

WORKFLOW 6: Find Similar Topics
--------------------------------------
analyzer.find_similar_topics('artificial intelligence', top_n=5)

WORKFLOW 7: Export Results
--------------------------------------
#analyzer.export_results("my_results.csv")

WORKFLOW 8: Analyze Specific Year
--------------------------------------
analyzer.analyze_by_year(2023, use_advanced=True)
""")

# -----------------------------------------------------------------------------
# PART 4: WORD EMBEDDINGS (WORD2VEC) - PYBIBX NATIVE
# -----------------------------------------------------------------------------

def run_word2vec_analysis(bibfile):
    """
    Runs Word2Vec analysis using pybibx
    
    Args:
        bibfile: pybibx bibfile object
    """
    print("⏳ Training Word2Vec model...")
    
    # Train Word2Vec
    model, corpus, w_emb, vocab = bibfile.word_embeddings(
        stop_words=['en'],
        lowercase=True,
        rmv_accents=True,
        rmv_special_chars=False,
        rmv_numbers=True,
        rmv_custom_words=[],
        vector_size=100,
        window=5,
        min_count=1,
        epochs=10
    )
    
    print("✅ Word2Vec model trained!")
    
    # 1. Similarity Test
    print("\n🔍 Similarity Test (urban vs city):")
    try:
        sim = bibfile.word_embeddings_sim(model, word_1='urban', word_2='city')
        print(f"   Score: {sim:.4f}")
    except:
        print("   ⚠️ Words not found in vocabulary")
    
    # 2. Find Documents
    print("\n📄 Document Search (['urban', 'human']):")
    res = bibfile.word_embeddings_find_doc(corpus, target_words=['urban', 'human'])
    print(f"   Found IDs: {res[:10]}")
    
    # 3. Word Operations
    print("\n🧮 Word Operations (positive=['urban', 'group'], negative=['risk']):")
    try:
        ops = bibfile.word_embeddings_operations(
            model, 
            positive=['urban', 'group'], 
            negative=['risk'], 
            topn=10
        )
        print(ops)
    except:
        print("   ⚠️ Operation failed")
    
    return model, corpus, w_emb, vocab

# Uncomment to run Word2Vec analysis
# w2v_model, w2v_corpus, w2v_emb, w2v_vocab = run_word2vec_analysis(bibfile)

# -----------------------------------------------------------------------------
# PART 5: PYBIBX NATIVE TOPIC MODELING (ALTERNATIVE)
# -----------------------------------------------------------------------------

def run_pybibx_topic_modeling(bibfile, selected_model='sentence-transformers/all-MiniLM-L6-v2'): #allenai/scibert_scivocab_uncased (alternative)
    """
    Runs topic modeling using pybibx native methods
    
    Args:
        bibfile: pybibx bibfile object
        selected_model: Embedding model to use
    """
    print(f"⚙️ Model: {selected_model}")
    
    # 1. Create Embeddings
    print("⏳ Creating embeddings...")
    bibfile.create_embeddings(
        stop_words=['en'],
        rmv_custom_words=[],
        corpus_type='abs',
        model=selected_model
    )
    
    # 2. Create Topics
    print("⏳ Creating topics...")
    bibfile.topics_creation(
        stop_words=['en'],
        rmv_custom_words=[],
        embeddings=True,
        model=selected_model
    )
    
    topics = bibfile.topics
    probs = bibfile.probs
    print(f"✅ Topic count: {len(set(topics)) - 1}")
    
    # 3. Visualizations
    print("\n📊 Creating visualizations...")
    
    # A) Distribution
    print("   • Distribution Chart")
    bibfile.graph_topics_distribution(view='notebook')
    
    # B) Intertopic Distance
    print("   • Intertopic Distance Map")
    bibfile.graph_topics(view='notebook')
    
    # C) Projection
    print("   • Topic Projection (2D Map)")
    bibfile.graph_topics_projection(view='notebook')
    
    # D) Heatmap
    print("   • Heatmap")
    bibfile.graph_topics_heatmap(view='notebook')
    
    # E) Time Series (if year column exists)
    if 'year' in bibfile.data.columns:
        print("   • Topics over Time")
        bibfile.graph_topics_time(view='notebook')
    
    # 4. Detailed Analysis
    print("\n📋 Detailed Tables...")
    
    # A) Representative Articles
    print("   • Representative Articles (First 5):")
    reps = bibfile.topics_representatives()
    print(reps.head())
    
    # B) Topic Words for Document
    try:
        doc_id = 0
        print(f"   • Word Analysis for Doc ID {doc_id}:")
        df_words = bibfile.topics_words(doc_id=doc_id)
        print(df_words.head())
    except:
        print("   ⚠️ Word analysis not available for specified ID")
    
    # C) Topic Search
    print("\n🔍 Topic Search: 'technology'")
    sim_topics, similarity = bibfile.topic_model.find_topics('technology', top_n=5)
    for i in range(len(sim_topics)):
        print(f"   Topic {sim_topics[i]}: {round(similarity[i], 3)}")
    
    # 5. Save Model
    print("\n💾 Saving model...")
    bibfile.topic_model.save(WORKING_DIR + 'pybibx_topic_model')
    print(f"✅ Model saved to: {WORKING_DIR}pybibx_topic_model")
    
    return topics, probs

# Uncomment to run pybibx native topic modeling
# pybibx_topics, pybibx_probs = run_pybibx_topic_modeling(bibfile)

