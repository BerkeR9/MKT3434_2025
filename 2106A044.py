import sys
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QTabWidget, QPushButton, QLabel, 
                           QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox,
                           QGroupBox, QScrollArea, QTextEdit, QStatusBar,
                           QProgressBar, QCheckBox, QGridLayout, QMessageBox,
                           QDialog, QLineEdit)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn import datasets, preprocessing, model_selection
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, confusion_matrix
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import plotly.express as px
import plotly.graph_objects as go
import umap
from sklearn.metrics import silhouette_score
import webbrowser
import os

class MLCourseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Course GUI")
        self.setGeometry(100, 100, 1400, 800)
        
        # Initialize main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        
        # Initialize data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.current_model = None
        self.layer_config = []
        
        # Create components
        self.create_data_section()
        self.create_tabs()
        self.create_visualization()
        self.create_status_bar()

    def create_data_section(self):
        """Create the data loading and preprocessing section"""
        data_group = QGroupBox("Data Management")
        data_layout = QVBoxLayout()
        
        dataset_layout = QHBoxLayout()
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems([
            "Load Custom Dataset", "Iris Dataset", "Breast Cancer Dataset",
            "Digits Dataset", "Boston Housing Dataset", "MNIST Dataset"
        ])
        self.dataset_combo.currentIndexChanged.connect(self.load_dataset)
        dataset_layout.addWidget(QLabel("Dataset:"))
        dataset_layout.addWidget(self.dataset_combo)
        
        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_custom_data)
        dataset_layout.addWidget(self.load_btn)
        
        missing_layout = QHBoxLayout()
        self.missing_combo = QComboBox()
        self.missing_combo.addItems(["Mean Imputation", "Interpolation", "Forward Fill", "Backward Fill"])
        missing_layout.addWidget(QLabel("Missing Values:"))
        missing_layout.addWidget(self.missing_combo)
        
        preprocess_layout = QHBoxLayout()
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems(["No Scaling", "Standard Scaling", "Min-Max Scaling", "Robust Scaling"])
        self.split_combo = QComboBox()  # Yeni: Özel bölme oranları
        self.split_combo.addItems(["80-20", "70-15-15", "60-20-20"])
        self.split_combo.setCurrentText("80-20")
        preprocess_layout.addWidget(QLabel("Scaling:"))
        preprocess_layout.addWidget(self.scaling_combo)
        preprocess_layout.addWidget(QLabel("Train-Valid-Test Split:"))
        preprocess_layout.addWidget(self.split_combo)
        
        data_layout.addLayout(dataset_layout)
        data_layout.addLayout(missing_layout)
        data_layout.addLayout(preprocess_layout)
        data_group.setLayout(data_layout)
        self.layout.addWidget(data_group)

    def load_dataset(self):
        """Load selected dataset with feature selection"""
        try:
            dataset_name = self.dataset_combo.currentText()
            if dataset_name == "Load Custom Dataset":
                return
            
            if dataset_name == "Iris Dataset":
                data = datasets.load_iris()
            elif dataset_name == "Breast Cancer Dataset":
                data = datasets.load_breast_cancer()
            elif dataset_name == "Digits Dataset":
                data = datasets.load_digits()
            elif dataset_name == "Boston Housing Dataset":
                data = datasets.load_boston()
            elif dataset_name == "MNIST Dataset":
                (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
                self.X_train, self.X_test = X_train, X_test
                self.y_train, self.y_test = y_train, y_test
                self.status_bar.showMessage(f"Loaded {dataset_name}")
                return
            
            X, y = data.data, data.target
            feature_names = data.feature_names if hasattr(data, 'feature_names') else [f"Feature_{i}" for i in range(X.shape[1])]
            
            # Özellik seçimi
            selected_features = self.select_features(feature_names)
            if not selected_features:
                self.show_error("No features selected! Loading cancelled.")
                return
            
            # Seçilen özelliklere göre X'i filtrele
            selected_indices = [list(feature_names).index(f) for f in selected_features]
            X = X[:, selected_indices]
            
            # NaN temizleme
            X = self.handle_missing_values(X)
            y = np.array(y)
            if np.isnan(X).any() or np.isnan(y).any():
                raise ValueError(f"NaN values detected in {dataset_name} after cleaning!")
            
            # Yeni: Özel bölme oranları
            split_option = self.split_combo.currentText()
            if split_option == "80-20":
                test_size = 0.2
                self.X_train, self.X_test, self.y_train, self.y_test = \
                    model_selection.train_test_split(X, y, test_size=test_size, random_state=42)
                self.X_valid, self.y_valid = None, None
            elif split_option == "70-15-15":
                train_size, valid_size, test_size = 0.7, 0.15, 0.15
                X_temp, self.X_test, y_temp, self.y_test = \
                    model_selection.train_test_split(X, y, test_size=test_size, random_state=42)
                self.X_train, self.X_valid, self.y_train, self.y_valid = \
                    model_selection.train_test_split(X_temp, y_temp, train_size=train_size/(train_size+valid_size), random_state=42)
            elif split_option == "60-20-20":
                train_size, valid_size, test_size = 0.6, 0.2, 0.2
                X_temp, self.X_test, y_temp, self.y_test = \
                    model_selection.train_test_split(X, y, test_size=test_size, random_state=42)
                self.X_train, self.X_valid, self.y_train, self.y_valid = \
                    model_selection.train_test_split(X_temp, y_temp, train_size=train_size/(train_size+valid_size), random_state=42)
            
            self.apply_scaling()
            
            # Ölçeklendirme sonrası kontrol
            if np.isnan(self.X_train).any() or np.isnan(self.y_train).any():
                raise ValueError(f"NaN values introduced in {dataset_name} after scaling! Check for constant features.")
            
            self.status_bar.showMessage(f"Loaded {dataset_name} with selected features: {', '.join(selected_features)}")
            
        except Exception as e:
            self.show_error(f"Error loading dataset: {str(e)}")

    def load_custom_data(self):
        """Load custom dataset from CSV file with feature selection"""
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "Load Dataset", "", "CSV files (*.csv)")
            if file_name:
                data = pd.read_csv(file_name)
                target_col = self.select_target_column(data.columns)
                if not target_col:
                    return
                
                X = data.drop(target_col, axis=1)
                y = data[target_col]
                
                # Hedef değişkeni sayısal hale getir
                if y.dtype == 'object':
                    try:
                        y = pd.to_numeric(y, errors='coerce')
                        if y.isna().any():
                            raise ValueError(f"Target column '{target_col}' contains non-numeric values that cannot be converted!")
                    except:
                        y = y.map({'M': 0, 'B': 1})
                        if y.isna().any():
                            raise ValueError(f"Target column '{target_col}' contains values other than 'M' or 'B'!")
                
                if 'id' in X.columns:
                    X = X.drop('id', axis=1)
                
                # Özellik seçimi
                selected_features = self.select_features(X.columns)
                if not selected_features:
                    self.show_error("No features selected! Loading cancelled.")
                    return
                
                X = X[selected_features]
                X = self.handle_missing_values(X)
                y = np.array(y)
                
                # NaN kontrolü
                if np.isnan(X).any() or np.isnan(y).any():
                    raise ValueError(f"NaN values detected in custom dataset {file_name} after cleaning!")
                
                # Yeni: Özel bölme oranları
                split_option = self.split_combo.currentText()
                if split_option == "80-20":
                    test_size = 0.2
                    self.X_train, self.X_test, self.y_train, self.y_test = \
                        model_selection.train_test_split(X, y, test_size=test_size, random_state=42)
                    self.X_valid, self.y_valid = None, None
                elif split_option == "70-15-15":
                    train_size, valid_size, test_size = 0.7, 0.15, 0.15
                    X_temp, self.X_test, y_temp, self.y_test = \
                        model_selection.train_test_split(X, y, test_size=test_size, random_state=42)
                    self.X_train, self.X_valid, self.y_train, self.y_valid = \
                        model_selection.train_test_split(X_temp, y_temp, train_size=train_size/(train_size+valid_size), random_state=42)
                elif split_option == "60-20-20":
                    train_size, valid_size, test_size = 0.6, 0.2, 0.2
                    X_temp, self.X_test, y_temp, self.y_test = \
                        model_selection.train_test_split(X, y, test_size=test_size, random_state=42)
                    self.X_train, self.X_valid, self.y_train, self.y_valid = \
                        model_selection.train_test_split(X_temp, y_temp, train_size=train_size/(train_size+valid_size), random_state=42)
                
                self.apply_scaling()
                
                # Ölçeklendirme sonrası kontrol
                if np.isnan(self.X_train).any() or np.isnan(self.y_train).any():
                    raise ValueError(f"NaN values introduced in custom dataset {file_name} after scaling! Check for constant features.")
                
                self.status_bar.showMessage(f"Loaded custom dataset: {file_name} with selected features: {', '.join(selected_features)}")
        except Exception as e:
            self.show_error(f"Error loading custom dataset: {str(e)}")

    def handle_missing_values(self, X):
        """Handle missing values in the dataset"""
        if isinstance(X, pd.DataFrame):
            # DataFrame için: Eksik değerleri sütun ortalamasıyla doldur
            X = X.fillna(X.mean(numeric_only=True))
            # Hala NaN varsa hata fırlat
            if X.isnull().any().any():
                raise ValueError("NaN values remain in DataFrame after filling missing values!")
            return X.to_numpy()
        else:
            # NumPy array için: Eksik değerleri sütun ortalamasıyla doldur
            if np.isnan(X).any():
                col_means = np.nanmean(X, axis=0)  # NaN'ları yok sayarak sütun ortalamalarını hesapla
                inds = np.where(np.isnan(X))       # NaN'ların indekslerini bul
                X[inds] = np.take(col_means, inds[1])  # NaN'ları ilgili sütun ortalamalarıyla doldur
            # Doldurduktan sonra hala NaN varsa hata fırlat (beklenmez ama güvenlik için)
            if np.isnan(X).any():
                raise ValueError("NaN values remain in NumPy array after filling missing values!")
            return X

    def apply_scaling(self):
        """Apply selected scaling method to the data"""
        scaling_method = self.scaling_combo.currentText()
        if scaling_method != "No Scaling":
            # Sabit sütunları kontrol et ve hariç tut
            variances = np.var(self.X_train, axis=0)
            non_constant_cols = variances > 0
            if not np.all(non_constant_cols):
                self.X_train = self.X_train[:, non_constant_cols]
                self.X_test = self.X_test[:, non_constant_cols]
                if self.X_valid is not None:
                    self.X_valid = self.X_valid[:, non_constant_cols]
                self.status_bar.showMessage("Warning: Constant features removed to prevent NaN during scaling.")
            
            if scaling_method == "Standard Scaling":
                scaler = preprocessing.StandardScaler()
            elif scaling_method == "Min-Max Scaling":
                scaler = preprocessing.MinMaxScaler()
            elif scaling_method == "Robust Scaling":
                scaler = preprocessing.RobustScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)
            if self.X_valid is not None:
                self.X_valid = scaler.transform(self.X_valid)
            
            if np.isnan(self.X_train).any() or np.isnan(self.X_test).any() or (self.X_valid is not None and np.isnan(self.X_valid).any()):
                raise ValueError("Scaling introduced NaN values despite checks!")

    def create_tabs(self):
        """Create tabs for different ML topics"""
        self.tab_widget = QTabWidget()
        tabs = [
            ("Classical ML", self.create_classical_ml_tab),
            ("Deep Learning", self.create_deep_learning_tab),
            ("Dimensionality Reduction", self.create_dim_reduction_tab),
            ("Reinforcement Learning", self.create_rl_tab),
            ("Dimensionality Reduction & Clustering", self.create_dim_reduction_clustering_tab)  # Yeni sekme
        ]
        for tab_name, create_func in tabs:
            scroll = QScrollArea()
            tab_widget = create_func()
            scroll.setWidget(tab_widget)
            scroll.setWidgetResizable(True)
            self.tab_widget.addTab(scroll, tab_name)
        self.layout.addWidget(self.tab_widget)

    def create_dim_reduction_clustering_tab(self):
        """Create the dimensionality reduction and clustering tab"""
        # Create main widget and grid layout
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # PCA Group: Parameter inputs and buttons
        pca_group = QGroupBox("Principal Component Analysis (PCA)")
        pca_layout = QVBoxLayout()
        pca_params = self.create_algorithm_group("PCA",
            {"n_components": "int", "whiten": "checkbox"},
            [])
        pca_layout.addWidget(pca_params)
        pca_visualize_btn = QPushButton("Visualize Explained Variance")
        pca_visualize_btn.clicked.connect(self.visualize_pca_variance)
        pca_layout.addWidget(pca_visualize_btn)
        pca_eigen_btn = QPushButton("Compute Eigenvectors")
        pca_eigen_btn.clicked.connect(self.compute_eigenvectors)
        pca_layout.addWidget(pca_eigen_btn)
        pca_group.setLayout(pca_layout)
        layout.addWidget(pca_group, 0, 0)
        
        # LDA Group: Parameter inputs and buttons
        lda_group = QGroupBox("Linear Discriminant Analysis (LDA)")
        lda_layout = QVBoxLayout()
        lda_params = self.create_algorithm_group("LDA",
            {"n_components": "int"},
            [])
        lda_layout.addWidget(lda_params)
        lda_metrics_btn = QPushButton("Show Class Separation Metrics")
        lda_metrics_btn.clicked.connect(self.show_lda_metrics)
        lda_layout.addWidget(lda_metrics_btn)
        lda_group.setLayout(lda_layout)
        layout.addWidget(lda_group, 0, 1)
        
        # K-Means Group: Parameters, existing, and new buttons
        kmeans_group = QGroupBox("K-Means Clustering")
        kmeans_layout = QVBoxLayout()
        kmeans_params = self.create_algorithm_group("K-Means",
            {"n_clusters": "int", "max_iter": "int", "n_init": "int"},
            ["MSE"])
        kmeans_layout.addWidget(kmeans_params)
        kmeans_elbow_btn = QPushButton("Show Elbow Plot")
        kmeans_elbow_btn.clicked.connect(self.show_kmeans_elbow)
        kmeans_layout.addWidget(kmeans_elbow_btn)
        # New: Add Silhouette Score button
        kmeans_silhouette_btn = QPushButton("Show Silhouette Score")
        kmeans_silhouette_btn.clicked.connect(self.show_silhouette_score)
        kmeans_layout.addWidget(kmeans_silhouette_btn)
        kmeans_group.setLayout(kmeans_layout)
        layout.addWidget(kmeans_group, 1, 0)

        
        # t-SNE Group: Parameters and projection button
        tsne_group = QGroupBox("t-SNE")
        tsne_layout = QVBoxLayout()
        tsne_params = self.create_algorithm_group("t-SNE",
            {"n_components": "int", "perplexity": "double", "n_iter": "int"},
            [])
        tsne_layout.addWidget(tsne_params)
        tsne_visualize_btn = QPushButton("Visualize 2D/3D Projection")
        tsne_visualize_btn.clicked.connect(self.visualize_tsne)
        tsne_layout.addWidget(tsne_visualize_btn)
        tsne_group.setLayout(tsne_layout)
        layout.addWidget(tsne_group, 1, 1)

        # UMAP Group: Parameters and projection button
        umap_group = QGroupBox("UMAP")
        umap_layout = QVBoxLayout()
        umap_params = self.create_algorithm_group("UMAP",
            {"n_components": "int", "n_neighbors": "int", "min_dist": "double"},
            [])
        umap_layout.addWidget(umap_params)
        umap_visualize_btn = QPushButton("Visualize 2D/3D Projection")
        umap_visualize_btn.clicked.connect(self.visualize_umap)
        umap_layout.addWidget(umap_visualize_btn)
        umap_group.setLayout(umap_layout)
        layout.addWidget(umap_group, 2, 0)
        
        return widget

    def visualize_pca_variance(self):
        """Visualize explained variance ratio for PCA"""
        try:
            if self.X_train is None:
                raise ValueError("No training data loaded!")
            pca = PCA()
            pca.fit(self.X_train)
            explained_variance = pca.explained_variance_ratio_
            
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(np.cumsum(explained_variance), marker='o')
            ax.set_xlabel("Number of Components")
            ax.set_ylabel("Cumulative Explained Variance")
            ax.set_title("PCA Explained Variance")
            ax.grid(True)
            self.canvas.draw()
            self.status_bar.showMessage("PCA explained variance plot generated")
        except Exception as e:
            self.show_error(f"Error visualizing PCA variance: {str(e)}")

    def compute_eigenvectors(self):
        try:
            cov_matrix = np.array([[5, 2], [2, 3]])
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            max_eigenvalue_idx = np.argmax(eigenvalues)
            principal_eigenvector = eigenvectors[:, max_eigenvalue_idx]
            metrics_text = f"Covariance Matrix Eigenvectors:\n\n"
            metrics_text += f"Eigenvalues: {eigenvalues}\n"
            metrics_text += f"Principal Eigenvector (for 1D projection): {principal_eigenvector}\n"
            self.metrics_text.setText(metrics_text)
            self.status_bar.showMessage("Eigenvectors computed for covariance matrix")
        except Exception as e:
            self.show_error(f"Error computing eigenvectors: {str(e)}")       

    def show_lda_metrics(self):
        """Show class separation metrics for LDA"""
        try:
            if self.X_train is None or self.y_train is None:
                raise ValueError("No training data or labels loaded!")
            n_components = min(self.X_train.shape[1], len(np.unique(self.y_train))-1)
            lda = LinearDiscriminantAnalysis(n_components=n_components)
            X_lda = lda.fit_transform(self.X_train, self.y_train)
            
            # Basit bir metrik: Sınıf ayrımı için varyans oranı
            metrics_text = "LDA Class Separation Metrics:\n\n"
            explained_variance = lda.explained_variance_ratio_
            metrics_text += f"Explained Variance Ratios: {explained_variance}\n"
            self.metrics_text.setText(metrics_text)
            self.status_bar.showMessage("LDA metrics displayed")
        except Exception as e:
            self.show_error(f"Error showing LDA metrics: {str(e)}")

    def show_kmeans_elbow(self):
        """Show elbow plot for K-Means clustering"""
        try:
            if self.X_train is None:
                raise ValueError("No training data loaded!")
            inertias = []
            K = range(1, 11)
            for k in K:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(self.X_train)
                inertias.append(kmeans.inertia_)
            
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(K, inertias, marker='o')
            ax.set_xlabel("Number of Clusters (k)")
            ax.set_ylabel("Inertia")
            ax.set_title("K-Means Elbow Plot")
            ax.grid(True)
            self.canvas.draw()
            self.status_bar.showMessage("K-Means elbow plot generated")
        except Exception as e:
            self.show_error(f"Error showing K-Means elbow plot: {str(e)}")

    def visualize_tsne(self):
        """Visualize 2D/3D t-SNE projection"""
        # For error management > try-except block
        try:
            if self.X_train is None:
                raise ValueError("No training data loaded!")
            # default parameters for t-SNE
            perplexity = 30.0  # the sensibility of the neighbour of t-SNE
            n_components = 2   # 2D projection
            # to make up t-SNE model and to turn data
            tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
            X_tsne = tsne.fit_transform(self.X_train)
            
            self.figure.clear()
            # to make up 2D veya 3D graph by using Plotly
            if n_components == 2:
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=self.y_train if self.y_train is not None else 'b')
                ax.set_xlabel("t-SNE Component 1")
                ax.set_ylabel("t-SNE Component 2")
                ax.set_title("t-SNE 2D Projection")
                if self.y_train is not None:
                    self.figure.colorbar(scatter)
            else:
                ax = self.figure.add_subplot(111, projection='3d')
                scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=self.y_train if self.y_train is not None else 'b')
                ax.set_xlabel("t-SNE Component 1")
                ax.set_ylabel("t-SNE Component 2")
                ax.set_zlabel("t-SNE Component 3")
                ax.set_title("t-SNE 3D Projection")
                if self.y_train is not None:
                    self.figure.colorbar(scatter)
            self.canvas.draw()
            self.status_bar.showMessage(f"t-SNE {n_components}D projection generated")
        except Exception as e:
            self.show_error(f"Error visualizing t-SNE: {str(e)}")

    def create_classical_ml_tab(self):
        """Create the classical machine learning algorithms tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
    
        regression_group = QGroupBox("Regression")
        regression_layout = QVBoxLayout()
    
        # Linear Regression için tüm parametreler
        lr_group = self.create_algorithm_group("Linear Regression", 
            {
                "learning_rate": "double", 
                "n_iterations": "int", 
                "fit_intercept": "checkbox", 
                "optimize_weights": "checkbox", 
                "weight_learning_rate": "double",
                "sample_weights": "custom"
            }, 
            ["MSE", "MAE", "Huber Loss"])
        regression_layout.addWidget(lr_group)
    
        svm_reg_group = self.create_algorithm_group("SVM Regression (SVR)",
            {"C": "double", "epsilon": "double", "kernel": ["linear", "rbf", "poly"]},
            ["MSE", "MAE", "Huber Loss"])
        regression_layout.addWidget(svm_reg_group)
    
        regression_group.setLayout(regression_layout)
        layout.addWidget(regression_group, 0, 0)
    
        classification_group = QGroupBox("Classification")
        classification_layout = QVBoxLayout()
    
        logistic_group = self.create_algorithm_group("Logistic Regression",
            {"C": "double", "max_iter": "int", "multi_class": ["ovr", "multinomial"]},
            ["Cross-Entropy", "Hinge Loss"])
        classification_layout.addWidget(logistic_group)
    
        nb_group = self.create_algorithm_group("Naive Bayes",
            {"var_smoothing": "double", "priors": "custom"},
            ["Cross-Entropy"])
        classification_layout.addWidget(nb_group)
    
        svm_group = self.create_algorithm_group("Support Vector Machine",
            {"C": "double", "kernel": ["linear", "rbf", "poly"], "degree": "int"},
            ["Cross-Entropy", "Hinge Loss"])
        classification_layout.addWidget(svm_group)
    
        dt_group = self.create_algorithm_group("Decision Tree",
            {"max_depth": "int", "min_samples_split": "int", "criterion": ["gini", "entropy"]},
            ["Cross-Entropy"])
        classification_layout.addWidget(dt_group)
    
        rf_group = self.create_algorithm_group("Random Forest",
            {"n_estimators": "int", "max_depth": "int", "min_samples_split": "int"},
            ["Cross-Entropy"])
        classification_layout.addWidget(rf_group)
    
        knn_group = self.create_algorithm_group("K-Nearest Neighbors",
            {"n_neighbors": "int", "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan"]},
            ["Cross-Entropy"])
        classification_layout.addWidget(knn_group)
    
        classification_group.setLayout(classification_layout)
        layout.addWidget(classification_group, 0, 1)
    
        return widget

    def create_algorithm_group(self, name, params, loss_options=None):
        """Helper method to create algorithm parameter groups"""
        group = QGroupBox(name)
        layout = QVBoxLayout()
        param_widgets = {}
        
        for param_name, param_type in params.items():
            param_layout = QHBoxLayout()
            param_layout.addWidget(QLabel(f"{param_name}:"))
            if param_type == "int":
                widget = QSpinBox()
                widget.setRange(1, 10000)
                widget.setValue(1000 if param_name == "n_iterations" else 1)
            elif param_type == "double":
                widget = QDoubleSpinBox()
                widget.setRange(0.0001, 1.0)
                widget.setSingleStep(0.01)
                widget.setValue(0.01 if param_name == "learning_rate" else 0.1)
            elif param_type == "checkbox":
                widget = QCheckBox()
            elif param_type == "custom":
                widget = QLineEdit()
                if param_name == "priors":
                    widget.setPlaceholderText("e.g., 0.3,0.7")
                    widget.setToolTip("Enter comma-separated priors (e.g., 0.3,0.7)")
                elif param_name == "sample_weights":
                    widget.setPlaceholderText("e.g., 1.0,0.5,2.0")
                    widget.setToolTip("Enter comma-separated sample weights (e.g., 1.0,0.5,2.0)")
                    auto_fill_btn = QPushButton("Auto Fill Weights")
                    auto_fill_btn.clicked.connect(lambda: self.auto_fill_weights(widget))
                    param_layout.addWidget(auto_fill_btn)
            elif isinstance(param_type, list):
                widget = QComboBox()
                widget.addItems(param_type)
            param_layout.addWidget(widget)
            param_widgets[param_name] = widget
            layout.addLayout(param_layout)
        
        if loss_options:
            loss_layout = QHBoxLayout()
            loss_combo = QComboBox()
            loss_combo.addItems(loss_options)
            param_widgets["loss"] = loss_combo
            loss_layout.addWidget(QLabel("Loss Function:"))
            loss_layout.addWidget(loss_combo)
            layout.addLayout(loss_layout)
        
        train_btn = QPushButton(f"Train {name}")
        train_btn.clicked.connect(lambda: self.train_model(name, param_widgets))
        layout.addWidget(train_btn)
        
        # Yeni: K-Fold Cross-Validation seçeneği
        kfold_layout = QHBoxLayout()
        kfold_label = QLabel("K-Fold CV (k):")
        kfold_spin = QSpinBox()
        kfold_spin.setRange(2, 10)
        kfold_spin.setValue(5)
        kfold_btn = QPushButton(f"Run {name} with K-Fold CV")
        kfold_btn.clicked.connect(lambda: self.run_kfold_cv(name, param_widgets, kfold_spin.value()))
        kfold_layout.addWidget(kfold_label)
        kfold_layout.addWidget(kfold_spin)
        kfold_layout.addWidget(kfold_btn)
        layout.addLayout(kfold_layout)
        
        group.setLayout(layout)
        return group
    
    def select_features(self, columns):
        """Dialog to select features from dataset columns"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Features")
        layout = QVBoxLayout(dialog)
        
        feature_checkboxes = {}
        for col in columns:
            checkbox = QCheckBox(col)
            checkbox.setChecked(True)
            feature_checkboxes[col] = checkbox
            layout.addWidget(checkbox)
        
        btn_layout = QHBoxLayout()
        select_btn = QPushButton("Select Features")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(select_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        selected_features = []
        
        def on_select():
            selected_features.clear()
            for col, checkbox in feature_checkboxes.items():
                if checkbox.isChecked():
                    selected_features.append(col)
            dialog.accept()
        
        select_btn.clicked.connect(on_select)
        cancel_btn.clicked.connect(dialog.reject)
        
        if dialog.exec() == QDialog.DialogCode.Accepted and selected_features:
            return selected_features
        return None

    def auto_fill_weights(self, widget):
        """Automatically fill sample weights with 1.0 based on training data size"""
        if self.X_train is None:
            self.show_error("Please load a dataset first!")
            return
        n_samples = len(self.X_train)
        weights = [1.0] * n_samples
        weights_text = ",".join(map(str, weights))
        widget.setText(weights_text)
        self.status_bar.showMessage(f"Auto-filled {n_samples} sample weights with 1.0")

    def run_kfold_cv(self, model_name, param_widgets, k):
        """Run k-fold cross-validation for the selected model with fold size reporting"""
        try:
            if self.X_train is None or self.y_train is None:
                raise ValueError("No training data loaded!")
        
            # Veriyi birleştir (X_train + X_test)
            X = np.vstack((self.X_train, self.X_test))
            y = np.hstack((self.y_train, self.y_test))
        
            # Parametreleri al
            params = {}
            sample_weights = None
            loss_function = param_widgets.get("loss", None)
            if loss_function:
                loss_function = loss_function.currentText()
            for param_name, widget in param_widgets.items():
                if param_name == "loss":
                    continue
                if isinstance(widget, QSpinBox):
                    params[param_name] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    params[param_name] = widget.value()
                elif isinstance(widget, QComboBox):
                    params[param_name] = widget.currentText()
                elif isinstance(widget, QCheckBox):
                    params[param_name] = widget.isChecked()
                elif isinstance(widget, QLineEdit) and param_name == "priors":
                    priors_text = widget.text()
                    if priors_text:
                        params["priors"] = [float(p) for p in priors_text.split(",")]
                    else:
                        params["priors"] = None
                elif isinstance(widget, QLineEdit) and param_name == "sample_weights":
                    weights_text = widget.text()
                    if weights_text:
                        try:
                            sample_weights = [float(w.strip()) for w in weights_text.split(",")]
                            if len(sample_weights) != len(X):
                                raise ValueError(f"Number of sample weights ({len(sample_weights)}) must match number of samples ({len(X)})")
                            if np.isnan(sample_weights).any():
                                raise ValueError("Sample weights contain NaN values from input!")
                        except ValueError as e:
                            raise ValueError(f"Invalid sample weights format: {str(e)}")
                    else:
                        sample_weights = None
        
            # K-Fold Cross-Validation
            kf = model_selection.KFold(n_splits=k, shuffle=True, random_state=42)
            scores = {"accuracy": [], "mse": [], "rmse": []}
            fold_sizes = []
        
            for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                X_train_fold, X_test_fold = X[train_idx], X[test_idx]
                y_train_fold, y_test_fold = y[train_idx], y[test_idx]
                fold_sizes.append(len(test_idx))
            
                # Modeli seç
                if model_name == "Linear Regression":
                    model = CustomGradientDescentLR(
                        learning_rate=params.get("learning_rate", 0.01),
                        n_iterations=params.get("n_iterations", 1000),
                        fit_intercept=params.get("fit_intercept", True),
                        optimize_weights=params.get("optimize_weights", False),
                        weight_learning_rate=params.get("weight_learning_rate", 0.01)
                    )
                elif model_name == "SVM Regression (SVR)":
                    model = SVR(**params)
                elif model_name == "Logistic Regression":
                    model = LogisticRegression(**params)
                elif model_name == "Naive Bayes":
                    model = GaussianNB(**params)
                elif model_name == "Support Vector Machine":
                    model = SVC(**params)
                elif model_name == "Decision Tree":
                    model = DecisionTreeClassifier(**params)
                elif model_name == "Random Forest":
                    model = RandomForestClassifier(**params)
                elif model_name == "K-Nearest Neighbors":
                    model = KNeighborsClassifier(**params)
                elif model_name == "K-Means":
                    model = KMeans(**params)
                elif model_name == "PCA":
                    model = PCA(**params)
            
                # Modeli eğit
                if model_name in ["K-Means", "PCA"]:
                    model.fit(X_train_fold)
                    y_pred = model.predict(X_train_fold) if model_name == "K-Means" else model.transform(X_train_fold)
                else:
                    model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights[train_idx] if sample_weights is not None else None)
                    y_pred = model.predict(X_test_fold)
            
                # Metrikleri hesapla
                if loss_function in ["MSE", "MAE", "Huber Loss"]:
                    mse = mean_squared_error(y_test_fold, y_pred)
                    rmse = np.sqrt(mse)
                    scores["mse"].append(mse)
                    scores["rmse"].append(rmse)
                else:
                    accuracy = accuracy_score(y_test_fold, y_pred)
                    scores["accuracy"].append(accuracy)
        
            # Sonuçları raporla
            metrics_text = f"K-Fold Cross-Validation (k={k}) Results:\n\n"
            metrics_text += f"Fold Sizes: {fold_sizes}\n"
            if len(X) == 100 and k == 5:
                expected_sizes = [20, 20, 20, 20, 20]
                metrics_text += f"Expected Fold Sizes for 100 samples, k=5: {expected_sizes}\n"
                metrics_text += f"Fold Size Match: {'OK' if fold_sizes == expected_sizes else 'Mismatch'}\n"
            for metric, values in scores.items():
                if values:
                    mean_score = np.mean(values)
                    std_score = np.std(values)
                    metrics_text += f"{metric.upper()} - Mean: {mean_score:.4f}, Std: {std_score:.4f}\n"
            self.metrics_text.setText(metrics_text)
            self.status_bar.showMessage(f"K-Fold CV for {model_name} completed")
        except Exception as e:
            self.show_error(f"Error running K-Fold CV: {str(e)}")

    def train_model(self, model_name, param_widgets):
        """Train the selected machine learning model"""
        try:
            if self.X_train is None or self.y_train is None:
                raise ValueError("No training data loaded! Please load a dataset first.")
            
            if np.isnan(self.X_train).any():
                raise ValueError("X_train contains NaN values!")
            if np.isnan(self.y_train).any():
                raise ValueError("y_train contains NaN values!")
            
            class CustomGradientDescentLR:
                def __init__(self, learning_rate=0.01, n_iterations=1000, fit_intercept=True, optimize_weights=False, weight_learning_rate=0.01):
                    self.learning_rate = learning_rate
                    self.n_iterations = n_iterations
                    self.fit_intercept = fit_intercept
                    self.optimize_weights = optimize_weights
                    self.weight_learning_rate = weight_learning_rate
                    self.weights = None
                    self.bias = None
                    self.sample_weights = None

                def fit(self, X, y, sample_weight=None):
                    n_samples, n_features = X.shape
                    
                    if self.fit_intercept:
                        self.weights = np.zeros(n_features)
                        self.bias = 0
                    else:
                        self.weights = np.zeros(n_features)
                        self.bias = 0
                    
                    if sample_weight is None or self.optimize_weights:
                        self.sample_weights = np.ones(n_samples)
                    else:
                        self.sample_weights = np.array(sample_weight)
                        if np.isnan(self.sample_weights).any():
                            raise ValueError("Initial sample_weights contain NaN values!")
                    
                    for i in range(self.n_iterations):
                        y_pred = self.predict(X)
                        error = y_pred - y
                        
                        if self.fit_intercept:
                            grad_w = (1 / n_samples) * np.dot(X.T, error * self.sample_weights)
                            grad_b = (1 / n_samples) * np.sum(error * self.sample_weights)
                            self.weights -= self.learning_rate * grad_w
                            self.bias -= self.learning_rate * grad_b
                        else:
                            grad_w = (1 / n_samples) * np.dot(X.T, error * self.sample_weights)
                            self.weights -= self.learning_rate * grad_w
                        
                        if self.optimize_weights:
                            grad_sample_weights = error * (y_pred - y)
                            self.sample_weights -= self.weight_learning_rate * grad_sample_weights
                            if np.isnan(self.sample_weights).any():
                                raise ValueError(f"NaN detected in sample_weights at iteration {i}!")
                            self.sample_weights = np.clip(self.sample_weights, 0.1, 10.0)

                def predict(self, X):
                    if self.fit_intercept:
                        return np.dot(X, self.weights) + self.bias
                    else:
                        return np.dot(X, self.weights)

            params = {}
            sample_weights = None
            loss_function = param_widgets.get("loss", None)
            if loss_function:
                loss_function = loss_function.currentText()
            for param_name, widget in param_widgets.items():
                if param_name == "loss":
                    continue
                if isinstance(widget, QSpinBox):
                    params[param_name] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    params[param_name] = widget.value()
                elif isinstance(widget, QComboBox):
                    params[param_name] = widget.currentText()
                elif isinstance(widget, QCheckBox):
                    params[param_name] = widget.isChecked()
                elif isinstance(widget, QLineEdit) and param_name == "priors":
                    priors_text = widget.text()
                    if priors_text:
                        params["priors"] = [float(p) for p in priors_text.split(",")]
                    else:
                        params["priors"] = None
                elif isinstance(widget, QLineEdit) and param_name == "sample_weights":
                    weights_text = widget.text()
                    if weights_text:
                        try:
                            sample_weights = [float(w.strip()) for w in weights_text.split(",")]
                            if len(sample_weights) != len(self.X_train):
                                raise ValueError(f"Number of sample weights ({len(sample_weights)}) must match number of training samples ({len(self.X_train)})")
                            if np.isnan(sample_weights).any():
                                raise ValueError("Sample weights contain NaN values from input!")
                        except ValueError as e:
                            raise ValueError(f"Invalid sample weights format: {str(e)}. Use comma-separated numbers (e.g., 1.0, 0.5, 2.0)")
                    else:
                        sample_weights = None

            if model_name == "Linear Regression":
                self.current_model = CustomGradientDescentLR(
                    learning_rate=params.get("learning_rate", 0.01),
                    n_iterations=params.get("n_iterations", 1000),
                    fit_intercept=params.get("fit_intercept", True),
                    optimize_weights=params.get("optimize_weights", False),
                    weight_learning_rate=params.get("weight_learning_rate", 0.01)
                )
                self.current_model.fit(self.X_train, self.y_train, sample_weight=sample_weights)
                if params.get("optimize_weights", False):
                    self.status_bar.showMessage(f"Optimized sample weights: {self.current_model.sample_weights[:5]}... (first 5 shown)")
            elif model_name == "SVM Regression (SVR)":
                self.current_model = SVR(**params)
                self.current_model.fit(self.X_train, self.y_train)
            elif model_name == "Logistic Regression":
                self.current_model = LogisticRegression(**params)
                self.current_model.fit(self.X_train, self.y_train)
            elif model_name == "Naive Bayes":
                self.current_model = GaussianNB(**params)
                self.current_model.fit(self.X_train, self.y_train)
            elif model_name == "Support Vector Machine":
                self.current_model = SVC(**params)
                self.current_model.fit(self.X_train, self.y_train)
            elif model_name == "Decision Tree":
                self.current_model = DecisionTreeClassifier(**params)
                self.current_model.fit(self.X_train, self.y_train)
            elif model_name == "Random Forest":
                self.current_model = RandomForestClassifier(**params)
                self.current_model.fit(self.X_train, self.y_train)
            elif model_name == "K-Nearest Neighbors":
                self.current_model = KNeighborsClassifier(**params)
                self.current_model.fit(self.X_train, self.y_train)
            elif model_name == "K-Means":
                self.current_model = KMeans(**params)
                self.current_model.fit(self.X_train)
            elif model_name == "PCA":
                self.current_model = PCA(**params)
                self.current_model.fit(self.X_train)
            elif model_name == "LDA":
                self.current_model = LinearDiscriminantAnalysis(**params)
                self.current_model.fit(self.X_train, self.y_train)
            elif model_name == "t-SNE":
                self.current_model = TSNE(**params)
                self.current_model.fit(self.X_train)

            y_pred = self.current_model.predict(self.X_test) if model_name not in ["PCA", "LDA", "t-SNE"] else self.current_model.transform(self.X_test)
            
            self.update_visualization(y_pred)
            self.update_metrics(y_pred, loss_function)
            self.status_bar.showMessage(f"{model_name} training complete")
            
            if model_name == "SVM Regression (SVR)" and self.dataset_combo.currentText() == "Boston Housing Dataset":
                mse = mean_squared_error(self.y_test, y_pred)
                mae = mean_absolute_error(self.y_test, y_pred)
                self.metrics_text.append(f"\nSVR on Boston Housing - MSE: {mse:.4f}, MAE: {mae:.4f}")

        except Exception as e:
            self.show_error(f"Error training {model_name}: {str(e)}")

    def update_metrics(self, y_pred, loss_function=None):
        """Update metrics display"""
        metrics_text = f"Model Performance Metrics{' (Loss: ' + loss_function + ')' if loss_function else ''}:\n\n"
        if loss_function in ["MSE", "MAE", "Huber Loss"]:
            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            metrics_text += f"Mean Squared Error: {mse:.4f}\n"
            metrics_text += f"Mean Absolute Error: {mae:.4f}\n"
        else:
            accuracy = accuracy_score(self.y_test, y_pred)
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            metrics_text += f"Accuracy: {accuracy:.4f}\n\n"
            metrics_text += "Confusion Matrix:\n"
            metrics_text += str(conf_matrix)
        self.metrics_text.setText(metrics_text)

    def create_deep_learning_tab(self):
        """Create the deep learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        mlp_group = QGroupBox("Multi-Layer Perceptron")
        mlp_layout = QVBoxLayout()
        self.layer_config = []
        layer_btn = QPushButton("Add Layer")
        layer_btn.clicked.connect(self.add_layer_dialog)
        mlp_layout.addWidget(layer_btn)
        
        training_params_group = self.create_training_params_group()
        mlp_layout.addWidget(training_params_group)
        
        train_btn = QPushButton("Train Neural Network")
        train_btn.clicked.connect(self.train_neural_network)
        mlp_layout.addWidget(train_btn)
        
        mlp_group.setLayout(mlp_layout)
        layout.addWidget(mlp_group, 0, 0)
        
        cnn_group = QGroupBox("Convolutional Neural Network")
        cnn_layout = QVBoxLayout()
        cnn_controls = self.create_cnn_controls()
        cnn_layout.addWidget(cnn_controls)
        cnn_group.setLayout(cnn_layout)
        layout.addWidget(cnn_group, 0, 1)
        
        rnn_group = QGroupBox("Recurrent Neural Network")
        rnn_layout = QVBoxLayout()
        rnn_controls = self.create_rnn_controls()
        rnn_layout.addWidget(rnn_controls)
        rnn_group.setLayout(rnn_layout)
        layout.addWidget(rnn_group, 1, 0)
        
        return widget

    def create_training_params_group(self):
        """Create group for neural network training parameters"""
        group = QGroupBox("Training Parameters")
        layout = QVBoxLayout()
        
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1000)
        self.batch_size_spin.setValue(32)
        batch_layout.addWidget(self.batch_size_spin)
        layout.addLayout(batch_layout)
        
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        epochs_layout.addWidget(self.epochs_spin)
        layout.addLayout(epochs_layout)
        
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setSingleStep(0.001)
        lr_layout.addWidget(self.lr_spin)
        layout.addLayout(lr_layout)
        
        loss_layout = QHBoxLayout()
        self.nn_loss_combo = QComboBox()
        self.nn_loss_combo.addItems(["Cross-Entropy", "MSE", "MAE", "Huber Loss"])
        loss_layout.addWidget(QLabel("Loss Function:"))
        loss_layout.addWidget(self.nn_loss_combo)
        layout.addLayout(loss_layout)
        
        group.setLayout(layout)
        return group

    def create_cnn_controls(self):
        """Create controls for Convolutional Neural Network"""
        group = QGroupBox("CNN Architecture")
        layout = QVBoxLayout()
        label = QLabel("CNN Controls (To be implemented)")
        layout.addWidget(label)
        group.setLayout(layout)
        return group

    def create_rnn_controls(self):
        """Create controls for Recurrent Neural Network"""
        group = QGroupBox("RNN Architecture")
        layout = QVBoxLayout()
        label = QLabel("RNN Controls (To be implemented)")
        layout.addWidget(label)
        group.setLayout(layout)
        return group

    def train_neural_network(self):
        """Train the neural network with current configuration"""
        if not self.layer_config:
            self.show_error("Please add at least one layer")
            return
        try:
            model = self.create_neural_network()
            batch_size = self.batch_size_spin.value()
            epochs = self.epochs_spin.value()
            learning_rate = self.lr_spin.value()
            loss_function = self.nn_loss_combo.currentText().lower().replace(" ", "_")
            
            if len(self.X_train.shape) == 1:
                X_train = self.X_train.reshape(-1, 1)
                X_test = self.X_test.reshape(-1, 1)
            else:
                X_train = self.X_train
                X_test = self.X_test
            
            y_train = tf.keras.utils.to_categorical(self.y_train) if "cross_entropy" in loss_function else self.y_train
            y_test = tf.keras.utils.to_categorical(self.y_test) if "cross_entropy" in loss_function else self.y_test
            
            optimizer = optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy' if "cross_entropy" in loss_function else 'mae'])
            history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                              validation_data=(X_test, y_test), callbacks=[self.create_progress_callback()])
            self.plot_training_history(history)
            self.status_bar.showMessage("Neural Network Training Complete")
        except Exception as e:
            self.show_error(f"Error training neural network: {str(e)}")

    def create_neural_network(self):
        """Create neural network based on current configuration"""
        model = models.Sequential()
        for layer_config in self.layer_config:
            layer_type = layer_config["type"]
            params = layer_config["params"]
            if layer_type == "Dense":
                model.add(layers.Dense(**params))
            elif layer_type == "Conv2D":
                if len(model.layers) == 0:
                    params['input_shape'] = self.X_train.shape[1:]
                model.add(layers.Conv2D(**params))
            elif layer_type == "MaxPooling2D":
                model.add(layers.MaxPooling2D())
            elif layer_type == "Flatten":
                model.add(layers.Flatten())
            elif layer_type == "Dropout":
                model.add(layers.Dropout(**params))
        num_classes = len(np.unique(self.y_train))
        activation = 'softmax' if self.nn_loss_combo.currentText() == "Cross-Entropy" else 'linear'
        model.add(layers.Dense(num_classes if self.nn_loss_combo.currentText() == "Cross-Entropy" else 1, activation=activation))
        return model

    def create_dim_reduction_tab(self):
        """Create the dimensionality reduction tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        kmeans_group = QGroupBox("K-Means Clustering")
        kmeans_layout = QVBoxLayout()
        kmeans_params = self.create_algorithm_group("K-Means Parameters",
            {"n_clusters": "int", "max_iter": "int", "n_init": "int"}, ["MSE"])
        kmeans_layout.addWidget(kmeans_params)
        kmeans_group.setLayout(kmeans_layout)
        layout.addWidget(kmeans_group, 0, 0)
        
        pca_group = QGroupBox("Principal Component Analysis")
        pca_layout = QVBoxLayout()
        pca_params = self.create_algorithm_group("PCA Parameters",
            {"n_components": "int", "whiten": "checkbox"}, [])
        pca_layout.addWidget(pca_params)
        pca_group.setLayout(pca_layout)
        layout.addWidget(pca_group, 0, 1)
        
        return widget

    def create_rl_tab(self):
        """Create the reinforcement learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        env_group = QGroupBox("Environment")
        env_layout = QVBoxLayout()
        self.env_combo = QComboBox()
        self.env_combo.addItems(["CartPole-v1", "MountainCar-v0", "Acrobot-v1"])
        env_layout.addWidget(self.env_combo)
        env_group.setLayout(env_layout)
        layout.addWidget(env_group, 0, 0)
        
        algo_group = QGroupBox("RL Algorithm")
        algo_layout = QVBoxLayout()
        self.rl_algo_combo = QComboBox()
        self.rl_algo_combo.addItems(["Q-Learning", "SARSA", "DQN"])
        algo_layout.addWidget(self.rl_algo_combo)
        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group, 0, 1)
        
        return widget

    def create_visualization(self):
        """Create the visualization section"""
        viz_group = QGroupBox("Visualization")
        viz_layout = QHBoxLayout()
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        viz_layout.addWidget(self.canvas)
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        viz_layout.addWidget(self.metrics_text)
        viz_group.setLayout(viz_layout)
        self.layout.addWidget(viz_group)

    def create_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)

    def update_visualization(self, y_pred):
        """Update the visualization with current results"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if len(np.unique(self.y_test)) > 10:
            ax.scatter(self.y_test, y_pred)
            ax.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
        else:
            if self.X_train.shape[1] > 2:
                pca = PCA(n_components=2)
                X_test_2d = pca.fit_transform(self.X_test)
                scatter = ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_pred, cmap='viridis')
                self.figure.colorbar(scatter)
            else:
                scatter = ax.scatter(self.X_test[:, 0], self.X_test[:, 1], c=y_pred, cmap='viridis')
                self.figure.colorbar(scatter)
        self.canvas.draw()

    def plot_training_history(self, history):
        """Plot neural network training history"""
        self.figure.clear()
        ax1 = self.figure.add_subplot(211)
        ax1.plot(history.history['accuracy' if 'accuracy' in history.history else 'mae'])
        ax1.plot(history.history['val_' + ('accuracy' if 'accuracy' in history.history else 'mae')])
        ax1.set_title('Model Accuracy' if 'accuracy' in history.history else 'Model Performance')
        ax1.set_ylabel('Accuracy' if 'accuracy' in history.history else 'MAE')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Test'])
        
        ax2 = self.figure.add_subplot(212)
        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Test'])
        self.figure.tight_layout()
        self.canvas.draw()

    def create_progress_callback(self):
        """Create callback for updating progress bar during training"""
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, progress_bar):
                super().__init__()
                self.progress_bar = progress_bar
            def on_epoch_end(self, epoch, logs=None):
                progress = int(((epoch + 1) / self.params['epochs']) * 100)
                self.progress_bar.setValue(progress)
        return ProgressCallback(self.progress_bar)

    def add_layer_dialog(self):
        """Open a dialog to add a neural network layer"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Neural Network Layer")
        layout = QVBoxLayout(dialog)
        
        type_layout = QHBoxLayout()
        type_label = QLabel("Layer Type:")
        type_combo = QComboBox()
        type_combo.addItems(["Dense", "Conv2D", "MaxPooling2D", "Flatten", "Dropout"])
        type_layout.addWidget(type_label)
        type_layout.addWidget(type_combo)
        layout.addLayout(type_layout)
        
        params_group = QGroupBox("Layer Parameters")
        params_layout = QVBoxLayout()
        self.layer_param_inputs = {}
        
        def update_params():
            for widget in list(self.layer_param_inputs.values()):
                params_layout.removeWidget(widget)
                widget.deleteLater()
            self.layer_param_inputs.clear()
            
            layer_type = type_combo.currentText()
            if layer_type == "Dense":
                units_label = QLabel("Units:")
                units_input = QSpinBox()
                units_input.setRange(1, 1000)
                units_input.setValue(32)
                self.layer_param_inputs["units"] = units_input
                
                activation_label = QLabel("Activation:")
                activation_combo = QComboBox()
                activation_combo.addItems(["relu", "sigmoid", "tanh", "softmax"])
                self.layer_param_inputs["activation"] = activation_combo
                
                params_layout.addWidget(units_label)
                params_layout.addWidget(units_input)
                params_layout.addWidget(activation_label)
                params_layout.addWidget(activation_combo)
            
            elif layer_type == "Conv2D":
                filters_label = QLabel("Filters:")
                filters_input = QSpinBox()
                filters_input.setRange(1, 1000)
                filters_input.setValue(32)
                self.layer_param_inputs["filters"] = filters_input
                
                kernel_label = QLabel("Kernel Size:")
                kernel_input = QLineEdit()
                kernel_input.setText("3, 3")
                self.layer_param_inputs["kernel_size"] = kernel_input
                
                params_layout.addWidget(filters_label)
                params_layout.addWidget(filters_input)
                params_layout.addWidget(kernel_label)
                params_layout.addWidget(kernel_input)
            
            elif layer_type == "Dropout":
                rate_label = QLabel("Dropout Rate:")
                rate_input = QDoubleSpinBox()
                rate_input.setRange(0.0, 1.0)
                rate_input.setValue(0.5)
                rate_input.setSingleStep(0.1)
                self.layer_param_inputs["rate"] = rate_input
                
                params_layout.addWidget(rate_label)
                params_layout.addWidget(rate_input)
        
        type_combo.currentIndexChanged.connect(update_params)
        update_params()
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Layer")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        def add_layer():
            layer_type = type_combo.currentText()
            layer_params = {}
            for param_name, widget in self.layer_param_inputs.items():
                if isinstance(widget, QSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QComboBox):
                    layer_params[param_name] = widget.currentText()
                elif isinstance(widget, QLineEdit):
                    if param_name == "kernel_size":
                        layer_params[param_name] = tuple(map(int, widget.text().split(',')))
            
            self.layer_config.append({"type": layer_type, "params": layer_params})
            dialog.accept()
        
        add_btn.clicked.connect(add_layer)
        cancel_btn.clicked.connect(dialog.reject)
        dialog.exec()

    def select_target_column(self, columns):
        """Dialog to select target column from dataset"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Target Column")
        layout = QVBoxLayout(dialog)
        combo = QComboBox()
        combo.addItems(columns)
        layout.addWidget(combo)
        btn = QPushButton("Select")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return combo.currentText()
        return None

    def show_error(self, message):
        """Show error message dialog"""
        QMessageBox.critical(self, "Error", message)

    def show_silhouette_score(self):
        """Calculate and display silhouette score for K-Means clustering"""
        # Use try-except block for error handling
        try:
            if self.X_train is None:
                raise ValueError("No training data loaded!")
            # Default number of clusters for K-Means (can be retrieved from GUI)
            n_clusters = 3
            # Create and fit K-Means model to the data
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(self.X_train)
            # Get cluster labels
            labels = kmeans.labels_
            # Calculate silhouette score
            silhouette_avg = silhouette_score(self.X_train, labels)
            # Prepare text to display in the metrics area
            metrics_text = f"K-Means Silhouette Score (n_clusters={n_clusters}):\n\n"
            metrics_text += f"Average Silhouette Score: {silhouette_avg:.4f}\n"
            # Display text in the metrics area
            self.metrics_text.setText(metrics_text)
            self.status_bar.showMessage("Silhouette score calculated for K-Means")
        except Exception as e:
            self.show_error(f"Error calculating silhouette score: {str(e)}")

    def visualize_tsne(self):
        """Visualize 2D/3D t-SNE projection using Plotly"""
        try:
            if self.X_train is None:
                raise ValueError("No training data loaded!")
            n_components = 2  # Varsayılan, GUI'den alınabilir
            perplexity = 30.0  # Varsayılan, GUI'den alınabilir
            tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
            X_tsne = tsne.fit_transform(self.X_train)
            
            if n_components == 2:
                fig = px.scatter(
                    x=X_tsne[:, 0],
                    y=X_tsne[:, 1],
                    color=self.y_train if self.y_train is not None else None,
                    title="t-SNE 2D Projection",
                    labels={"x": "t-SNE Component 1", "y": "t-SNE Component 2"}
                )
            else:
                fig = px.scatter_3d(
                    x=X_tsne[:, 0],
                    y=X_tsne[:, 1],
                    z=X_tsne[:, 2],
                    color=self.y_train if self.y_train is not None else None,
                    title="t-SNE 3D Projection",
                    labels={"x": "t-SNE Component 1", "y": "t-SNE Component 2", "z": "t-SNE Component 3"}
                )
            
            fig.update_layout(showlegend=True)
            fig.update_xaxes(showgrid=True)
            fig.update_yaxes(showgrid=True)
            fig.update_traces(marker=dict(size=5))
            
            output_file = "tsne_plot.html"
            fig.write_html(output_file)
            webbrowser.open("file://" + os.path.realpath(output_file))
            self.status_bar.showMessage(f"t-SNE {n_components}D projection generated with Plotly")
        except Exception as e:
            self.show_error(f"Error visualizing t-SNE: {str(e)}")

    def visualize_umap(self):
        """Visualize 2D/3D UMAP projection using Plotly"""
        # Hata yönetimi için try-except bloğu
        try:
            if self.X_train is None:
                raise ValueError("No training data loaded!")
            n_components = 2  # 2D projeksiyon
            n_neighbors = 15  # Komşu sayısı
            min_dist = 0.1    # Minimum mesafe
            # UMAP modelini oluşumturma ve veriyi dönüştürme
            umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
            X_umap = umap_model.fit_transform(self.X_train)
            # 2D veya 3D grafiği Plotly ile oluşturma
            if n_components == 2:
                fig = px.scatter(
                    x=X_umap[:, 0],
                    y=X_umap[:, 1],
                    color=self.y_train if self.y_train is not None else None,
                    title="UMAP 2D Projection",
                    labels={"x": "UMAP Component 1", "y": "UMAP Component 2"}
                )
            else:
                fig = px.scatter_3d(
                    x=X_umap[:, 0],
                    y=X_umap[:, 1],
                    z=X_umap[:, 2],
                    color=self.y_train if self.y_train is not None else None,
                    title="UMAP 3D Projection",
                    labels={"x": "UMAP Component 1", "y": "UMAP Component 2", "z": "UMAP Component 3"}
                )

            # Grafik görünümünü özelleştirme
            fig.update_layout(showlegend=True)
            fig.update_xaxes(showgrid=True)
            fig.update_yaxes(showgrid=True)
            fig.update_traces(marker=dict(size=5))
            
            # Grafiği HTML dosyasına kaydet ve tarayıcıda açmak
            output_file = "umap_plot.html"
            fig.write_html(output_file)
            webbrowser.open("file://" + os.path.realpath(output_file))
            # Durum çubuğunda bilgilendirme mesajı göster
            self.status_bar.showMessage(f"UMAP {n_components}D projection generated with Plotly")
        except Exception as e:
            self.show_error(f"Error visualizing UMAP: {str(e)}")        

def main():
    """Main function to start the application"""
    app = QApplication(sys.argv)
    window = MLCourseGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()