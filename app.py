import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from io import BytesIO
import base64
from datetime import datetime

# ML and preprocessing imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, mutual_info_regression, mutual_info_classif
from scipy import stats
from scipy.stats import zscore

warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Smart Data Cleaning Assistant",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SmartDataCleaner:
    def __init__(self):
        self.df = None
        self.original_df = None
        self.cleaning_actions = []
        self.feature_importance = None
        self.target_column = None
        
    def load_data(self, uploaded_file):
        """Load CSV data from uploaded file"""
        try:
            self.df = pd.read_csv(uploaded_file)
            self.original_df = self.df.copy()
            return True, "Data loaded successfully!"
        except Exception as e:
            return False, f"Error loading data: {str(e)}"
    
    def perform_eda(self):
        """Perform automatic Exploratory Data Analysis"""
        if self.df is None:
            return {}
        
        eda_results = {
            'basic_info': {
                'shape': self.df.shape,
                'columns': list(self.df.columns),
                'dtypes': self.df.dtypes.to_dict(),
                'memory_usage': self.df.memory_usage(deep=True).sum()
            },
            'missing_values': self.df.isnull().sum().to_dict(),
            'missing_percentage': (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'summary_stats': self.df.describe().to_dict() if len(self.df.select_dtypes(include=[np.number]).columns) > 0 else {},
            'categorical_info': {},
            'outliers': {},
            'skewness': {},
            'correlation_issues': []
        }
        
        # Analyze categorical columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            eda_results['categorical_info'][col] = {
                'unique_count': self.df[col].nunique(),
                'unique_values': list(self.df[col].unique()[:10]),  # Show first 10
                'value_counts': self.df[col].value_counts().head().to_dict()
            }
        
        # Analyze numerical columns for outliers and skewness
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if self.df[col].notna().sum() > 0:
                # Outliers using IQR method
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                
                eda_results['outliers'][col] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(self.df) * 100,
                    'bounds': {'lower': lower_bound, 'upper': upper_bound}
                }
                
                # Skewness
                eda_results['skewness'][col] = self.df[col].skew()
        
        # Check for highly correlated features
        if len(numerical_cols) > 1:
            corr_matrix = self.df[numerical_cols].corr()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_matrix.iloc[i, j]
                        })
            eda_results['correlation_issues'] = high_corr_pairs
        
        return eda_results
    
    def generate_cleaning_suggestions(self, eda_results):
        """Generate automatic cleaning suggestions based on EDA"""
        suggestions = {
            'missing_values': [],
            'outliers': [],
            'scaling': [],
            'encoding': [],
            'feature_engineering': []
        }
        
        # Missing values suggestions
        for col, missing_pct in eda_results['missing_percentage'].items():
            if missing_pct > 0:
                dtype = eda_results['basic_info']['dtypes'][col]
                if missing_pct < 5:
                    if dtype in ['int64', 'float64']:
                        suggestions['missing_values'].append({
                            'column': col,
                            'action': 'impute_mean',
                            'reason': f'Low missing percentage ({missing_pct:.1f}%) - safe to impute with mean'
                        })
                    else:
                        suggestions['missing_values'].append({
                            'column': col,
                            'action': 'impute_mode',
                            'reason': f'Low missing percentage ({missing_pct:.1f}%) - safe to impute with mode'
                        })
                elif missing_pct < 20:
                    suggestions['missing_values'].append({
                        'column': col,
                        'action': 'impute_knn',
                        'reason': f'Moderate missing percentage ({missing_pct:.1f}%) - KNN imputation recommended'
                    })
                else:
                    suggestions['missing_values'].append({
                        'column': col,
                        'action': 'drop_column',
                        'reason': f'High missing percentage ({missing_pct:.1f}%) - consider dropping'
                    })
        
        # Outliers suggestions
        for col, outlier_info in eda_results['outliers'].items():
            if outlier_info['percentage'] > 5:
                suggestions['outliers'].append({
                    'column': col,
                    'action': 'cap_outliers',
                    'reason': f'High outlier percentage ({outlier_info["percentage"]:.1f}%) - cap at bounds'
                })
        
        # Scaling suggestions
        numerical_cols = [col for col, dtype in eda_results['basic_info']['dtypes'].items() 
                         if dtype in ['int64', 'float64']]
        if len(numerical_cols) > 1:
            suggestions['scaling'].append({
                'columns': numerical_cols,
                'action': 'standardize',
                'reason': 'Multiple numerical features detected - standardization recommended'
            })
        
        # Encoding suggestions
        for col, cat_info in eda_results['categorical_info'].items():
            if cat_info['unique_count'] <= 10:
                suggestions['encoding'].append({
                    'column': col,
                    'action': 'one_hot',
                    'reason': f'Low cardinality ({cat_info["unique_count"]} categories) - One-Hot Encoding suitable'
                })
            else:
                suggestions['encoding'].append({
                    'column': col,
                    'action': 'label_encode',
                    'reason': f'High cardinality ({cat_info["unique_count"]} categories) - Label Encoding recommended'
                })
        
        # Feature engineering suggestions
        datetime_cols = self.df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            suggestions['feature_engineering'].append({
                'columns': list(datetime_cols),
                'action': 'datetime_features',
                'reason': 'Datetime columns detected - extract temporal features'
            })
        
        if len(numerical_cols) > 1:
            suggestions['feature_engineering'].append({
                'columns': numerical_cols,
                'action': 'polynomial_features',
                'reason': 'Multiple numerical features - create interaction terms'
            })
        
        return suggestions
    
    def apply_cleaning_action(self, action_type, column, action, params=None):
        """Apply a specific cleaning action"""
        if self.df is None:
            return False, "No data loaded"
        
        try:
            if action_type == 'missing_values':
                if action == 'impute_mean':
                    self.df[column].fillna(self.df[column].mean(), inplace=True)
                    self.cleaning_actions.append(f"Imputed missing values in '{column}' with mean")
                elif action == 'impute_median':
                    self.df[column].fillna(self.df[column].median(), inplace=True)
                    self.cleaning_actions.append(f"Imputed missing values in '{column}' with median")
                elif action == 'impute_mode':
                    self.df[column].fillna(self.df[column].mode()[0], inplace=True)
                    self.cleaning_actions.append(f"Imputed missing values in '{column}' with mode")
                elif action == 'impute_knn':
                    numerical_cols = self.df.select_dtypes(include=[np.number]).columns
                    if len(numerical_cols) > 1:
                        imputer = KNNImputer(n_neighbors=5)
                        self.df[numerical_cols] = imputer.fit_transform(self.df[numerical_cols])
                        self.cleaning_actions.append(f"Applied KNN imputation to numerical columns")
                elif action == 'drop_column':
                    self.df.drop(columns=[column], inplace=True)
                    self.cleaning_actions.append(f"Dropped column '{column}' due to high missing values")
            
            elif action_type == 'outliers':
                if action == 'cap_outliers':
                    Q1 = self.df[column].quantile(0.25)
                    Q3 = self.df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    self.df[column] = self.df[column].clip(lower=lower_bound, upper=upper_bound)
                    self.cleaning_actions.append(f"Capped outliers in '{column}' at IQR bounds")
                elif action == 'remove_outliers':
                    z_scores = np.abs(zscore(self.df[column].dropna()))
                    self.df = self.df[z_scores < 3]
                    self.cleaning_actions.append(f"Removed outliers in '{column}' using Z-score method")
            
            elif action_type == 'scaling':
                if action == 'standardize':
                    scaler = StandardScaler()
                    columns = params if params else [column]
                    self.df[columns] = scaler.fit_transform(self.df[columns])
                    self.cleaning_actions.append(f"Standardized columns: {', '.join(columns)}")
                elif action == 'normalize':
                    scaler = MinMaxScaler()
                    columns = params if params else [column]
                    self.df[columns] = scaler.fit_transform(self.df[columns])
                    self.cleaning_actions.append(f"Normalized columns: {', '.join(columns)}")
            
            elif action_type == 'encoding':
                if action == 'one_hot':
                    dummies = pd.get_dummies(self.df[column], prefix=column)
                    self.df = pd.concat([self.df.drop(columns=[column]), dummies], axis=1)
                    self.cleaning_actions.append(f"Applied One-Hot Encoding to '{column}'")
                elif action == 'label_encode':
                    le = LabelEncoder()
                    self.df[column] = le.fit_transform(self.df[column].astype(str))
                    self.cleaning_actions.append(f"Applied Label Encoding to '{column}'")
            
            return True, "Action applied successfully"
        
        except Exception as e:
            return False, f"Error applying action: {str(e)}"
    
    def engineer_features(self, target_col=None):
        """Perform feature engineering"""
        if self.df is None:
            return False, "No data loaded"
        
        try:
            # Store original column count
            original_cols = len(self.df.columns)
            
            # Datetime feature extraction
            datetime_cols = self.df.select_dtypes(include=['datetime64']).columns
            for col in datetime_cols:
                self.df[f'{col}_year'] = self.df[col].dt.year
                self.df[f'{col}_month'] = self.df[col].dt.month
                self.df[f'{col}_day'] = self.df[col].dt.day
                self.df[f'{col}_dayofweek'] = self.df[col].dt.dayofweek
                self.df[f'{col}_quarter'] = self.df[col].dt.quarter
            
            # Create polynomial features for numerical columns (limited to avoid explosion)
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) >= 2:
                # Select top 3 most important numerical features to avoid feature explosion
                selected_cols = list(numerical_cols)[:3]
                poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
                poly_features = poly.fit_transform(self.df[selected_cols])
                poly_feature_names = poly.get_feature_names_out(selected_cols)
                
                # Add only interaction terms (not squared terms)
                for i, name in enumerate(poly_feature_names):
                    if '*' in name and name not in self.df.columns:
                        self.df[name] = poly_features[:, i]
            
            new_features = len(self.df.columns) - original_cols
            self.cleaning_actions.append(f"Generated {new_features} new engineered features")
            
            return True, f"Generated {new_features} new features"
        
        except Exception as e:
            return False, f"Error in feature engineering: {str(e)}"
    
    def calculate_feature_importance(self, target_column):
        """Calculate feature importance using Random Forest"""
        if self.df is None or target_column not in self.df.columns:
            return None
        
        try:
            # Prepare features (only numerical columns for simplicity)
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numerical_cols if col != target_column]
            
            if len(feature_cols) == 0:
                return None
            
            X = self.df[feature_cols].fillna(0)  # Handle any remaining NaN values
            y = self.df[target_column].fillna(0)
            
            # Determine if classification or regression
            if self.df[target_column].dtype == 'object' or self.df[target_column].nunique() < 10:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                task_type = 'classification'
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                task_type = 'regression'
            
            # Fit model and get feature importance
            model.fit(X, y)
            importance_scores = model.feature_importances_
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': importance_scores
            }).sort_values('importance', ascending=False)
            
            self.feature_importance = importance_df
            self.target_column = target_column
            
            return importance_df
        
        except Exception as e:
            st.error(f"Error calculating feature importance: {str(e)}")
            return None
    
    def generate_visualizations(self, eda_results):
        """Generate comprehensive visualizations"""
        visualizations = {}
        
        # 1. Missing values heatmap
        if any(val > 0 for val in eda_results['missing_percentage'].values()):
            fig_missing = plt.figure(figsize=(12, 8))
            sns.heatmap(self.df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
            plt.title('Missing Values Heatmap')
            plt.tight_layout()
            visualizations['missing_heatmap'] = fig_missing
        
        # 2. Correlation heatmap
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            fig_corr = plt.figure(figsize=(12, 10))
            corr_matrix = self.df[numerical_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.5)
            plt.title('Correlation Heatmap')
            plt.tight_layout()
            visualizations['correlation_heatmap'] = fig_corr
        
        # 3. Distribution plots
        if len(numerical_cols) > 0:
            n_cols = min(3, len(numerical_cols))
            n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
            fig_dist = plt.figure(figsize=(15, 5 * n_rows))
            
            for i, col in enumerate(numerical_cols):
                plt.subplot(n_rows, n_cols, i + 1)
                plt.hist(self.df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
            
            plt.tight_layout()
            visualizations['distributions'] = fig_dist
        
        # 4. Boxplots for outlier detection
        if len(numerical_cols) > 0:
            n_cols = min(3, len(numerical_cols))
            n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
            fig_box = plt.figure(figsize=(15, 5 * n_rows))
            
            for i, col in enumerate(numerical_cols):
                plt.subplot(n_rows, n_cols, i + 1)
                plt.boxplot(self.df[col].dropna())
                plt.title(f'Boxplot of {col}')
                plt.ylabel(col)
            
            plt.tight_layout()
            visualizations['boxplots'] = fig_box
        
        # 5. Feature importance plot
        if self.feature_importance is not None:
            fig_importance = plt.figure(figsize=(12, 8))
            top_features = self.feature_importance.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top Feature Importance (Target: {self.target_column})')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            visualizations['feature_importance'] = fig_importance
        
        return visualizations
    
    def generate_report(self, eda_results, suggestions):
        """Generate comprehensive cleaning report"""
        report = f"""
# Smart Data Cleaning Report

## Dataset Overview
- **Shape**: {eda_results['basic_info']['shape'][0]} rows × {eda_results['basic_info']['shape'][1]} columns
- **Memory Usage**: {eda_results['basic_info']['memory_usage'] / 1024 / 1024:.2f} MB
- **Duplicate Rows**: {eda_results['duplicates']}

## Data Quality Issues Detected

### Missing Values
"""
        
        missing_issues = [f"- **{col}**: {count} missing ({pct:.1f}%)" 
                         for col, count in eda_results['missing_values'].items() 
                         if count > 0]
        
        if missing_issues:
            report += "\n".join(missing_issues)
        else:
            report += "- No missing values detected ✅"
        
        report += "\n\n### Outliers Detected\n"
        outlier_issues = [f"- **{col}**: {info['count']} outliers ({info['percentage']:.1f}%)" 
                         for col, info in eda_results['outliers'].items() 
                         if info['count'] > 0]
        
        if outlier_issues:
            report += "\n".join(outlier_issues)
        else:
            report += "- No significant outliers detected ✅"
        
        report += "\n\n### Data Distribution Issues\n"
        skew_issues = [f"- **{col}**: Skewness = {skew:.2f}" 
                      for col, skew in eda_results['skewness'].items() 
                      if abs(skew) > 2]
        
        if skew_issues:
            report += "\n".join(skew_issues)
        else:
            report += "- No severe skewness detected ✅"
        
        report += "\n\n### High Correlation Pairs\n"
        if eda_results['correlation_issues']:
            corr_issues = [f"- **{pair['feature1']}** ↔ **{pair['feature2']}**: {pair['correlation']:.3f}" 
                          for pair in eda_results['correlation_issues']]
            report += "\n".join(corr_issues)
        else:
            report += "- No highly correlated features detected ✅"
        
        # Add cleaning actions taken
        if self.cleaning_actions:
            report += "\n\n## Cleaning Actions Applied\n"
            for i, action in enumerate(self.cleaning_actions, 1):
                report += f"{i}. {action}\n"
        
        # Add feature importance if available
        if self.feature_importance is not None:
            report += f"\n\n## Feature Importance Analysis\n"
            report += f"**Target Variable**: {self.target_column}\n\n"
            report += "**Top 10 Most Important Features**:\n"
            for i, (_, row) in enumerate(self.feature_importance.head(10).iterrows(), 1):
                report += f"{i}. **{row['feature']}**: {row['importance']:.4f}\n"
        
        return report
    
    def get_download_link(self, df, filename="cleaned_data.csv"):
        """Generate download link for cleaned dataset"""
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Cleaned Dataset</a>'
        return href

# Initialize the cleaner
@st.cache_data
def load_sample_data():
    """Load sample dataset for demonstration"""
    # Create a sample dataset similar to Titanic
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Age': np.random.normal(30, 12, n_samples),
        'Fare': np.random.lognormal(3, 1, n_samples),
        'SibSp': np.random.poisson(0.5, n_samples),
        'Parch': np.random.poisson(0.4, n_samples),
        'Sex': np.random.choice(['male', 'female'], n_samples),
        'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.7, 0.2, 0.1]),
        'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.2, 0.6]),
        'Survived': np.random.choice([0, 1], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some missing values
    missing_indices = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
    df.loc[missing_indices[:50], 'Age'] = np.nan
    df.loc[missing_indices[50:80], 'Embarked'] = np.nan
    
    # Introduce some outliers
    outlier_indices = np.random.choice(df.index, size=20, replace=False)
    df.loc[outlier_indices, 'Fare'] = df.loc[outlier_indices, 'Fare'] * 10
    
    return df

def main():
    st.title("🧹 Smart Data Cleaning Assistant")
    st.markdown("### Automatic EDA, Data Cleaning & Feature Engineering")
    
    # Initialize session state
    if 'cleaner' not in st.session_state:
        st.session_state.cleaner = SmartDataCleaner()
    
    cleaner = st.session_state.cleaner
    
    # Sidebar for file upload and options
    with st.sidebar:
        st.header("Data Upload")
        
        # Option to use sample data
        if st.button("📊 Use Sample Dataset"):
            sample_data = load_sample_data()
            cleaner.df = sample_data
            cleaner.original_df = sample_data.copy()
            st.success("Sample dataset loaded!")
            st.rerun()
        
        st.markdown("**OR**")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your CSV file",
            type=['csv'],
            help="Upload a CSV file to begin analysis"
        )
        
        if uploaded_file is not None:
            success, message = cleaner.load_data(uploaded_file)
            if success:
                st.success(message)
            else:
                st.error(message)
    
    # Main content
    if cleaner.df is not None:
        # Tabs for different sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data Overview", 
            "🔍 EDA Results", 
            "🛠️ Data Cleaning", 
            "⚡ Feature Engineering", 
            "📈 Visualizations"
        ])
        
        # Perform EDA
        eda_results = cleaner.perform_eda()
        suggestions = cleaner.generate_cleaning_suggestions(eda_results)
        
        with tab1:
            st.header("Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", eda_results['basic_info']['shape'][0])
            with col2:
                st.metric("Columns", eda_results['basic_info']['shape'][1])
            with col3:
                st.metric("Missing Values", sum(eda_results['missing_values'].values()))
            with col4:
                st.metric("Duplicates", eda_results['duplicates'])
            
            st.subheader("Data Preview")
            st.dataframe(cleaner.df.head(), use_container_width=True)
            
            st.subheader("Column Information")
            info_df = pd.DataFrame({
                'Column': cleaner.df.columns,
                'Data Type': cleaner.df.dtypes.values,
                'Non-Null Count': cleaner.df.count().values,
                'Missing Count': cleaner.df.isnull().sum().values,
                'Missing %': (cleaner.df.isnull().sum() / len(cleaner.df) * 100).values
            })
            st.dataframe(info_df, use_container_width=True)
        
        with tab2:
            st.header("Exploratory Data Analysis Results")
            
            # Missing values analysis
            if any(val > 0 for val in eda_results['missing_percentage'].values()):
                st.subheader("Missing Values Analysis")
                missing_df = pd.DataFrame({
                    'Column': list(eda_results['missing_values'].keys()),
                    'Missing Count': list(eda_results['missing_values'].values()),
                    'Missing Percentage': [f"{pct:.1f}%" for pct in eda_results['missing_percentage'].values()]
                })
                missing_df = missing_df[missing_df['Missing Count'] > 0]
                st.dataframe(missing_df, use_container_width=True)
            
            # Statistical summary
            if eda_results['summary_stats']:
                st.subheader("Statistical Summary")
                st.dataframe(pd.DataFrame(eda_results['summary_stats']), use_container_width=True)
            
            # Categorical analysis
            if eda_results['categorical_info']:
                st.subheader("Categorical Variables Analysis")
                for col, info in eda_results['categorical_info'].items():
                    with st.expander(f"📊 {col}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Unique Values", info['unique_count'])
                        with col2:
                            st.metric("Most Frequent", list(info['value_counts'].keys())[0] if info['value_counts'] else "N/A")
                        
                        if info['value_counts']:
                            st.bar_chart(pd.Series(info['value_counts']))
            
            # Outliers analysis
            if any(info['count'] > 0 for info in eda_results['outliers'].values()):
                st.subheader("Outliers Detection")
                outliers_data = []
                for col, info in eda_results['outliers'].items():
                    if info['count'] > 0:
                        outliers_data.append({
                            'Column': col,
                            'Outlier Count': info['count'],
                            'Outlier Percentage': f"{info['percentage']:.1f}%"
                        })
                
                if outliers_data:
                    st.dataframe(pd.DataFrame(outliers_data), use_container_width=True)
            
            # Skewness analysis
            if eda_results['skewness']:
                st.subheader("Data Skewness Analysis")
                skew_data = []
                for col, skew_val in eda_results['skewness'].items():
                    skew_interpretation = "Normal" if abs(skew_val) < 0.5 else ("Moderate" if abs(skew_val) < 1 else "High")
                    skew_data.append({
                        'Column': col,
                        'Skewness': f"{skew_val:.3f}",
                        'Interpretation': skew_interpretation
                    })
                st.dataframe(pd.DataFrame(skew_data), use_container_width=True)
        
        with tab3:
            st.header("Data Cleaning & Preprocessing")
            
            # Display suggestions
            st.subheader("🔧 Automated Cleaning Suggestions")
            
            # Missing values suggestions
            if suggestions['missing_values']:
                st.markdown("**Missing Values Treatment:**")
                for suggestion in suggestions['missing_values']:
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.write(f"**{suggestion['column']}**")
                    with col2:
                        st.write(suggestion['reason'])
                    with col3:
                        action_key = f"missing_{suggestion['column']}_{suggestion['action']}"
                        if st.button("Apply", key=action_key):
                            success, message = cleaner.apply_cleaning_action(
                                'missing_values', suggestion['column'], suggestion['action']
                            )
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)
            
            # Outliers suggestions
            if suggestions['outliers']:
                st.markdown("**Outlier Treatment:**")
                for suggestion in suggestions['outliers']:
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.write(f"**{suggestion['column']}**")
                    with col2:
                        st.write(suggestion['reason'])
                    with col3:
                        action_key = f"outlier_{suggestion['column']}_{suggestion['action']}"
                        if st.button("Apply", key=action_key):
                            success, message = cleaner.apply_cleaning_action(
                                'outliers', suggestion['column'], suggestion['action']
                            )
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)
            
            # Scaling suggestions
            if suggestions['scaling']:
                st.markdown("**Feature Scaling:**")
                for suggestion in suggestions['scaling']:
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.write("**Numerical Features**")
                    with col2:
                        st.write(suggestion['reason'])
                    with col3:
                        action_key = f"scaling_{suggestion['action']}"
                        if st.button("Apply", key=action_key):
                            success, message = cleaner.apply_cleaning_action(
                                'scaling', None, suggestion['action'], suggestion['columns']
                            )
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)
            
            # Encoding suggestions
            if suggestions['encoding']:
                st.markdown("**Categorical Encoding:**")
                for suggestion in suggestions['encoding']:
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.write(f"**{suggestion['column']}**")
                    with col2:
                        st.write(suggestion['reason'])
                    with col3:
                        action_key = f"encoding_{suggestion['column']}_{suggestion['action']}"
                        if st.button("Apply", key=action_key):
                            success, message = cleaner.apply_cleaning_action(
                                'encoding', suggestion['column'], suggestion['action']
                            )
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)
            
            # Manual cleaning options
            st.subheader("🎛️ Manual Cleaning Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Missing Value Treatment**")
                missing_col = st.selectbox("Select Column", 
                    [col for col in cleaner.df.columns if cleaner.df[col].isnull().sum() > 0],
                    key="missing_manual"
                )
                if missing_col:
                    missing_method = st.selectbox("Imputation Method", 
                        ["Mean", "Median", "Mode", "KNN"], key="missing_method")
                    if st.button("Apply Manual Imputation"):
                        method_map = {"Mean": "impute_mean", "Median": "impute_median", 
                                    "Mode": "impute_mode", "KNN": "impute_knn"}
                        success, message = cleaner.apply_cleaning_action(
                            'missing_values', missing_col, method_map[missing_method]
                        )
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
            
            with col2:
                st.markdown("**Outlier Treatment**")
                numerical_cols = cleaner.df.select_dtypes(include=[np.number]).columns
                outlier_col = st.selectbox("Select Column", numerical_cols, key="outlier_manual")
                if outlier_col:
                    outlier_method = st.selectbox("Treatment Method", 
                        ["Cap Outliers", "Remove Outliers"], key="outlier_method")
                    if st.button("Apply Outlier Treatment"):
                        method_map = {"Cap Outliers": "cap_outliers", "Remove Outliers": "remove_outliers"}
                        success, message = cleaner.apply_cleaning_action(
                            'outliers', outlier_col, method_map[outlier_method]
                        )
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
            
            # Show cleaning actions taken
            if cleaner.cleaning_actions:
                st.subheader("📋 Cleaning Actions Applied")
                for i, action in enumerate(cleaner.cleaning_actions, 1):
                    st.write(f"{i}. {action}")
        
        with tab4:
            st.header("Feature Engineering & Importance Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔧 Feature Engineering")
                if st.button("🚀 Auto-Generate Features"):
                    success, message = cleaner.engineer_features()
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                
                # Show current feature count
                st.metric("Current Feature Count", len(cleaner.df.columns))
                
                # Feature engineering suggestions
                if suggestions['feature_engineering']:
                    st.markdown("**Suggested Feature Engineering:**")
                    for suggestion in suggestions['feature_engineering']:
                        st.write(f"• {suggestion['reason']}")
            
            with col2:
                st.subheader("🎯 Feature Importance Analysis")
                target_col = st.selectbox(
                    "Select Target Column", 
                    cleaner.df.columns,
                    key="target_selection"
                )
                
                if st.button("Calculate Feature Importance"):
                    importance_df = cleaner.calculate_feature_importance(target_col)
                    if importance_df is not None:
                        st.success("Feature importance calculated!")
                        st.rerun()
                    else:
                        st.error("Could not calculate feature importance")
                
                # Show feature importance if available
                if cleaner.feature_importance is not None:
                    st.markdown(f"**Target Variable**: {cleaner.target_column}")
                    st.dataframe(cleaner.feature_importance.head(10), use_container_width=True)
        
        with tab5:
            st.header("Data Visualizations")
            
            # Generate visualizations
            visualizations = cleaner.generate_visualizations(eda_results)
            
            # Display visualizations
            if 'missing_heatmap' in visualizations:
                st.subheader("Missing Values Heatmap")
                st.pyplot(visualizations['missing_heatmap'])
            
            if 'correlation_heatmap' in visualizations:
                st.subheader("Feature Correlation Heatmap")
                st.pyplot(visualizations['correlation_heatmap'])
            
            if 'distributions' in visualizations:
                st.subheader("Feature Distributions")
                st.pyplot(visualizations['distributions'])
            
            if 'boxplots' in visualizations:
                st.subheader("Outlier Detection (Boxplots)")
                st.pyplot(visualizations['boxplots'])
            
            if 'feature_importance' in visualizations:
                st.subheader("Feature Importance Ranking")
                st.pyplot(visualizations['feature_importance'])
        
        # Generate and display report
        st.header("📊 Comprehensive Analysis Report")
        
        with st.expander("📋 View Full Report", expanded=False):
            report = cleaner.generate_report(eda_results, suggestions)
            st.markdown(report)
        
        # Download section
        st.header("💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download Cleaned Dataset"):
                csv = cleaner.df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"cleaned_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📄 Download Report"):
                report = cleaner.generate_report(eda_results, suggestions)
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name=f"data_cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
        
        # Before/After comparison
        if cleaner.cleaning_actions:
            st.header("🔄 Before vs After Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Dataset")
                st.write(f"Shape: {cleaner.original_df.shape}")
                st.write(f"Missing Values: {cleaner.original_df.isnull().sum().sum()}")
                st.dataframe(cleaner.original_df.head(), use_container_width=True)
            
            with col2:
                st.subheader("Cleaned Dataset")
                st.write(f"Shape: {cleaner.df.shape}")
                st.write(f"Missing Values: {cleaner.df.isnull().sum().sum()}")
                st.dataframe(cleaner.df.head(), use_container_width=True)
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to the Smart Data Cleaning Assistant! 🧹
        
        This tool will help you:
        - 📊 **Perform Automatic EDA**: Detect missing values, outliers, and data quality issues
        - 🛠️ **Get Cleaning Suggestions**: Receive intelligent recommendations for data preprocessing
        - ⚡ **Engineer Features**: Create new features and analyze feature importance
        - 📈 **Generate Visualizations**: Create comprehensive plots and charts
        - 📋 **Export Results**: Download cleaned datasets and detailed reports
        
        ### Getting Started:
        1. **Upload a CSV file** using the sidebar, or
        2. **Try the sample dataset** to see the tool in action
        
        ### Features:
        - **Automated Missing Value Detection & Treatment**
        - **Outlier Detection using Z-score and IQR methods**
        - **Feature Scaling (Standardization & Normalization)**
        - **Categorical Encoding (One-Hot & Label Encoding)**
        - **Feature Engineering (Polynomial & Interaction Features)**
        - **Feature Importance Analysis using Random Forest**
        - **Comprehensive Visualizations**
        - **Detailed Analysis Reports**
        
        Ready to clean your data? Upload a file or try the sample dataset! 🚀
        """)

if __name__ == "__main__":
    main()    main()
