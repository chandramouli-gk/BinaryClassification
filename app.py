import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef)
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Binary Classification Model Comparison", layout="wide")

# Title and description
st.title("🤖 Binary Classification Model Comparison")
st.markdown("Compare 6 different machine learning models for binary classification tasks")

# Sidebar for file upload and parameters
st.sidebar.header("Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type=['csv'])

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    
    st.sidebar.success(f"✅ File uploaded: {uploaded_file.name}")
    st.sidebar.write(f"Shape: {df.shape}")
    
    # Display dataset preview
    with st.expander("📊 View Dataset Preview"):
        st.dataframe(df.head(10))
        st.write(f"**Total rows:** {df.shape[0]}, **Total columns:** {df.shape[1]}")
    
    # Data preparation
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Test size slider
    test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20) / 100
    random_state = st.sidebar.number_input("Random State", 0, 100, 42)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Display split information
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Set Size", X_train.shape[0])
    with col2:
        st.metric("Test Set Size", X_test.shape[0])
    
    st.markdown("---")
    
    # Model selection
    st.sidebar.header("Model Selection")
    
    model_options = {
        'Logistic Regression': st.sidebar.checkbox('Logistic Regression', value=True),
        'Decision Tree': st.sidebar.checkbox('Decision Tree', value=True),
        'K-Nearest Neighbors': st.sidebar.checkbox('K-Nearest Neighbors', value=True),
        'Naive Bayes': st.sidebar.checkbox('Naive Bayes', value=True),
        'Random Forest': st.sidebar.checkbox('Random Forest', value=True),
        'XGBoost': st.sidebar.checkbox('XGBoost', value=True)
    }
    
    selected_models = [name for name, selected in model_options.items() if selected]
    
    # Train button
    if st.sidebar.button("🚀 Train Models", type="primary"):
        if not selected_models:
            st.error("⚠️ Please select at least one model to train!")
        else:
            # Create instances of selected models
            models = {}
            if 'Logistic Regression' in selected_models:
                models['Logistic Regression'] = LogisticRegression(random_state=random_state)
            if 'Decision Tree' in selected_models:
                models['Decision Tree'] = DecisionTreeClassifier(random_state=random_state)
            if 'K-Nearest Neighbors' in selected_models:
                models['K-Nearest Neighbors'] = KNeighborsClassifier()
            if 'Naive Bayes' in selected_models:
                models['Naive Bayes'] = GaussianNB()
            if 'Random Forest' in selected_models:
                models['Random Forest'] = RandomForestClassifier(random_state=random_state)
            if 'XGBoost' in selected_models:
                models['XGBoost'] = XGBClassifier(random_state=random_state, eval_metric='logloss')
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Train all models
            trained_models = {}
            for idx, (name, model) in enumerate(models.items()):
                status_text.text(f"Training {name}...")
                model.fit(X_train, y_train)
                trained_models[name] = model
                progress_bar.progress((idx + 1) / len(models))
            
            status_text.text("✅ All models trained successfully!")
            progress_bar.empty()
            
            st.markdown("---")
            st.header("📈 Model Performance Results")
            
            # Evaluate all models
            results = {}
            
            for name, model in trained_models.items():
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='binary')
                recall = recall_score(y_test, y_pred, average='binary')
                f1 = f1_score(y_test, y_pred, average='binary')
                auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
                mcc = matthews_corrcoef(y_test, y_pred)
                
                # Store results
                results[name] = {
                    'Accuracy': accuracy,
                    'AUC Score': auc if auc else 0,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1,
                    'MCC Score': mcc
                }
            
            # Create a summary dataframe
            results_df = pd.DataFrame(results).T
            
            # Display results as styled table
            st.subheader("Model Performance Summary")
            st.dataframe(results_df.style.format("{:.4f}").highlight_max(axis=0, color='lightgreen'))
            
            # Display individual model metrics
            st.markdown("---")
            st.subheader("Detailed Model Metrics")
            
            cols = st.columns(2)
            for idx, (name, metrics) in enumerate(results.items()):
                with cols[idx % 2]:
                    with st.expander(f"📊 {name}", expanded=True):
                        st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                        st.metric("AUC Score", f"{metrics['AUC Score']:.4f}" if metrics['AUC Score'] > 0 else "N/A")
                        st.metric("Precision", f"{metrics['Precision']:.4f}")
                        st.metric("Recall", f"{metrics['Recall']:.4f}")
                        st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                        st.metric("MCC Score", f"{metrics['MCC Score']:.4f}")
            
            # Bar chart comparison
            st.markdown("---")
            st.subheader("📊 Visual Comparison")
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Model Performance Comparison', fontsize=16)
            
            metrics_to_plot = ['Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC Score']
            
            for idx, metric in enumerate(metrics_to_plot):
                ax = axes[idx // 3, idx % 3]
                values = [results[model][metric] for model in trained_models.keys()]
                bars = ax.bar(range(len(trained_models)), values, color='skyblue')
                ax.set_xticks(range(len(trained_models)))
                ax.set_xticklabels(trained_models.keys(), rotation=45, ha='right')
                ax.set_ylabel(metric)
                ax.set_title(metric)
                ax.set_ylim([0, 1])
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Best model
            st.markdown("---")
            best_model = results_df['Accuracy'].idxmax()
            best_accuracy = results_df['Accuracy'].max()
            st.success(f"🏆 **Best Model:** {best_model} with Accuracy: {best_accuracy:.4f}")

else:
    st.info("👈 Please upload a CSV file to get started")
    st.markdown("""
    ### Instructions:
    1. Upload your CSV dataset using the sidebar
    2. The last column should be the target variable (binary classification)
    3. Configure the test size and random state
    4. Select the models you want to compare
    5. Click 'Train Models' to start the analysis
    
    ### Supported Models:
    - Logistic Regression
    - Decision Tree
    - K-Nearest Neighbors
    - Naive Bayes
    - Random Forest
    - XGBoost
    
    ### Evaluation Metrics:
    - Accuracy
    - AUC Score
    - Precision
    - Recall
    - F1 Score
    - Matthews Correlation Coefficient (MCC)
    """)
