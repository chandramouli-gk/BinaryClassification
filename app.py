import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from datetime import datetime

# Page configuration with custom colors
st.set_page_config(
    page_title="Binary Classification Model Compare", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for color scheme (white, blue, black, red)
st.markdown("""
<style>
    .main {
        background-color: white;
        position: relative;
    }
    .watermark {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%) rotate(-45deg);
        font-size: 100px;
        color: rgba(0, 0, 128, 0.025);
        z-index: 999;
        pointer-events: none;
        font-weight: bold;
        white-space: nowrap;
    }
    h1, h2, h3 {
        color: #000080 !important;
    }
    .stButton>button {
        background-color: #000080;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #0000CD;
    }
    .author-info {
        background-color: #000080;
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        text-align: right;
        margin-bottom: 20px;
        font-size: 14px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .stMetric {
        background-color: #f0f8ff;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #000080;
    }
    .success-box {
        background-color: #000080;
        color: white;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
    }
    .error-box {
        background-color: #DC143C;
        color: white;
        padding: 15px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Add visible watermark
st.markdown("""
<div class="watermark">2025AA05418</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'main'
if 'results' not in st.session_state:
    st.session_state.results = None
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = []
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}

models_dir = 'trained_models'

@st.cache_resource
def load_models():
    """Load all trained models from pickle files"""
    models = {}
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'K-Nearest Neighbors': 'k-nearest_neighbors.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl'
    }
    
    if not os.path.exists(models_dir):
        return None
    
    for name, filename in model_files.items():
        filepath = os.path.join(models_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                models[name] = pickle.load(f)
    
    return models if models else None

@st.cache_resource
def load_preprocessor():
    """Load the preprocessor"""
    preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
    if os.path.exists(preprocessor_path):
        with open(preprocessor_path, 'rb') as f:
            return pickle.load(f)
    return None

def get_expected_columns():
    """Get expected columns from preprocessor"""
    return ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
            'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

@st.cache_data
def load_readme_from_github():
    """Load README.md from GitHub repository"""
    try:
        readme_url = "https://raw.githubusercontent.com/chandramouli-gk/2025AA05418/main/README.md"
        response = pd.read_csv(readme_url, sep='\n', header=None, dtype=str)
        readme_content = '\n'.join(response[0].tolist())
        return readme_content
    except:
        return None

def validate_test_data(df, preprocessor):
    """Validate that test data meets preprocessor requirements"""
    expected_cols = get_expected_columns()
    
    # Check if dataframe has correct number of columns
    if df.shape[1] != len(expected_cols):
        return False, f"Expected {len(expected_cols)} columns, but got {df.shape[1]}"
    
    # Check column names (assuming last column is target)
    df_cols = df.columns.tolist()
    
    # Define expected feature types
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    # Check if all required columns exist
    required_features = numerical_cols + categorical_cols
    missing_cols = [col for col in required_features if col not in df_cols[:-1]]
    
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}"
    
    # Check data types
    for col in numerical_cols:
        if col in df.columns and not np.issubdtype(df[col].dtype, np.number):
            return False, f"Column '{col}' should be numeric but is {df[col].dtype}"
    
    return True, "Data validation successful"

def create_confusion_matrix_plot(y_true, y_pred, model_name, color_scheme):
    """Create confusion matrix plot with custom colors"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=color_scheme, 
                cbar_kws={'label': 'Count'}, ax=ax,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    ax.set_xlabel('Predicted Label', fontsize=12, color='#000080')
    ax.set_ylabel('True Label', fontsize=12, color='#000080')
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, color='#000080', fontweight='bold')
    
    return fig

def export_to_pdf(results_df, predictions, y_test, selected_models):
    """Export comprehensive results to PDF with watermark, confusion matrices, and README content from GitHub"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=80, bottomMargin=50)
    
    elements = []
    styles = getSampleStyleSheet()
    
    # Load README from GitHub
    readme_content = load_readme_from_github()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#000080'),
        spaceAfter=20,
        alignment=TA_CENTER
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#000080'),
        spaceAfter=10,
        spaceBefore=10
    )
    
    # Title
    title = Paragraph("Binary Classification Model Comparison Report", title_style)
    elements.append(title)
    elements.append(Spacer(1, 10))
    
    # Author info
    author_text = f"<b>Author:</b> ChandraMouli GK | <b>BITS ID:</b> 2025AA05418 | <b>Date:</b> {datetime.now().strftime('%Y-%m-%d')}"
    elements.append(Paragraph(author_text, styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Extract content from README if available
    if readme_content:
        # Extract Problem Statement
        if '## a. Problem Statement' in readme_content:
            problem_start = readme_content.find('## a. Problem Statement')
            problem_end = readme_content.find('## b. Dataset Description', problem_start)
            if problem_end != -1:
                problem_section = readme_content[problem_start:problem_end].strip()
                # Clean up markdown formatting
                problem_text = problem_section.replace('## a. Problem Statement', '').strip()
                problem_text = problem_text.split('\n\n')[0]  # Get first paragraph
            else:
                problem_text = "Binary classification of heart disease using machine learning models."
        else:
            problem_text = "Binary classification of heart disease using machine learning models."
        
        # Extract Dataset Description
        if '## b. Dataset Description' in readme_content:
            dataset_start = readme_content.find('## b. Dataset Description')
            dataset_end = readme_content.find('## c. Models Used', dataset_start)
            if dataset_end != -1:
                dataset_section = readme_content[dataset_start:dataset_end].strip()
                # Extract key info
                dataset_lines = dataset_section.split('\n')
                dataset_info = []
                for line in dataset_lines:
                    if 'Source:' in line or 'Total Records:' in line or 'Features:' in line or 'Target Variable:' in line or 'Data Split:' in line:
                        dataset_info.append(line.strip('- ').strip())
                dataset_text = '<br/>'.join(dataset_info[:5]) if dataset_info else "Heart Disease Dataset with 13 features"
            else:
                dataset_text = "Heart Disease Dataset with 13 features"
        else:
            dataset_text = "Heart Disease Dataset with 13 features"
    else:
        # Fallback content
        problem_text = """This assignment develops and compares six machine learning models for binary classification 
        of heart disease. The task predicts whether a patient has heart disease (1) or not (0) based on clinical and 
        demographic features. Models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and MCC metrics."""
        dataset_text = """<b>Source:</b> Heart Disease Dataset from Kaggle<br/>
        <b>Total Records:</b> 1,025 patients<br/>
        <b>Features:</b> 13 clinical and demographic features<br/>
        <b>Target Variable:</b> Binary (0 = No disease, 1 = Disease)<br/>
        <b>Data Split:</b> 80% Training (820 records) / 20% Testing (205 records)"""
    
    # Problem Statement
    elements.append(Paragraph("<b>Problem Statement</b>", heading2_style))
    elements.append(Paragraph(problem_text, styles['Normal']))
    elements.append(Spacer(1, 15))
    
    # Dataset Description
    elements.append(Paragraph("<b>Dataset Description</b>", heading2_style))
    elements.append(Paragraph(dataset_text, styles['Normal']))
    elements.append(Spacer(1, 15))
    
    # Model Performance Summary
    elements.append(Paragraph("<b>Model Performance Summary</b>", heading2_style))
    elements.append(Spacer(1, 10))
    
    # Prepare table data
    table_data = [['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'MCC']]
    for idx, row in results_df.iterrows():
        table_data.append([idx] + [f"{val:.4f}" for val in row])
    
    # Create table
    t = Table(table_data, colWidths=[100, 60, 60, 60, 60, 60, 60])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#000080')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    
    elements.append(t)
    elements.append(Spacer(1, 15))
    
    # Best model
    best_model = results_df['Accuracy'].idxmax()
    best_accuracy = results_df['Accuracy'].max()
    best_text = f"<b>Best Performing Model:</b> {best_model} (Accuracy: {best_accuracy:.4f})"
    elements.append(Paragraph(best_text, styles['Normal']))
    elements.append(PageBreak())
    
    # Confusion Matrices
    elements.append(Paragraph("<b>Confusion Matrices</b>", heading2_style))
    elements.append(Spacer(1, 10))
    
    color_schemes = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges', 'YlOrRd']
    
    for idx, model_name in enumerate(selected_models):
        # Create confusion matrix
        cm = confusion_matrix(y_test, predictions[model_name])
        
        # Save confusion matrix as image
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap=color_schemes[idx % len(color_schemes)],
                    cbar=False, ax=ax,
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'])
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('True', fontsize=9)
        ax.set_title(f'{model_name}', fontsize=10, fontweight='bold')
        plt.tight_layout()
        
        # Save to BytesIO
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        # Add to PDF
        img = Image(img_buffer, width=3*inch, height=2.5*inch)
        elements.append(Paragraph(f"<b>{model_name}</b>", styles['Normal']))
        elements.append(img)
        elements.append(Spacer(1, 10))
        
        # Add 2 per page
        if (idx + 1) % 2 == 0 and idx < len(selected_models) - 1:
            elements.append(PageBreak())
    
    elements.append(PageBreak())
    
    # Body style for wrapped text
    body_style = ParagraphStyle(
        name='BodyStyle',
        fontName='Helvetica',
        fontSize=9,
        leading=12,   # line spacing
    )

    # Header style
    header_style = ParagraphStyle(
        name='HeaderStyle',
        fontName='Helvetica-Bold',
        fontSize=9,
        textColor=colors.whitesmoke
    )

    # Model Observations
    elements.append(Paragraph("<b>Model Performance Observations</b>", heading2_style))
    elements.append(Spacer(1, 10))
    
    observations = {
        'Logistic Regression': 'Good baseline performance with strong interpretability. Balanced precision-recall trade-off makes it reliable for general predictions.',
        'Decision Tree': 'Excellent performance with near-perfect metrics. Perfect precision indicates no false positives, but potential risk of overfitting.',
        'K-Nearest Neighbors': 'Moderate performance with good AUC. Sensitive to feature scaling and may struggle with high-dimensional data.',
        'Naive Bayes': 'Reasonable performance with good recall. Independence assumption may limit performance with correlated medical features.',
        'Random Forest': 'Best performing model with perfect scores across all metrics. Ensemble approach effectively captures complex patterns and feature interactions.',
        'XGBoost': 'Outstanding performance comparable to Random Forest. Gradient boosting with regularization provides excellent balance of performance and generalization.'
    }
    
    obs_data = [
    [Paragraph('<b>Model</b>', header_style),
     Paragraph('<b>Observation</b>', header_style)]
    ]
    for model in selected_models:
        if model in observations:
            obs_data.append([
            Paragraph(model, body_style),
            Paragraph(observations[model], body_style)
        ])
    
    obs_table = Table(obs_data, colWidths=[120, 200])
    obs_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#000080')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(obs_table)
    elements.append(Spacer(1, 15))
    
    # Conclusion
    elements.append(Paragraph("<b>Conclusion</b>", heading2_style))
    conclusion_text = f"""This analysis successfully compared six machine learning models for heart disease prediction. 
    Ensemble methods (Random Forest and XGBoost) significantly outperformed traditional classifiers. 
    <b>{best_model}</b> achieved the best accuracy of <b>{best_accuracy:.4f}</b>, demonstrating exceptional 
    predictive capability for clinical decision support."""
    elements.append(Paragraph(conclusion_text, styles['Normal']))
    
    # Watermark footer on each page
    def add_watermark(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 50)
        canvas.setFillColorRGB(0.95, 0.95, 0.95)
        canvas.rotate(45)
        canvas.drawString(100, 50, "2025AA05418")
        canvas.restoreState()
    
    doc.build(elements, onFirstPage=add_watermark, onLaterPages=add_watermark)
    buffer.seek(0)
    return buffer

# Main page logic
if st.session_state.page == 'main':
    # Author details at the top
    st.markdown("""
    <div class="author-info">
        <strong style="font-size: 16px;">ChandraMouli GK</strong><br>
        <strong style="font-size: 20px;">BITS ID:2025AA05418</strong><br>
        <strong>Machine Learning Assignment 2</strong>
    </div>
    """, unsafe_allow_html=True)
    
    st.title("Binary Classification Model Compare")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Load models
    trained_models = load_models()
    preprocessor = load_preprocessor()
    
    if trained_models is None or preprocessor is None:
        st.markdown('<div class="error-box">WARNING: Models or preprocessor not found! Please train models first using the Jupyter notebook.</div>', unsafe_allow_html=True)
        st.stop()
    
    # Create main layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Upload Test Data")
        
        # Two options: Upload or Load Test Data from GitHub
        col_upload, col_test_github = st.columns([2, 1])
        
        with col_upload:
            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key='file_uploader')
        
        with col_test_github:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Load Test Data", use_container_width=True, help="Load test_data.csv from GitHub"):
                try:
                    github_url = "https://raw.githubusercontent.com/chandramouli-gk/2025AA05418/main/test_data.csv"
                    st.session_state.uploaded_df = pd.read_csv(github_url)
                    st.success(f"Test data loaded! Shape: {st.session_state.uploaded_df.shape}")
                except Exception as e:
                    st.error(f"Failed to load test data: {str(e)}")
        
        if uploaded_file is not None:
            st.session_state.uploaded_df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully! Shape: {st.session_state.uploaded_df.shape}")
        
        # Preview data if any data is loaded
        if st.session_state.uploaded_df is not None:
            with st.expander("Preview Data"):
                st.dataframe(st.session_state.uploaded_df.head(10))
        
        st.markdown("---")
        
        # Model selection
        st.markdown("### Select Models")
        
        col_a, col_b = st.columns(2)
        with col_a:
            select_all = st.checkbox("Select All Models", value=False)
        
        if select_all:
            st.session_state.selected_models = list(trained_models.keys())
        else:
            st.session_state.selected_models = []
            
        model_checkboxes = {}
        cols = st.columns(2)
        for idx, model_name in enumerate(trained_models.keys()):
            with cols[idx % 2]:
                if select_all:
                    checked = st.checkbox(model_name, value=True, key=f"model_{model_name}", disabled=True)
                else:
                    checked = st.checkbox(model_name, value=False, key=f"model_{model_name}")
                    if checked and model_name not in st.session_state.selected_models:
                        st.session_state.selected_models.append(model_name)
                    elif not checked and model_name in st.session_state.selected_models:
                        st.session_state.selected_models.remove(model_name)
        
        st.markdown("---")
        
        # Run button
        if st.button("Run Analysis", use_container_width=True, type="primary"):
            if st.session_state.uploaded_df is None:
                st.error("Please upload a test file first!")
            elif len(st.session_state.selected_models) == 0:
                st.error("Please select at least one model!")
            else:
                # Validate data
                is_valid, message = validate_test_data(st.session_state.uploaded_df, preprocessor)
                
                if is_valid:
                    st.session_state.page = 'results'
                    st.rerun()
                else:
                    st.session_state.page = 'error'
                    st.session_state.error_message = message
                    st.rerun()

elif st.session_state.page == 'error':
    # Author details at the top
    st.markdown("""
    <div class="author-info">
        <strong style="font-size: 16px;">ChandraMouli GK</strong><br>
        <strong style="font-size: 20px;">BITS ID:2025AA05418</strong><br>
        <strong>Machine Learning Assignment 2</strong>
    </div>
    """, unsafe_allow_html=True)
    
    st.title("Data Validation Failed")
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(f'<div class="error-box">{st.session_state.error_message}</div>', unsafe_allow_html=True)
    
    st.markdown("### Required Data Format")
    st.markdown("""
    Your test data must have the following columns in order:
    
    **Numerical Features:**
    - `age`: Age in years
    - `trestbps`: Resting blood pressure (mm Hg)
    - `chol`: Serum cholesterol (mg/dL)
    - `thalach`: Maximum heart rate achieved
    - `oldpeak`: ST depression induced by exercise
    
    **Categorical Features:**
    - `sex`: Biological sex (0 or 1)
    - `cp`: Chest pain type (0-3)
    - `fbs`: Fasting blood sugar >120 mg/dL (0 or 1)
    - `restecg`: Resting ECG results (0-2)
    - `exang`: Exercise induced angina (0 or 1)
    - `slope`: Slope of peak exercise ST segment (0-2)
    - `ca`: Number of major vessels (0-3)
    - `thal`: Thalassemia (3, 6, or 7)
    
    **Target Variable (last column):**
    - `target`: Heart disease presence (0 or 1)
    """)
    
    expected_cols = get_expected_columns()
    st.markdown("### Expected Columns")
    st.write(expected_cols)
    
    if st.session_state.uploaded_df is not None:
        st.markdown("### Your Data Columns")
        st.write(st.session_state.uploaded_df.columns.tolist())
    
    st.markdown("---")
    if st.button("Back to Main Page", use_container_width=True):
        st.session_state.page = 'main'
        st.rerun()

elif st.session_state.page == 'results':
    # Author details at the top
    st.markdown("""
    <div class="author-info">
        <strong style="font-size: 16px;">ChandraMouli GK</strong><br>
        <strong style="font-size: 20px;">BITS ID:2025AA05418</strong><br>
        <strong>Machine Learning Assignment 2</strong>
    </div>
    """, unsafe_allow_html=True)
    
    st.title("Model Performance Results")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Load models and preprocessor
    trained_models = load_models()
    preprocessor = load_preprocessor()
    
    # Prepare data
    test_df = st.session_state.uploaded_df
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    st.session_state.y_test = y_test
    
    # Preprocess data
    X_test_processed = preprocessor.transform(X_test)
    
    # Evaluate models
    with st.spinner("Evaluating models..."):
        results = {}
        predictions = {}
        
        for name in st.session_state.selected_models:
            model = trained_models[name]
            
            # Make predictions
            y_pred = model.predict(X_test_processed)
            y_pred_proba = model.predict_proba(X_test_processed)[:, 1] if hasattr(model, 'predict_proba') else None
            
            predictions[name] = y_pred
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
            auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
            mcc = matthews_corrcoef(y_test, y_pred)
            
            results[name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'AUC Score': auc,
                'MCC Score': mcc
            }
        
        st.session_state.results = results
        st.session_state.predictions = predictions
    
    # Find best model
    results_df = pd.DataFrame(st.session_state.results).T
    best_model = results_df['Accuracy'].idxmax()
    best_accuracy = results_df['Accuracy'].max()
    
    # Display best model
    st.markdown(f'<div class="success-box">Best Model: {best_model} (Accuracy: {best_accuracy:.4f})</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display performance table
    st.markdown("### Performance Metrics")
    
    # Style the dataframe
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: #90EE90' if v else '' for v in is_max]
    
    styled_df = results_df.style.format("{:.4f}").apply(highlight_max, axis=0)
    st.dataframe(styled_df, use_container_width=True)
    
    st.markdown("---")
    
    # Confusion matrices
    st.markdown("### Confusion Matrices")
    
    # Define color schemes for different models
    color_schemes = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges', 'YlOrRd']
    
    cols = st.columns(min(3, len(st.session_state.selected_models)))
    for idx, model_name in enumerate(st.session_state.selected_models):
        with cols[idx % 3]:
            fig = create_confusion_matrix_plot(
                st.session_state.y_test, 
                st.session_state.predictions[model_name],
                model_name,
                color_schemes[idx % len(color_schemes)]
            )
            st.pyplot(fig)
            plt.close()
    
    st.markdown("---")
    
    # Comparison chart
    st.markdown("### Model Comparison")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    results_df.plot(kind='bar', ax=ax, color=['#000080', '#4169E1', '#6495ED', '#87CEEB', '#B0C4DE', '#DC143C'])
    ax.set_xlabel('Models', fontsize=16, color='#000080')
    ax.set_ylabel('Score', fontsize=16, color='#000080')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=6, frameon=True, fancybox=True, shadow=True)
    ax.set_ylim([0, 1])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    # Export options
    st.markdown("### Export Results")
    
    pdf_buffer = export_to_pdf(results_df, st.session_state.predictions, st.session_state.y_test, st.session_state.selected_models)
    st.download_button(
        label="Download Complete PDF Report",
        data=pdf_buffer,
        file_name=f"ML_Report_2025AA05418_{datetime.now().strftime('%Y%m%d')}.pdf",
        mime="application/pdf",
        use_container_width=True,
        help="Download comprehensive report with performance metrics, confusion matrices, and analysis"
    )
    
    st.markdown("---")
    
    # Back button
    if st.button("Back to Main Page", use_container_width=True):
        st.session_state.page = 'main'
        st.session_state.uploaded_df = None
        st.session_state.results = None
        st.session_state.predictions = {}
        st.session_state.selected_models = []
        st.rerun()
