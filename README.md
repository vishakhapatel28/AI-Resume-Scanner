# Smart Resume Scanner

A Python application using NLP and ML to scan and rank resumes against job descriptions.

## Features

- Upload multiple PDF resumes
- Input custom job descriptions
- Automatic text extraction and preprocessing
- TF-IDF and BERT-based semantic matching using Sentence    Transformers
- Ranked results display with table and bar chart

## Installation

1. Clone or download the project.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## Usage

- Upload PDF resumes using the file uploader.
- Paste the job description in the text area.
- Click "Analyze Resumes" to get ranked results.

## Libraries Used

- Streamlit for the web interface
- Scikit-learn for TF-IDF and cosine similarity
- NLTK for text preprocessing
- Pandas for data handling
- PyPDF2 for PDF text extraction
- Sentence Transformers (BERT)

## Project Structure

- `app.py`: Main Streamlit application
- `utils.py`: Utility functions for text extraction and preprocessing
- `requirements.txt`: Python dependencies
- `sample_resumes/`: Folder for sample PDF resumes (optional)