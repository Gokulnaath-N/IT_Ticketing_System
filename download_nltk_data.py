"""
Download required NLTK data for text processing
"""
import nltk
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def download_nltk_data():
    """Download required NLTK data packages."""
    try:
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        
        logging.info("Successfully downloaded all required NLTK data")
        return True
    except Exception as e:
        logging.error(f"Error downloading NLTK data: {str(e)}")
        return False

if __name__ == "__main__":
    logging.info("Starting NLTK data download...")
    if download_nltk_data():
        logging.info("NLTK data is ready to use!")
    else:
        logging.error("Failed to download NLTK data")
