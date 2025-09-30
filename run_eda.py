"""
Exploratory Data Analysis (EDA) for IT Support Ticket Data
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/logs/eda.log'),
        logging.StreamHandler()
    ]
)

def load_processed_data(data_path: str) -> pd.DataFrame:
    """Load the processed ticket data."""
    try:
        df = pd.read_csv(data_path)
        logging.info(f"Loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def plot_ticket_distribution(df: pd.DataFrame, output_dir: str) -> None:
    """Plot distribution of tickets by type, priority, and queue."""
    try:
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot ticket type distribution
        type_counts = df['type'].value_counts()
        axes[0].pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0].set_title('Ticket Type Distribution')
        
        # Plot priority distribution
        priority_order = sorted(df['priority'].unique())
        sns.countplot(data=df, x='priority', order=priority_order, ax=axes[1])
        axes[1].set_title('Ticket Priority Distribution')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Plot queue distribution (top 10 most common)
        queue_counts = df['queue'].value_counts().head(10)
        sns.barplot(y=queue_counts.index, x=queue_counts.values, ax=axes[2])
        axes[2].set_title('Top 10 Ticket Queues')
        axes[2].set_xlabel('Count')
        axes[2].set_ylabel('Queue')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ticket_distributions.png'))
        plt.close()
        logging.info("Saved ticket distribution plots")
        
    except Exception as e:
        logging.error(f"Error in plot_ticket_distribution: {str(e)}")
        raise

def plot_language_distribution(df: pd.DataFrame, output_dir: str) -> None:
    """Analyze and plot language distribution."""
    try:
        if 'language' in df.columns:
            plt.figure(figsize=(10, 6))
            lang_counts = df['language'].value_counts()
            sns.barplot(x=lang_counts.values, y=lang_counts.index)
            plt.title('Ticket Distribution by Language')
            plt.xlabel('Count')
            plt.ylabel('Language')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'language_distribution.png'))
            plt.close()
            logging.info("Saved language distribution plot")
            
    except Exception as e:
        logging.warning(f"Could not create language distribution plot: {str(e)}")

def generate_wordcloud(text_series: pd.Series, title: str, output_path: str) -> None:
    """Generate and save a word cloud from text data."""
    try:
        text = ' '.join(text_series.dropna().astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(title, fontsize=16)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        logging.info(f"Generated word cloud: {output_path}")
    except Exception as e:
        logging.warning(f"Could not generate word cloud: {str(e)}")

def analyze_tag_distribution(df: pd.DataFrame, output_dir: str) -> None:
    """Analyze and plot tag distribution."""
    try:
        # Get all tag columns
        tag_columns = [col for col in df.columns if col.startswith('tag_')]
        if tag_columns:
            # Count occurrences of each tag
            tag_counts = {}
            for col in tag_columns:
                for tag in df[col].dropna().unique():
                    if tag != 'Unknown':
                        tag_counts[tag] = tag_counts.get(tag, 0) + (df[col] == tag).sum()
            
            if tag_counts:
                # Convert to DataFrame for easier plotting
                tag_df = pd.DataFrame({
                    'tag': list(tag_counts.keys()),
                    'count': list(tag_counts.values())
                }).sort_values('count', ascending=False).head(15)  # Top 15 tags
                
                plt.figure(figsize=(12, 8))
                sns.barplot(data=tag_df, x='count', y='tag')
                plt.title('Top 15 Most Common Tags')
                plt.xlabel('Count')
                plt.ylabel('Tag')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'tag_distribution.png'))
                plt.close()
                logging.info("Saved tag distribution plot")
                
    except Exception as e:
        logging.warning(f"Could not create tag distribution plot: {str(e)}")

def main():
    try:
        # Set up directories
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, 'data', 'processed', 'processed_tickets.csv')
        output_dir = os.path.join(base_dir, 'outputs', 'eda_plots')
        os.makedirs(output_dir, exist_ok=True)
        
        logging.info("Starting EDA process...")
        
        # Load the processed data
        df = load_processed_data(data_path)
        
        # Basic info
        logging.info("\n=== Dataset Info ===")
        logging.info(f"Number of tickets: {len(df)}")
        logging.info("\n=== First 5 rows ===")
        logging.info(df.head().to_string())
        
        # Generate visualizations
        plot_ticket_distribution(df, output_dir)
        plot_language_distribution(df, output_dir)
        analyze_tag_distribution(df, output_dir)
        
        # Generate word clouds for text columns
        text_columns = ['subject', 'body', 'answer']
        for col in text_columns:
            if col in df.columns:
                generate_wordcloud(
                    df[col],
                    f'Word Cloud for {col.capitalize()}',
                    os.path.join(output_dir, f'wordcloud_{col}.png')
                )
        
        # Save summary statistics
        summary_stats = df.describe(include='all').T
        summary_stats.to_csv(os.path.join(output_dir, 'summary_statistics.csv'))
        
        logging.info("\n=== EDA Completed Successfully ===")
        logging.info(f"Visualizations saved to: {output_dir}")
        
    except Exception as e:
        logging.error(f"Error in EDA process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
