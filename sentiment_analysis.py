import json
import boto3
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from opensearchpy import OpenSearch, RequestsHttpConnection
import logging
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_opensearch_credentials():
    try:
        secret_name = "opensearch/master-credentials"
        region_name = "us-east-1"
        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager',
            region_name=region_name
        )
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
        secret = json.loads(get_secret_value_response['SecretString'])
        return secret['username'], secret['password']
    except Exception as e:
        logger.error(f"Failed to retrieve OpenSearch credentials: {str(e)}")
        raise

# OpenSearch configuration
REGION = 'us-east-1'
HOST = 'search-bsky-posts-prod-fo3mg6u3d6zcjmv57tvwpxaaei.us-east-1.es.amazonaws.com'
INDEX_NAME = 'bsky-posts-prod'

# Get credentials and initialize client
MASTER_USER, MASTER_PASSWORD = get_opensearch_credentials()

opensearch_client = OpenSearch(
    hosts=[{'host': HOST, 'port': 443}],
    http_auth=(MASTER_USER, MASTER_PASSWORD),
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    retry_on_timeout=True,
    max_retries=3,
    timeout=30
)

# Initialize sentiment analysis model (BERTweet)
model_name = "vinai/bertweet-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def process_text(text):
    """Process text and return sentiment results"""
    try:
        if not text or not text.strip():
            return None

        # Check if text is primarily Chinese/Japanese/Korean
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            logger.info(f"Skipping CJK text: {text[:50]}...")
            return {
                'sentiment': {'label': 'NEU', 'score': 0.5}
            }

        sentiment_result = sentiment_pipeline(text)[0]
        
        return {
            'sentiment': {
                'label': sentiment_result['label'],
                'score': float(sentiment_result['score'])
            }
        }
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        logger.error(f"Problematic text ({len(text)} chars): {text[:100]}...")
        return None

def process_batch(posts, batch_size=100):
    """Process a batch of posts"""
    for post in posts:
        try:
            text = post.get('_source', {}).get('text', '')
            if not text:
                continue

            analysis_results = process_text(text)
            if not analysis_results:
                continue

            # Update the document with sentiment analysis
            update_body = {
                'doc': {
                    'sentiment_analysis': analysis_results,
                    'analyzed_at': datetime.utcnow().isoformat()
                }
            }

            opensearch_client.update(
                index=INDEX_NAME,
                id=post['_id'],
                body=update_body,
                retry_on_conflict=3
            )

            logger.info(f"Updated post {post['_id']} with sentiment: {analysis_results['sentiment']}")

        except Exception as e:
            logger.error(f"Error processing post {post.get('_id')}: {str(e)}")
            continue

def test_results(num_entries=10):
    """Test function to verify sentiment analysis results"""
    try:
        # Query for posts that have been analyzed
        query = {
            "query": {
                "exists": {
                    "field": "sentiment_analysis"
                }
            },
            "size": num_entries,
            "sort": [{"analyzed_at": "desc"}]
        }

        response = opensearch_client.search(
            index=INDEX_NAME,
            body=query
        )

        hits = response['hits']['hits']
        logger.info(f"\nFound {len(hits)} analyzed posts:")
        
        for hit in hits:
            post = hit['_source']
            logger.info(f"\nPost ID: {hit['_id']}")
            logger.info(f"Text: {post.get('text', '')[:100]}...")
            logger.info(f"Sentiment: {post.get('sentiment_analysis', {}).get('sentiment')}")
            logger.info(f"Analyzed at: {post.get('analyzed_at')}")
            
    except Exception as e:
        logger.error(f"Error testing results: {e}")

def main(test_mode=True, num_test_entries=10):
    """
    Main function with test mode option
    test_mode: If True, only processes a few entries and displays results
    num_test_entries: Number of entries to process in test mode
    """
    logger.info("Starting sentiment analysis...")
    
    try:
        if test_mode:
            logger.info("Running in test mode...")
            
            # Query for unanalyzed posts
            query = {
                "query": {
                    "bool": {
                        "must_not": [
                            {"exists": {"field": "sentiment_analysis"}}
                        ]
                    }
                },
                "sort": [{"timestamp": "asc"}],
                "size": num_test_entries
            }

            response = opensearch_client.search(
                index=INDEX_NAME,
                body=query
            )

            hits = response['hits']['hits']
            
            if not hits:
                logger.info("No unanalyzed posts found.")
                return
                
            logger.info(f"Processing {len(hits)} posts for testing...")
            process_batch(hits)
            
            # Show the results
            logger.info("\nChecking results...")
            test_results(num_test_entries)
            
        else:
            # Original production code
            batch_size = 100
            scroll_size = 1000
            
            while True:
                try:
                    # Query for unanalyzed posts
                    query = {
                        "query": {
                            "bool": {
                                "must_not": [
                                    {"exists": {"field": "sentiment_analysis"}}
                                ]
                            }
                        },
                        "sort": [{"timestamp": "asc"}],
                        "size": scroll_size
                    }

                    # Initialize scroll
                    response = opensearch_client.search(
                        index=INDEX_NAME,
                        body=query,
                        scroll='5m'
                    )

                    scroll_id = response['_scroll_id']
                    hits = response['hits']['hits']

                    while hits:
                        logger.info(f"Processing batch of {len(hits)} posts")
                        
                        # Process posts in parallel using ThreadPoolExecutor
                        with ThreadPoolExecutor(max_workers=4) as executor:
                            for i in range(0, len(hits), batch_size):
                                batch = hits[i:i + batch_size]
                                executor.submit(process_batch, batch, batch_size)

                        # Get next batch using scroll
                        response = opensearch_client.scroll(
                            scroll_id=scroll_id,
                            scroll='5m'
                        )
                        
                        hits = response['hits']['hits']
                        
                    logger.info("No more unanalyzed posts found. Waiting before next check...")
                    time.sleep(60)  # Wait a minute before checking again

                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}")
                    time.sleep(60)  # Wait before retrying
                    continue
                    
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    # Run in test mode with 10 entries
    main(test_mode=True, num_test_entries=10)
    
    # For production, use:
    # main(test_mode=False)