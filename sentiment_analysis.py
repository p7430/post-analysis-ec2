import json
import boto3
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from opensearchpy import OpenSearch, RequestsHttpConnection
import logging
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import fasttext
import urllib.request
import os
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_fasttext_model():
    """Download the FastText language identification model if not present"""
    model_path = "lid.176.ftz"
    if not os.path.exists(model_path):
        logger.info("Downloading FastText language identification model...")
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
        urllib.request.urlretrieve(url, model_path)
    return model_path

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

# Initialize sentiment analysis model
model_name = "finiteautomata/bertweet-base-sentiment-analysis"  # Pre-trained for sentiment
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Initialize FastText model
try:
    fasttext.FastText.eprint = lambda x: None  # Suppress FastText warning messages
    model_path = download_fasttext_model()
    ft_model = fasttext.load_model(model_path)
    logger.info("FastText language detection model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load FastText model: {str(e)}")
    raise

def detect_language(text):
    try:
        if not text or text.isspace():
            return 'unknown', 0.0

        # Clean the text
        cleaned_text = text.replace('\n', ' ').strip()
        if not cleaned_text:
            return 'unknown', 0.0

        # Get prediction from FastText and handle the array creation
        labels, probs = ft_model.predict(cleaned_text)
        # Use np.asarray instead of np.array with copy=False
        probs = np.asarray(probs)
        
        # Get the first prediction
        lang = labels[0].replace('__label__', '')
        conf = float(probs[0])

        logger.info(f"Language detection successful: {lang}, {conf}")
        return lang, conf

    except Exception as e:
        logger.error(f"Error detecting language: {str(e)}", exc_info=True)
        return 'unknown', 0.0

def process_text(text, langs=None):
    """Process text and return sentiment results"""
    try:
        if not text or not text.strip():
            return None

        # Detect language with confidence
        detected_lang, confidence = detect_language(text)
        
        if detected_lang != 'en':
            logger.info(f"Marking non-English text (detected: {detected_lang}, conf: {confidence:.2f})")
            return {
                'sentiment': {
                    'label': f'NON_ENGLISH_{detected_lang.upper()}',
                    'score': 0.0
                },
                'language': {
                    'detected': detected_lang,
                    'confidence': confidence
                }
            }

        # Only analyze English text
        sentiment_result = sentiment_pipeline(text)[0]
        
        # Add debug logging
        logger.info(f"Raw sentiment result: {sentiment_result}")
        
        # Map the labels properly
        label = sentiment_result['label']
        if label == 'NEG':
            mapped_label = 'NEG'
        elif label == 'POS':
            mapped_label = 'POS'
        else:  # 'NEU'
            mapped_label = 'NEU'
        
        return {
            'sentiment': {
                'label': mapped_label,
                'score': float(sentiment_result['score'])
            }
        }
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        logger.error(f"Problematic text ({len(text)} chars): {text[:100]}...")
        return {
            'sentiment': {
                'label': 'ANALYSIS_ERROR',
                'score': 0.0
            }
        }

def process_batch(posts, batch_size=100):
    """Process a batch of posts"""
    for post in posts:
        try:
            source = post.get('_source', {})
            text = source.get('text', '')
            langs = source.get('langs', [])
            
            if not text:
                # Mark empty posts
                update_body = {
                    'doc': {
                        'sentiment_analysis': {'sentiment': {'label': 'EMPTY', 'score': 0.0}},
                        'analyzed_at': datetime.utcnow().isoformat()
                    }
                }
                opensearch_client.update(
                    index=INDEX_NAME,
                    id=post['_id'],
                    body=update_body,
                    retry_on_conflict=3
                )
                continue

            # Handle non-English posts
            try:
                # If langs is empty, rely only on detection
                # If langs contains 'en', treat as English
                # Otherwise, check both langs and detection
                if not langs:  # Empty langs array
                    detected_lang, confidence = detect_language(text)
                    if detected_lang == 'en':
                        # Process as English
                        analysis_results = process_text(text)
                    else:
                        logger.info(f"Marking non-English text (detected: {detected_lang})")
                        update_body = {
                            'doc': {
                                'sentiment_analysis': {
                                    'sentiment': {'label': f'NON_ENGLISH_{detected_lang.upper()}', 'score': 0.0}
                                },
                                'analyzed_at': datetime.utcnow().isoformat(),
                                'detected_language': detected_lang
                            }
                        }
                        opensearch_client.update(
                            index=INDEX_NAME,
                            id=post['_id'],
                            body=update_body,
                            retry_on_conflict=3
                        )
                        continue
                elif 'en' in langs:  # Explicitly marked as English
                    analysis_results = process_text(text)
                else:  # Has langs but not English
                    detected_lang, confidence = detect_language(text)
                    if detected_lang == 'low_confidence':
                        logger.info(f"Low confidence in language detection for text: {text}. Skipping sentiment analysis.")
                        update_body = {
                            'doc': {
                                'sentiment_analysis': {
                                    'sentiment': {'label': 'LOW_CONFIDENCE', 'score': 0.0}
                                },
                                'analyzed_at': datetime.utcnow().isoformat(),
                                'detected_language': detected_lang
                            }
                        }
                    else:
                        logger.info(f"Marking non-English text (detected: {detected_lang}, provided: {langs})")
                        update_body = {
                            'doc': {
                                'sentiment_analysis': {
                                    'sentiment': {'label': f'NON_ENGLISH_{detected_lang.upper()}', 'score': 0.0}
                                },
                                'analyzed_at': datetime.utcnow().isoformat(),
                                'detected_language': detected_lang
                            }
                        }
                    opensearch_client.update(
                        index=INDEX_NAME,
                        id=post['_id'],
                        body=update_body,
                        retry_on_conflict=3
                    )
                    continue
            except Exception as e:
                logger.warning(f"Language detection failed for text: '{text[:100]}...': {str(e)}", exc_info=True)
                # Even if language detection fails, try to process as English
                try:
                    analysis_results = process_text(text)
                    update_body = {
                        'doc': {
                            'sentiment_analysis': {'sentiment': analysis_results},
                            'analyzed_at': datetime.utcnow().isoformat()
                        }
                    }
                except Exception as sentiment_error:
                    logger.error(f"Sentiment analysis failed after language detection error: {str(sentiment_error)}")
                    update_body = {
                        'doc': {
                            'sentiment_analysis': {'sentiment': {'label': 'PROCESSING_ERROR', 'score': 0.0}},
                            'analyzed_at': datetime.utcnow().isoformat()
                        }
                    }
                
                opensearch_client.update(
                    index=INDEX_NAME,
                    id=post['_id'],
                    body=update_body,
                    retry_on_conflict=3
                )
                continue

            # Update successful analysis
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

def test_model_outputs():
    test_texts = [
        "I love this! It's amazing!",
        "I hate this, it's terrible",
        "This is okay, nothing special"
    ]
    
    logger.info("Testing model outputs directly:")
    for text in test_texts:
        result = sentiment_pipeline(text)[0]
        logger.info(f"Text: {text}")
        logger.info(f"Raw output: {result}")
        logger.info("---")

def main(test_mode=True, num_test_entries=10):
    """
    Main function with test mode option
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
            
            test_model_outputs()
            
        else:
            # Production settings
            batch_size = 250  # Increased from 100
            scroll_size = 2500  # Increased from 1000
            max_workers = 8  # Increased from 4 to utilize all vCPUs effectively
            
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

                    # Initialize scroll with longer timeout
                    response = opensearch_client.search(
                        index=INDEX_NAME,
                        body=query,
                        scroll='15m'  # Increased from 5m
                    )

                    scroll_id = response['_scroll_id']
                    hits = response['hits']['hits']
                    
                    processed_count = 0
                    start_time = time.time()

                    while hits:
                        batch_start = time.time()
                        logger.info(f"Processing batch of {len(hits)} posts")
                        
                        # Process posts in parallel
                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            futures = []
                            for i in range(0, len(hits), batch_size):
                                batch = hits[i:i + batch_size]
                                futures.append(executor.submit(process_batch, batch, batch_size))
                            
                            # Wait for all futures to complete
                            for future in as_completed(futures):
                                try:
                                    future.result()
                                except Exception as e:
                                    logger.error(f"Batch processing error: {str(e)}")

                        processed_count += len(hits)
                        batch_time = time.time() - batch_start
                        rate = len(hits) / batch_time
                        
                        logger.info(f"Batch processed at {rate:.2f} posts/second")
                        logger.info(f"Total processed: {processed_count} posts")

                        # Get next batch using scroll
                        response = opensearch_client.scroll(
                            scroll_id=scroll_id,
                            scroll='15m'
                        )
                        
                        hits = response['hits']['hits']

                    total_time = time.time() - start_time
                    avg_rate = processed_count / total_time
                    logger.info(f"Processing complete. Average rate: {avg_rate:.2f} posts/second")
                    
                    if processed_count == 0:
                        logger.info("No more unanalyzed posts found. Waiting before next check...")
                        time.sleep(30)  # Reduced wait time from 60s to 30s
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}")
                    time.sleep(30)  # Reduced wait time
                    continue
                    
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    # Add a quick test before running main
    try:
        test_text = "Hello, this is a test message"
        lang, conf = detect_language(test_text)
        logger.info(f"Test detection result: {lang}, {conf}")
    except Exception as e:
        logger.error(f"Test detection failed: {str(e)}", exc_info=True)
    
    # Production mode
    main(test_mode=False)