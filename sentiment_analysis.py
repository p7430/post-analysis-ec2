import boto3
from transformers import pipeline
from boto3.dynamodb.conditions import Attr
import logging
from datetime import datetime
from decimal import Decimal

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('bsky-posts')

# Initialize NLP pipelines
sentiment_analysis = pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment')
ner = pipeline('ner', model='dslim/bert-base-NER')

def process_text(text):
    """Process text and return sentiment and NER results"""
    try:
        # Skip empty text
        if not text or not text.strip():
            return None

        # Check if text is primarily Chinese/Japanese/Korean
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            logger.info(f"Skipping CJK text: {text[:50]}...")
            return {
                'sentiment': {'label': 'LABEL_1', 'score': Decimal('0.5')},  # Neutral sentiment
                'named_entities': []
            }

        sentiment_result = sentiment_analysis(text)
        ner_result = ner(text)
        
        # Convert sentiment score to Decimal
        sentiment_result[0]['score'] = Decimal(str(sentiment_result[0]['score']))
        
        # Clean up NER results and convert scores to Decimal
        cleaned_ner = [{
            'word': item['word'],
            'entity': item['entity'],
            'score': Decimal(str(item['score']))
        } for item in ner_result]
        
        return {
            'sentiment': sentiment_result[0],
            'named_entities': cleaned_ner
        }
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        logger.error(f"Problematic text ({len(text)} chars): {text[:100]}...")
        return None

while True:
    try:
        items = []
        last_evaluated_key = None
        
        # Scan with pagination
        while True:
            if last_evaluated_key:
                response = table.scan(
                    FilterExpression=~Attr('analysis_data').exists(),
                    ProjectionExpression='post_id, #txt, #ts, indexed_at',
                    ExpressionAttributeNames={
                        '#txt': 'text',
                        '#ts': 'timestamp'
                    },
                    ExclusiveStartKey=last_evaluated_key,
                    Limit=10
                )
            else:
                response = table.scan(
                    FilterExpression=~Attr('analysis_data').exists(),
                    ProjectionExpression='post_id, #txt, #ts, indexed_at',
                    ExpressionAttributeNames={
                        '#txt': 'text',
                        '#ts': 'timestamp'
                    },
                    Limit=10
                )
            
            current_items = response.get('Items', [])
            items.extend(current_items)
            
            last_evaluated_key = response.get('LastEvaluatedKey')
            logger.info(f"Scan returned {len(current_items)} items")
            logger.info(f"Last evaluated key: {last_evaluated_key}")
            
            if not last_evaluated_key or len(items) >= 10:  # Stop if we have enough items or no more pages
                break
        
        # Sort items by timestamp
        sorted_items = sorted(items, key=lambda x: x.get('timestamp', ''))

        if not sorted_items:
            logger.info("No unanalyzed items found. Checking again immediately...")
            continue

        for item in sorted_items:
            text = item.get('text', '')
            if not text:
                logger.warning(f"No text found for post {item.get('post_id')}")
                continue
            
            try:
                # Process the text
                analysis_results = process_text(text)
                if not analysis_results:
                    continue

                # Update the item with analysis results
                table.update_item(
                    Key={
                        'post_id': item['post_id'],
                        'timestamp': item['timestamp']
                    },
                    UpdateExpression="""
                        SET sentiment = :s, 
                            named_entities = :n, 
                            analysis_data = :a,
                            analyzed_at = :t
                    """,
                    ExpressionAttributeValues={
                        ':s': analysis_results['sentiment'],
                        ':n': analysis_results['named_entities'],
                        ':a': True,
                        ':t': datetime.utcnow().isoformat()
                    }
                )
                
                logger.info(f"Updated post {item['post_id']} with sentiment: {analysis_results['sentiment']} and named_entities: {analysis_results['named_entities']}")
                
            except Exception as e:
                logger.error(f"Error processing post {item.get('post_id')}: {str(e)}")
                continue
        
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
        continue