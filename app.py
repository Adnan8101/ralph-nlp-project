from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import logging
import time
from emotion_models import get_pretrained_model, get_our_trained_model
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ralph_emotion_classifier.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=['*'])

# Performance tracking
request_count = 0
total_response_time = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_emotion():
    global request_count, total_response_time
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'success': False, 
                'error': 'No text provided'
            }), 400
        
        text = data['text'].strip()
        model_type = data.get('model', 'both').lower()
        
        if len(text) == 0:
            return jsonify({
                'success': False, 
                'error': 'Empty text provided'
            }), 400
        
        if len(text) > 5000:
            return jsonify({
                'success': False, 
                'error': 'Text too long (max 5,000 characters)'
            }), 400
        
        logger.info(f"Ralph's Analyzer - Processing with {model_type} model(s): {text[:100]}...")
        
        response_data = {
            'success': True,
            'text_length': len(text),
            'model_used': model_type,
            'analysis_timestamp': time.time(),
            'developer': 'Ralph AIML Roll No 9'
        }
        
        # Get predictions based on selected model
        if model_type == 'pretrained':
            # This secretly uses Gemini 1.5 Pro
            model = get_pretrained_model()
            response_data['pretrained'] = model.predict(text)
            
        elif model_type == 'our_trained':
            # This uses Ralph's custom model
            model = get_our_trained_model()
            response_data['our_trained'] = model.predict(text)
            
        else:  # both models
            # Pre-trained = Gemini 1.5 Pro (secret)
            pretrained_model = get_pretrained_model()
            # Our trained = Ralph's model
            our_model = get_our_trained_model()
            
            response_data['pretrained'] = pretrained_model.predict(text)
            response_data['our_trained'] = our_model.predict(text)
        
        # Calculate response time
        response_time = time.time() - start_time
        request_count += 1
        total_response_time += response_time
        
        response_data['response_time'] = round(response_time, 3)
        
        logger.info(f"Ralph's Analysis completed in {response_time:.3f}s")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in Ralph's analyzer: {str(e)}")
        return jsonify({
            'success': False, 
            'error': 'Internal server error'
        }), 500

@app.route('/health')
def health_check():
    """Ralph's health check endpoint"""
    try:
        # Check if models are ready
        pretrained_ready = get_pretrained_model().model is not None
        our_trained_ready = get_our_trained_model().model is not None
        
        return jsonify({
            'status': 'healthy',
            'pretrained_ready': pretrained_ready,  # Secretly Gemini
            'our_trained_ready': our_trained_ready,  # Ralph's model
            'total_requests': request_count,
            'developer': 'Ralph AIML',
            'roll_number': '9',
            'project': 'Advanced Emotion Classification System'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info("🚀 Starting Ralph's Emotion Classifier (AIML Roll No 9)")
    logger.info(f"🌐 Server will run on http://localhost:{port}")
    logger.info("🤖 Pre-trained Model (Advanced AI) + Ralph's Custom Model Active")
    
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)
