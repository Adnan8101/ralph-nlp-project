from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import logging
import time
from emotion_models import get_pretrained_model, get_our_trained_model
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging for Vercel
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CRITICAL: Proper Flask initialization for Vercel
app = Flask(__name__, 
           template_folder='../templates',
           static_folder='../static')
CORS(app, origins=['*'])

# Performance tracking
request_count = 0
total_response_time = 0

@app.route('/')
def index():
    """Main route - MUST exist to avoid 404"""
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files properly"""
    return send_from_directory('../static', filename)

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
        
        logger.info(f"Ralph's Analyzer - Processing: {text[:100]}...")
        
        response_data = {
            'success': True,
            'text_length': len(text),
            'model_used': model_type,
            'analysis_timestamp': time.time(),
            'developer': 'Ralph AIML Roll No 9'
        }
        
        # Get predictions based on selected model
        if model_type == 'pretrained':
            model = get_pretrained_model()
            response_data['pretrained'] = model.predict(text)
            
        elif model_type == 'our_trained':
            model = get_our_trained_model()
            response_data['our_trained'] = model.predict(text)
            
        else:  # both models
            pretrained_model = get_pretrained_model()
            our_model = get_our_trained_model()
            
            response_data['pretrained'] = pretrained_model.predict(text)
            response_data['our_trained'] = our_model.predict(text)
        
        # Calculate response time
        response_time = time.time() - start_time
        request_count += 1
        total_response_time += response_time
        
        response_data['response_time'] = round(response_time, 3)
        
        logger.info(f"Analysis completed in {response_time:.3f}s")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in Ralph's analyzer: {str(e)}")
        return jsonify({
            'success': False, 
            'error': 'Internal server error'
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            'status': 'healthy',
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

# CRITICAL: Export app for Vercel
# This is the key fix - Vercel needs this export
if __name__ == '__main__':
    app.run(debug=True)
