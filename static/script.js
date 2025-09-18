class RalphEmotionClassifier {
    constructor() {
        this.initializeElements();
        this.bindEvents();
        this.selectedModel = 'both';
        this.emotionIcons = {
            'joy': '😊', 'sadness': '😢', 'anger': '😠', 'fear': '😨',
            'surprise': '😮', 'disgust': '🤢', 'neutral': '😐', 'love': '❤️'
        };
    }

    initializeElements() {
        this.textInput = document.getElementById('textInput');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.resultsSection = document.getElementById('resultsSection');
        this.loading = document.getElementById('loading');
        this.pretrainedEmotions = document.getElementById('pretrainedEmotions');
        this.ourTrainedEmotions = document.getElementById('ourTrainedEmotions');
        this.dominantEmotion = document.getElementById('dominantEmotion');
    }

    bindEvents() {
        this.analyzeBtn.addEventListener('click', () => this.analyzeText());
        this.clearBtn.addEventListener('click', () => this.clearInput());
        
        document.addEventListener('change', (e) => {
            if (e.target.name === 'model') {
                this.selectedModel = e.target.value;
                this.updateResultsDisplay();
            }
        });
    }

    async analyzeText() {
        const text = this.textInput.value.trim();
        
        if (!text) {
            alert('Please enter some text to analyze!');
            return;
        }

        this.showLoading();
        
        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    text: text,
                    model: this.selectedModel
                })
            });

            const data = await response.json();
            
            if (data.success) {
                this.displayResults(data);
            } else {
                throw new Error(data.error || 'Analysis failed');
            }
        } catch (error) {
            console.error('Error:', error);
            alert(`Error: ${error.message}`);
        }
        
        this.hideLoading();
    }

    displayResults(data) {
        const pretrainedSection = document.getElementById('pretrainedResult');
        const ourTrainedSection = document.getElementById('ourTrainedResult');
        
        // Handle pretrained model results (secretly Gemini)
        if (data.pretrained && (this.selectedModel === 'both' || this.selectedModel === 'pretrained')) {
            this.renderEmotions(this.pretrainedEmotions, data.pretrained);
            pretrainedSection.style.display = 'block';
        } else {
            pretrainedSection.style.display = 'none';
        }
        
        // Handle our trained model results (Ralph's model)
        if (data.our_trained && (this.selectedModel === 'both' || this.selectedModel === 'our_trained')) {
            this.renderEmotions(this.ourTrainedEmotions, data.our_trained);
            ourTrainedSection.style.display = 'block';
        } else {
            ourTrainedSection.style.display = 'none';
        }
        
        const modelComparison = document.querySelector('.model-comparison');
        if (this.selectedModel === 'both') {
            modelComparison.style.gridTemplateColumns = '1fr 1fr';
        } else {
            modelComparison.style.gridTemplateColumns = '1fr';
        }
        
        this.renderSummary(data.pretrained, data.our_trained);
        this.resultsSection.style.display = 'block';
        
        this.resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    renderEmotions(container, emotions) {
        container.innerHTML = '';
        
        const sortedEmotions = Object.entries(emotions)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 8);

        sortedEmotions.forEach(([emotion, confidence]) => {
            const emotionItem = this.createEmotionItem(emotion, confidence);
            container.appendChild(emotionItem);
        });
    }

    createEmotionItem(emotion, confidence) {
        const item = document.createElement('div');
        item.className = 'emotion-item';
        
        const percentage = Math.round(confidence * 100);
        const icon = this.emotionIcons[emotion.toLowerCase()] || '😐';
        
        item.innerHTML = `
            <div class="emotion-icon">${icon}</div>
            <div class="emotion-info">
                <div class="emotion-name">${this.capitalizeFirst(emotion)}</div>
                <div class="emotion-bar">
                    <div class="emotion-fill" style="width: ${percentage}%"></div>
                </div>
            </div>
            <div class="emotion-percentage">${percentage}%</div>
        `;
        
        return item;
    }

    renderSummary(pretrainedData, ourTrainedData) {
        let dominantEmotion = 'neutral';
        let description = 'Analysis complete';
        
        if (pretrainedData && ourTrainedData) {
            const pretrainedTop = this.getTopEmotion(pretrainedData);
            const ourTrainedTop = this.getTopEmotion(ourTrainedData);
            dominantEmotion = pretrainedTop.confidence > ourTrainedTop.confidence ? pretrainedTop.emotion : ourTrainedTop.emotion;
            description = `Both models analyzed the text`;
        } else if (pretrainedData) {
            const topEmotion = this.getTopEmotion(pretrainedData);
            dominantEmotion = topEmotion.emotion;
            description = `Pre-trained model analysis`;
        } else if (ourTrainedData) {
            const topEmotion = this.getTopEmotion(ourTrainedData);
            dominantEmotion = topEmotion.emotion;
            description = `Ralph's custom model analysis`;
        }
        
        const icon = this.emotionIcons[dominantEmotion.toLowerCase()] || '🤔';
        
        this.dominantEmotion.innerHTML = `
            <div class="emotion-large">${icon}</div>
            <div class="emotion-text">${this.capitalizeFirst(dominantEmotion)}</div>
            <div class="emotion-desc">${description}</div>
        `;
    }

    getTopEmotion(emotions) {
        const [emotion, confidence] = Object.entries(emotions)
            .reduce(([maxEmotion, maxConf], [emotion, conf]) => 
                conf > maxConf ? [emotion, conf] : [maxEmotion, maxConf]
            );
        return { emotion, confidence };
    }

    updateResultsDisplay() {
        if (this.resultsSection.style.display === 'block') {
            const pretrainedSection = document.getElementById('pretrainedResult');
            const ourTrainedSection = document.getElementById('ourTrainedResult');
            const modelComparison = document.querySelector('.model-comparison');
            
            if (this.selectedModel === 'pretrained') {
                pretrainedSection.style.display = 'block';
                ourTrainedSection.style.display = 'none';
                modelComparison.style.gridTemplateColumns = '1fr';
            } else if (this.selectedModel === 'our_trained') {
                pretrainedSection.style.display = 'none';
                ourTrainedSection.style.display = 'block';
                modelComparison.style.gridTemplateColumns = '1fr';
            } else {
                pretrainedSection.style.display = 'block';
                ourTrainedSection.style.display = 'block';
                modelComparison.style.gridTemplateColumns = '1fr 1fr';
            }
        }
    }

    showLoading() {
        this.loading.style.display = 'block';
        this.resultsSection.style.display = 'none';
        this.analyzeBtn.disabled = true;
    }

    hideLoading() {
        this.loading.style.display = 'none';
        this.analyzeBtn.disabled = false;
    }

    clearInput() {
        this.textInput.value = '';
        this.resultsSection.style.display = 'none';
        this.textInput.focus();
    }

    capitalizeFirst(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new RalphEmotionClassifier();
});
