import os
import pyttsx3
import pyaudio
import wave
from io import BytesIO
from flask import Flask, request, jsonify, send_file
from pydub import AudioSegment
from pydub.playback import play
from flask_cors import CORS
from secret_key import azure_openai_key
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import whisper
from langchain_openai import AzureChatOpenAI
from langchain.schema.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tempfile import NamedTemporaryFile
import threading
from gtts import gTTS

# Azure OpenAI Configuration
os.environ["AZURE_OPENAI_API_KEY"] = azure_openai_key
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://21etc-m3br78jg-francecentral.openai.azure.com/"
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "gpt-35-turbo"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-08-01-preview"

# Initialize Azure Chat Model
model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# LangGraph Workflow Setup
workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    chain = prompt | model
    response = chain.invoke(state)
    return {"messages": response}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
compiled_workflow = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

prompt = ChatPromptTemplate.from_messages([
    ("system", "You talk like a psychologist. Ask follow-up questions mandatorily and do not suggest anything like solution or some remedy quickly. After asking some questions and understanding the situation properly recommend some soothing music playlists, motivational videos or meditation sessions. Never keep everything in the same chat first ask questions then after an answer by the user then suggest something according to user's emotional situation. Answer all questions to the best of your ability."),
    MessagesPlaceholder(variable_name="messages"),
])

# Microphone recording configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

# Initialize components
analyzer = SentimentIntensityAnalyzer()
whisper_model = whisper.load_model("base")
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 150)
tts_engine.setProperty("volume", 0.9)

# Conversation State Management
class ConversationState:
    def __init__(self):
        self.anxiety_indicators = 0
        self.stress_indicators = 0
        self.depression_indicators = 0
        self.conversation_history = []
        self.last_sentiment = None
        self.triggers_identified = []
        self.coping_methods_discussed = []

conversation_states = {}

# Context Analysis Functions
def analyze_message_context(message, user_id):
    if user_id not in conversation_states:
        conversation_states[user_id] = ConversationState()
    
    state = conversation_states[user_id]
    state.conversation_history.append(message)
    
    # Analyze for mental health indicators
    anxiety_keywords = ["anxious", "worried", "panic", "overwhelmed", "nervous", 
    "fearful", "uneasy", "restless", "agitated", "apprehensive", "fear", 
    "racing thoughts", "cant relax", "dread"]
    stress_keywords = ["stress", "stressed", "pressure", "tension", "exhausted", 
    "burnout", "frustrated", "irritated", "impatient", "overworked", "cant cope", 
    "deadline", "too much"]
    depression_keywords = ["sad", "hopeless", "worthless", "empty", "depressed", 
    "miserable", "lonely", "guilty", "isolated", "unhappy", "numb", "tired all time", 
    "no interest", "cant sleep", "sleep too much", "no energy", "suicidal", "give up"]
    
    message_lower = message.lower()
    
    # Update indicators based on message content
    state.anxiety_indicators += sum(1 for word in anxiety_keywords if word in message_lower)
    state.stress_indicators += sum(1 for word in stress_keywords if word in message_lower)
    state.depression_indicators += sum(1 for word in depression_keywords if word in message_lower)
    
    return state

def should_provide_resources(state):
    return (
        state.anxiety_indicators >= 2 or 
         state.stress_indicators >= 2 or 
         state.depression_indicators >= 2
    )

def recommend_resources_based_on_context(state):
    if not should_provide_resources(state):
        return {}
    
    primary_concern = max(
        ("anxiety", state.anxiety_indicators),
        ("stress", state.stress_indicators),
        ("depression", state.depression_indicators),
        key=lambda x: x[1]
    )[0]
    
    resources = {
        "anxiety": {
            "articles": [
                {"name": "Understanding and Managing Anxiety", 
                "url": "https://www.helpguide.org/articles/anxiety/anxiety-disorders-and-anxiety-attacks.htm"}
            ],
            "youtube": [
                {"name": "Guided Anxiety Relief Exercise", 
                "url": "https://youtu.be/LBEAJcp0lTs"}
            ],
            "music": {"name": "Calming Anxiety Playlist", 
            "url": "https://open.spotify.com/playlist/0eU3ubPAnqeSMi9K3YKVpC"}
        },
        "stress": {
            "articles": [
                {"name": "Stress Management Techniques", 
                "url": "https://www.helpguide.org/articles/stress/stress-management.htm"}
            ],
            "youtube": [
                {"name": "Stress Relief Meditation", 
                "url": "https://youtu.be/sfSDQRdIvTc"}
            ],
            "music": {"name": "Stress Relief Playlist", 
            "url": "https://open.spotify.com/album/0SO7kjCHZ6Trf6xUDBnfoR"}
        },
        "depression": {
            "articles": [
                {"name": "Coping with Depression", 
                "url": "https://www.helpguide.org/articles/depression/coping-with-depression.htm"}
            ],
            "youtube": [
                {"name": "Depression Management Techniques", 
                "url": "https://youtu.be/MQB3UUTh8aQ"}
            ],
            "music": {"name": "Mood Lifting Playlist", 
            "url": "https://open.spotify.com/playlist/75uOpSRxXAYm6ZXP8yZ2DR"}
        }
    }
    
    return resources.get(primary_concern, {})

# Utility Functions
def classify_sentiment(scores):
    """
    Classify sentiment based on the compound score.
    """
    compound = scores["compound"]
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

ALLOWED_EXTENSIONS = {'wav', 'mp3'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def record_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                       rate=RATE, input=True,
                       frames_per_buffer=CHUNK)
    frames = []

    print("Recording...")
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording complete.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    return WAVE_OUTPUT_FILENAME

@app.route('/api/message', methods=['POST'])
def handle_message():
    data = request.get_json()
    user_message = data.get("message", "")
    user_id = data.get("user_id", "default_user")

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    try:
        # Analyze context and update state
        state = analyze_message_context(user_message, user_id)
        
        # Get sentiment for activity scheduling
        sentiment_scores = analyzer.polarity_scores(user_message)
        sentiment = classify_sentiment(sentiment_scores)
        
        # Process message with GPT
        input_messages = [HumanMessage(user_message)]
        state_input = {"messages": input_messages}
        output = compiled_workflow.invoke(state_input, config)
        gpt_response = output["messages"][-1].content.strip()
        
        response_data = {
            "response": gpt_response
        }
        
        # Only add recommendations if we have sufficient context
        if should_provide_resources(state):
            recommendations = recommend_resources_based_on_context(state)
            if recommendations:
                response_data["recommendations"] = recommendations
        
        return jsonify(response_data)

    except Exception as e:
        app.logger.error(f"Error processing the request: {e}", exc_info=True)
        return jsonify({"error": "Failed to process message"}), 500

@app.route('/api/voice-message', methods=['POST'])
def handle_voice_message():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    user_id = request.form.get("user_id", "default_user")

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        with NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            # Save uploaded file temporarily
            file.save(temp_file.name)

            # Convert MP3 to WAV if necessary
            if temp_file.name.endswith(".mp3"):
                audio = AudioSegment.from_mp3(temp_file.name)
                temp_file.close()
                temp_file.name = temp_file.name.rsplit('.', 1)[0] + ".wav"
                audio.export(temp_file.name, format="wav")

            # Perform transcription
            transcription = whisper_model.transcribe(temp_file.name)

        # Analyze transcription
        state = analyze_message_context(transcription["text"], user_id)
        sentiment_scores = analyzer.polarity_scores(transcription["text"])
        sentiment = classify_sentiment(sentiment_scores)

        # Process with GPT
        input_messages = [HumanMessage(transcription["text"])]
        state_input = {"messages": input_messages}
        output = compiled_workflow.invoke(state_input, config)
        gpt_response = output["messages"][-1].content.strip()

        response_data = {
            "response": gpt_response,
            "transcription": transcription["text"]
        }

        if should_provide_resources(state):
            recommendations = recommend_resources_based_on_context(state)
            if recommendations:
                response_data["recommendations"] = recommendations

        return jsonify(response_data)

    return jsonify({"error": "Invalid file format"}), 400

@app.route('/api/record', methods=['POST'])
def handle_recording():
    try:
        user_id = request.form.get("user_id", "default_user")
        audio_file = record_audio()
        transcription = whisper_model.transcribe(audio_file)
        os.remove(audio_file)
        
        # Analyze context and sentiment
        state = analyze_message_context(transcription["text"], user_id)
        sentiment_scores = analyzer.polarity_scores(transcription["text"])
        sentiment = classify_sentiment(sentiment_scores)
        
        # Process with GPT
        input_messages = [HumanMessage(transcription["text"])]
        state_input = {"messages": input_messages}
        output = compiled_workflow.invoke(state_input, config)
        gpt_response = output["messages"][-1].content.strip()
        
        response_data = {
            "response": gpt_response,
            "transcription": transcription["text"]
        }
        
        if should_provide_resources(state):
            recommendations = recommend_resources_based_on_context(state)
            if recommendations:
                response_data["recommendations"] = recommendations
        
        return jsonify(response_data)
        
    except Exception as e:
        app.logger.error(f"Error recording audio: {e}", exc_info=True)
        return jsonify({"error": "Failed to record audio"}), 500

@app.route('/api/synthesize', methods=['POST'])
def synthesize_speech():
    try:
        data = request.get_json()
        text = data.get("text")
        if not text:
            return jsonify({"error": "Text is required"}), 400

        with NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            temp_file_path = temp_audio_file.name

        # Use gTTS to synthesize speech
        tts = gTTS(text=text, lang='en')
        tts.save(temp_file_path)

        return send_file(
            temp_file_path,
            mimetype="audio/mpeg",
            as_attachment=True,
            download_name="synthesized_speech.mp3"
        )

    except Exception as e:
        app.logger.error(f"Error synthesizing speech: {e}", exc_info=True)
        return jsonify({"error": "Failed to synthesize speech"}), 500

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.route('/api/sentiment', methods=['POST'])
def analyze_sentiment():
    """
    API endpoint to analyze sentiment of a given message using VADER.
    """
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    try:
        sentiment_scores = analyzer.polarity_scores(user_message)
        sentiment = classify_sentiment(sentiment_scores)
        return jsonify({
            "message": user_message,
            "sentiment": sentiment,
            "scores": sentiment_scores
        })
    except Exception as e:
        app.logger.error(f"Error processing the request: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Error Handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(Exception)
def unhandled_exception(error):
    app.logger.error(f"Unhandled exception: {error}", exc_info=True)
    return jsonify({"error": "An unexpected error occurred"}), 500

# Create required directories
def setup_directories():
    directories = ["uploads", "logs"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

# Application startup configuration
def configure_app():
    # Setup logging
    import logging
    from logging.handlers import RotatingFileHandler
    
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    file_handler = RotatingFileHandler(
        'logs/mental_health_assistant.log',
        maxBytes=10240,
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Mental Health Assistant startup')

if __name__ == '__main__':
    setup_directories()
    configure_app()
    app.run(debug=True, port=3000)