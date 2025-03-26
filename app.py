from flask import Flask, render_template, request, jsonify, send_file, session
import os
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import PyPDF2
import docx
import ebooklib
from ebooklib import epub
import re
import openai
from gtts import gTTS
import time
from io import BytesIO
import base64
import tiktoken

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configuration
app.static_folder = 'static'
UPLOAD_FOLDER = '/tmp/uploads'  # Use /tmp for Render
AUDIO_FOLDER = '/tmp/audio'     # Use /tmp for Render
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # Reduce to 8MB for free tier

# Cleanup function for temporary files
def cleanup_temp_files():
    for folder in [UPLOAD_FOLDER, AUDIO_FOLDER]:
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            try:
                if os.path.isfile(filepath) and time.time() - os.path.getmtime(filepath) > 3600:  # 1 hour
                    os.remove(filepath)
            except Exception as e:
                print(f"Error cleaning up {filepath}: {e}")

@app.before_request
def before_request():
    cleanup_temp_files()  # Clean old files before each request

# OpenAI setup
openai.api_key = os.getenv('OPENAI_API_KEY')

# Allowed extensions
ALLOWED_EXTENSIONS = {'pdf', 'epub', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Text extraction functions
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as file:
            try:
                reader = PyPDF2.PdfReader(file)
                if len(reader.pages) == 0:
                    return "Error: PDF appears to be empty"
                
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            except PyPDF2.errors.PdfReadError:
                return extract_text_from_pdf_fallback(file_path)
    except Exception as e:
        return f"Error: Could not read PDF ({str(e)})"
    return text.strip()

def extract_text_from_pdf_fallback(file_path):
    """Fallback PDF extraction method"""
    try:
        import pdftotext
        with open(file_path, "rb") as f:
            pdf = pdftotext.PDF(f)
        return "\n\n".join(pdf)
    except:
        return "Error: PDF could not be read with either method"

def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text)
    except Exception as e:
        return f"Error: Could not read DOCX ({str(e)})"

def extract_text_from_epub(file_path):
    text = []
    try:
        book = epub.read_epub(file_path)
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                content = item.get_content().decode('utf-8')
                text.append(re.sub('<[^<]+?>', '', content))
    except Exception as e:
        return f"Error: Could not read EPUB ({str(e)})"
    return "\n".join(text)

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.epub'):
        return extract_text_from_epub(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    return "Error: Unsupported file format"

# Text processing
def process_text(text):
    text = text.replace('\r\n', '\n')
    return re.sub(r'\s+', ' ', text).strip()

def count_tokens(text, model="gpt-3.5-turbo-0125"):
    """Count tokens using the appropriate encoder for the model"""
    try:
        encoder = tiktoken.encoding_for_model(model)
        return len(encoder.encode(text))
    except:
        # Fallback for unknown models: approximate 4 characters per token
        return len(text) // 4

def summarize_long_text(text, max_tokens=2000):
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        total_tokens = len(encoding.encode(text))
        
        if total_tokens <= max_tokens:
            return text

        # More aggressive chunking for memory optimization
        chunk_size = max_tokens // 2  # Smaller chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in re.split(r'(?<=[.!?])\s+', text):
            sentence_tokens = len(encoding.encode(sentence))
            if current_length + sentence_tokens > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_length += sentence_tokens
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        # Process chunks with delay to avoid rate limits
        summaries = []
        for chunk in chunks:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Summarize the following text concisely while keeping important details:"},
                        {"role": "user", "content": chunk}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.5
                )
                summaries.append(response.choices[0].message['content'].strip())
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"Error processing chunk: {e}")
                continue

        # Combine summaries if needed
        combined_summary = " ".join(summaries)
        if len(encoding.encode(combined_summary)) > max_tokens:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Create a final concise summary of these combined summaries:"},
                        {"role": "user", "content": combined_summary}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.5
                )
                return response.choices[0].message['content'].strip()
            except Exception as e:
                print(f"Error in final summarization: {e}")
                return combined_summary[:max_tokens*4]  # Fallback to truncation
        
        return combined_summary
    except Exception as e:
        print(f"Error in summarization: {e}")
        return text[:max_tokens*4]  # Emergency fallback

def generate_podcast_script(text, grade_level="middle school"):
    # Check text length and summarize if needed
    token_count = count_tokens(text)
    max_safe_tokens = 6000  # Conservative limit for input to avoid context length issues
    
    if token_count > max_safe_tokens:
        print(f"Text too long ({token_count} tokens). Summarizing...")
        text = summarize_long_text(text, max_tokens=2000)
        print(f"Summarized to {count_tokens(text)} tokens")
    
    try:
        # Ensure we're within safe limits for the prompt
        content_text = text[:7000]  # Reduced from 8000 to be more conservative
        
        prompt = f"""
        Create an engaging, interactive educational podcast between:
        - Ms. Johnson (40-year-old female teacher, warm but authoritative, British accent)
        - Alex (15-year-old male student, curious and enthusiastic, American accent)
        
        Format requirements:
        [Teacher] Actual dialogue (use British English spellings)
        [Student] Actual dialogue (use casual American English)
        
        Content to discuss:
        {content_text}
        
        Guidelines:
        1. Make it sound like a natural conversation with interruptions and follow-up questions
        2. Ms. Johnson should explain concepts clearly with real-world examples
        3. Alex should ask thoughtful questions and make occasional jokes
        4. Include:
           - Warm introduction
           - 3-5 main discussion points from the text
           - Summary conclusion
           - Funny outtake at the end
        5. Use these exact tags for each speaker: [Teacher] and [Student]
        6. Keep exchanges short (1-3 sentences per turn)
        7. Add personality through word choice and speaking style
        """

        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",  # Using GPT-4 for better conversational quality
            messages=[
                {"role": "system", "content": "You are a talented podcast script writer specializing in educational content."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2500,
            temperature=0.8,  # Slightly higher for more creative responses
            top_p=0.9
        )
        
        script = response.choices[0].message.content
        
        # Post-processing to ensure consistent formatting
        script = re.sub(r'\[Teacher\]\s*', '[Teacher] ', script)
        script = re.sub(r'\[Student\]\s*', '[Student] ', script)
        
        # Add some metadata about the voices
        voice_metadata = {
            "teacher": {
                "name": "Ms. Johnson",
                "gender": "female",
                "accent": "British",
                "pace": "moderate",
                "tone": "warm, authoritative"
            },
            "student": {
                "name": "Alex",
                "gender": "male",
                "accent": "American",
                "pace": "slightly faster",
                "tone": "enthusiastic, curious"
            }
        }
        
        return script, voice_metadata
        
    except Exception as e:
        print(f"Script generation error: {e}")
        return None, None

def text_to_speech_with_voices(script):
    """Enhanced TTS with better voice differentiation"""
    try:
        # Process script to maintain conversation order
        dialogue_parts = []
        lines = script.split('\n')
        
        # Extract dialogue parts with speaker information
        current_speaker = None
        current_text = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('[Teacher]'):
                # Save previous speaker's text
                if current_speaker and current_text:
                    dialogue_parts.append((current_speaker, current_text))
                    current_text = ""
                
                current_speaker = 'teacher'
                current_text = line[9:].strip()  # Remove tag
            elif line.startswith('[Student]'):
                # Save previous speaker's text
                if current_speaker and current_text:
                    dialogue_parts.append((current_speaker, current_text))
                    current_text = ""
                
                current_speaker = 'student'
                current_text = line[9:].strip()  # Remove tag
            elif current_speaker:  # Continue previous speaker's text
                current_text += " " + line
        
        # Add the last dialogue part
        if current_speaker and current_text:
            dialogue_parts.append((current_speaker, current_text))
        
        # Generate audio for each dialogue part in sequence
        combined = BytesIO()
        
        for speaker, text in dialogue_parts:
            if not text.strip():
                continue
                
            part_audio = BytesIO()
            tts = gTTS(
                text=text,
                lang='en',
                tld='co.uk' if speaker == 'teacher' else 'us',
                slow=False,
                lang_check=False
            )
            tts.write_to_fp(part_audio)
            part_audio.seek(0)
            
            # Add to combined audio
            combined.write(part_audio.getvalue())
            # Add a brief pause between speakers
            combined.write(b'\x00' * 22050)  # 0.5s silence
        
        combined.seek(0)
        return base64.b64encode(combined.getvalue()).decode()
    
    except Exception as e:
        print(f"TTS generation error: {e}")
        return None

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Unsupported file type'})
        
        # Verify file size
        file.seek(0, os.SEEK_END)
        if file.tell() > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({'success': False, 'error': 'File too large'})
        file.seek(0)
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        text = extract_text(file_path)
        if text.startswith("Error:"):
            return jsonify({'success': False, 'error': text})
        
        book_title = os.path.splitext(filename)[0]
        with open(os.path.join(UPLOAD_FOLDER, f"{book_title}.txt"), 'w', encoding='utf-8') as f:
            f.write(text)
        
        return jsonify({
            'success': True,
            'book': {
                'title': book_title,
                'text': text[:200] + ('...' if len(text) > 200 else '')
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': f"Server error: {str(e)}"})

@app.route('/generate_podcast/<grade_level>', methods=['POST'])
def generate_podcast(grade_level):
    try:
        cleanup_temp_files()  # Clean up before processing
        if 'text' not in session:
            return jsonify({'error': 'No text found in session'}), 400

        text = session['text']
        if not text:
            return jsonify({'error': 'Empty text'}), 400

        # Limit text size for free tier
        max_text_length = 50000  # ~50KB
        if len(text) > max_text_length:
            text = text[:max_text_length]

        # Generate podcast script with memory-optimized summarization
        script = generate_podcast_script(text, grade_level)
        if not script:
            return jsonify({'error': 'Failed to generate script'}), 500

        # Generate audio with error handling
        try:
            tts = gTTS(text=script, lang='en', slow=False)
            audio_path = os.path.join(AUDIO_FOLDER, f'podcast_{int(time.time())}.mp3')
            tts.save(audio_path)
            
            with open(audio_path, 'rb') as audio_file:
                audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
            
            os.remove(audio_path)  # Clean up immediately after use
            return jsonify({'audio': audio_data, 'script': script})
        except Exception as e:
            print(f"Error generating audio: {e}")
            return jsonify({'error': 'Failed to generate audio'}), 500

    except Exception as e:
        print(f"Error in generate_podcast: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '').strip()
        book_title = data.get('book_title', '').strip()

        if not message or not book_title:
            return jsonify({'success': False, 'error': 'Missing parameters'})

        book_path = os.path.join(UPLOAD_FOLDER, f"{book_title}.txt")
        if not os.path.exists(book_path):
            return jsonify({'success': False, 'error': 'Book not found'})

        with open(book_path, 'r', encoding='utf-8') as f:
            book_content = f.read()
        
        conversation_history = session.get('conversation_history', [])
        conversation_history.append({"role": "user", "content": message})
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": f"You are discussing '{book_title}'. Answer based on the book content."
                },
                *conversation_history[-5:]
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": ai_response})
        session['conversation_history'] = conversation_history
        
        return jsonify({
            'success': True,
            'message': {
                'text': ai_response,
                'isAI': True
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)