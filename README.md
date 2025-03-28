# Book-to-Podcast-AI

Book-to-Podcast-AI is an AI-powered application that converts books into engaging educational podcasts. It uses advanced natural language processing and text-to-speech technology to create interactive conversations between a teacher and student.

## Features

- Convert PDF, EPUB, and DOCX files into educational podcasts
- Natural conversation-style format with teacher and student roles
- Real-time text-to-speech conversion
- Automatic summarization and key point extraction
- Engaging educational content with real-world examples
- Customizable voice settings for both teacher and student

## Setup and Installation

1. Install Python 3.8 or higher
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up OpenAI API key:
- Create an `.env` file in the project root
- Add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Upload a book file (PDF, EPUB, or DOCX)
2. The application will:
   - Process the text
   - Generate a conversation script
   - Convert text to speech
   - Create an MP3 podcast file

## Voice Configuration

The application uses two distinct voices:

### Teacher (Ms. Ann)
- Voice: `nova`
- Speed: 0.95x
- Accent: British English
- Style: Warm but authoritative

### Student (Adam)
- Voice: `echo`
- Speed: 1.1x
- Accent: American English
- Style: Curious and enthusiastic

## Project Structure

```
book-to-podcast-ai/
├── app.py              # Main application file
├── templates/          # Template files
├── static/            # Static assets
├── requirements.txt   # Python dependencies
└── .env              # Environment variables
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for the GPT-4 and text-to-speech APIs
- Flask for the web framework
- PyPDF2 for PDF processing
- python-docx for DOCX processing
- ebooklib for EPUB processing
