import os
import io
import tempfile
from pydub import AudioSegment
import ffmpeg_static
from groq import Groq

# Initialize Groq client - will use GROQ_API_KEY from environment variables
client = Groq()

def convert_audio_to_wav(audio_data, original_filename=None):
    """
    Convert any audio data to WAV format (16kHz, mono, 16-bit).
    
    Args:
        audio_data (bytes): The binary audio data.
        original_filename (str, optional): Original filename for format detection.
        
    Returns:
        bytes: The converted WAV file as binary data.
    """
    try:
        # Get the path to the ffmpeg executable
        ffmpeg_path = ffmpeg_static.get_ffmpeg_exe()
        AudioSegment.converter = ffmpeg_path
        
        # Create a temporary file to store the input audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=original_filename if original_filename else '.audio') as temp_in:
            temp_in.write(audio_data)
            temp_in_path = temp_in.name
        
        # Detect input format from file extension if provided
        input_format = None
        if original_filename:
            input_format = os.path.splitext(original_filename)[1].lower().replace('.', '')
        
        # Load the audio file
        try:
            if input_format and input_format in ['mp3', 'wav', 'flac', 'ogg', 'aac', 'm4a', 'wma']:
                sound = AudioSegment.from_file(temp_in_path, format=input_format)
            else:
                sound = AudioSegment.from_file(temp_in_path)
        except Exception as e:
            print(f"Error loading audio: {e}")
            # Try again letting pydub detect format
            sound = AudioSegment.from_file(temp_in_path)
            
        # Set the desired parameters
        sound = sound.set_frame_rate(16000)  # Set sample rate to 16000 Hz
        sound = sound.set_channels(1)        # Set number of channels to 1 (mono)
        sound = sound.set_sample_width(2)    # Set sample width to 16 bits (2 bytes)
        
        # Create a BytesIO object to hold the WAV data
        wav_data = io.BytesIO()
        
        # Export the audio as WAV with PCM S16LE codec to the BytesIO object
        sound.export(wav_data, format="wav", codec="pcm_s16le")
        
        # Clean up the temporary input file
        os.unlink(temp_in_path)
        
        # Get the binary data from the BytesIO object
        return wav_data.getvalue()
        
    except Exception as e:
        print(f"Error converting audio: {e}")
        raise

def transcribe_audio(audio_data, filename=None, language="en"):
    """
    Transcribe audio data using Groq.
    
    Args:
        audio_data (bytes): The binary audio data.
        filename (str, optional): Filename for the audio.
        language (str): Language code for transcription.
        
    Returns:
        str: The transcription text.
    """
    try:
        # Create a temporary WAV file for Groq
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        # Open the temporary file and send to Groq
        with open(temp_file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(filename if filename else "audio.wav", file.read()),
                model="whisper-large-v3-turbo",
                prompt="Please separate different speakers and label them.",
                language=language,
                response_format="verbose_json",
            )
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        
        return transcription
        
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        raise

def main(args):
    """
    Digital Ocean Functions handler.
    
    Expected input:
    - audio_data: Base64 encoded audio data (if directly uploading binary)
    - audio_url: URL to the audio file (alternative to audio_data)
    - filename: Original filename (optional, helps with format detection)
    - language: Language code (default: "en")
    
    Returns:
        dict: Response with the transcription.
    """
    try:
        # Check if we have audio data or URL
        audio_data = None
        filename = args.get('filename', 'audio.wav')
        language = args.get('language', 'en')
        
        if 'audio_data' not in args and 'audio_url' not in args:
            return {
                'statusCode': 400,
                'body': {'error': 'Missing required parameter: either audio_data or audio_url'}
            }
        
        # If we have direct audio data
        if 'audio_data' in args:
            # In Digital Ocean Functions, binary data is passed as Base64 string
            import base64
            audio_data = base64.b64decode(args['audio_data'])
        
        # If we have a URL, download the audio
        elif 'audio_url' in args:
            import requests
            response = requests.get(args['audio_url'])
            if response.status_code != 200:
                return {
                    'statusCode': 400,
                    'body': {'error': f'Failed to download audio from URL. Status code: {response.status_code}'}
                }
            audio_data = response.content
            
        # Convert the audio to WAV
        wav_data = convert_audio_to_wav(audio_data, filename)
        
        # Transcribe the audio
        transcription = transcribe_audio(wav_data, filename, language)
        
        # Return the transcription
        return {
            'statusCode': 200,
            'body': {
                'transcription_text': transcription.text,
                'full_transcription': transcription
            }
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': {'error': str(e)}
        }
