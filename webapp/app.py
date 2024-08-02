import os
from flask import Flask, request, send_from_directory, render_template
import os
import uuid

# Initialize the Flask application
app = Flask(__name__)

# Path to store the generated audio files
AUDIO_FOLDER = os.path.join(os.getcwd(), 'static', 'audio')
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Replace this function with your actual audio generation logic
def generate_audio(prompt):
    # Dummy implementation for demonstration purposes
    audio_file_path = os.path.join(AUDIO_FOLDER, f"{uuid.uuid4()}.wav")
    with open(audio_file_path, 'wb') as f:
        f.write(b'')  # Placeholder for audio data
    return audio_file_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form.get('prompt')
        if prompt:
            audio_file_path = generate_audio(prompt)
            audio_filename = os.path.basename(audio_file_path)
            return render_template('index.html', download_link=f'/download/{audio_filename}')
    return render_template('index.html', download_link=None)

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(AUDIO_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)