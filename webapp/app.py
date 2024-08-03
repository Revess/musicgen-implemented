from flask import Flask, request, jsonify, send_from_directory, render_template
from transformers import pipeline
import scipy
import os
import uuid

synthesiser = pipeline("text-to-audio", "facebook/musicgen-small", device="cpu")

# Initialize the Flask application
app = Flask(__name__)

# Path to store the generated audio files
AUDIO_FOLDER = os.path.join(os.getcwd(), 'static', 'audio')
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Replace this function with your actual audio generation logic
def generate_audio(prompt):
    music = synthesiser(prompt, forward_params={"do_sample": True})

    audio_file_path = os.path.join(AUDIO_FOLDER, f"{uuid.uuid4()}.wav")
    scipy.io.wavfile.write(audio_file_path, rate=music["sampling_rate"], data=music["audio"])

    return audio_file_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form.get('prompt')
        if prompt:
            try:
                audio_file_path = generate_audio(prompt)
                audio_filename = os.path.basename(audio_file_path)
                download_link = f'/download/{audio_filename}'
                return jsonify({'success': True, 'download_link': download_link})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
    return render_template('index.html', download_link=None)

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(AUDIO_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)