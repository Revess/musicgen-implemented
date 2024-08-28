from flask import Flask, request, jsonify, send_from_directory, render_template, abort
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import os, torchaudio, uuid, glob, time, shutil
from apscheduler.schedulers.background import BackgroundScheduler

simple_debug = False

if not simple_debug:
    model_prompt = MusicGen.get_pretrained("facebook/musicgen-stereo-small", device='cpu')
    model_melody = MusicGen.get_pretrained("facebook/musicgen-stereo-melody", device='cpu')
    print('Done loading models')

# Initialize the Flask application
app = Flask(__name__, static_url_path='/static')

# Path to store the generated audio files
AUDIO_FOLDER = os.path.join(os.getcwd(), 'static', 'audio')
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Function to delete folders not modified in the last 24 hours
def clean_old_folders():
    current_time = time.time()
    for folder_name in os.listdir(AUDIO_FOLDER):
        folder_path = os.path.join(AUDIO_FOLDER, folder_name)
        if os.path.isdir(folder_path):
            last_modified_time = os.path.getmtime(folder_path)
            # If the folder is older than 24 hours (86400 seconds), delete it
            if current_time - last_modified_time > 86400: 
                shutil.rmtree(folder_path)
                print(f"Deleted folder: {folder_path}")

# Schedule the clean_old_folders function to run every day
scheduler = BackgroundScheduler()
scheduler.add_job(func=clean_old_folders, trigger="interval", days=1)
scheduler.start()

# Replace this function with your actual audio generation logic
def generate_audio(prompt, temperature, topk, topp, cfg, samples, duration, dropped=False, userid=None):
    if simple_debug:
        print('loading prompt')
        print(prompt, temperature, topk, topp, cfg, samples, duration)
        print([f"{AUDIO_FOLDER}/{userid}/8b83f79f-80e9-4b5a-b62a-0d4124d1c809.wav"] * samples)
        return [f"{AUDIO_FOLDER}/{userid}/8b83f79f-80e9-4b5a-b62a-0d4124d1c809.wav"] * samples
    
    model_prompt.set_generation_params(
        duration=duration,
        temperature=temperature,
        top_k=topk,
        top_p=topp,
        cfg_coef=cfg
    )

    audio_file_paths = [os.path.join(AUDIO_FOLDER, userid, f"{uuid.uuid4()}") for _ in range(samples)]
    txt_file = '\n'.join(audio_file_paths)
    os.system(f'echo "{txt_file}" > {os.path.join(AUDIO_FOLDER, userid, "recent_audio.txt")}')
    if dropped:
        melody, sr = torchaudio.load(dropped)
        for audio_file_path in audio_file_paths:
            wav = model_melody.generate_with_chroma([prompt], melody[None].expand(1, -1, -1), sr)
            audio_write(audio_file_path, wav[0].cpu(), model_prompt.sample_rate, strategy="loudness")
    else:
        for audio_file_path in audio_file_paths:
            wav = model_prompt.generate([prompt])
            audio_write(audio_file_path, wav[0].cpu(), model_prompt.sample_rate, strategy="loudness")

    return [f"{audio_file_path}.wav" for audio_file_path in audio_file_paths ]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = str(request.form.get('prompt')).lower()
        userid = str(request.form.get('userid'))
        print(userid)
        if userid == 'undefined' or userid not in os.listdir(AUDIO_FOLDER):
            userid = str(uuid.uuid4())
            os.makedirs(f"{AUDIO_FOLDER}/{userid}", exist_ok=True)
        else:
            for file_ in glob.glob(f"{AUDIO_FOLDER}/{userid}/*"):
                os.unlink(file_)
        # Handle the uploaded audio file
        audio_file = request.files.get('audioInput')
        dropped = False
        if audio_file and audio_file.filename != '':
            dropped = os.path.join(AUDIO_FOLDER, userid, f"{uuid.uuid4()}.wav")
            audio_file.save(dropped)
        if prompt:
            try:
                temperature = float(request.form.get('Temperature'))
                topk = int(request.form.get('Top K'))
                topp = float(request.form.get('Top P'))
                cfg = int(request.form.get('Classifier Free Guidance'))
                samples = int(request.form.get('Samples'))
                duration = int(request.form.get('Duration'))
                audio_file_paths = generate_audio(prompt, temperature, topk, topp, cfg, samples, duration, dropped=dropped, userid=userid)
                audio_filenames = [os.path.basename(audio_file_path) for audio_file_path in audio_file_paths]
                download_links = [f'/download/{userid}/{audio_filename}' for audio_filename in audio_filenames]
                return jsonify({'success': True, 'download_links': download_links, 'userid': userid})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
    return render_template('index.html', download_link=None)

@app.route('/api/download_links', methods=['POST'])
def get_download_links():
    userid = str(request.json.get('userid'))
    print(userid)
    if userid == 'undefined' or userid not in os.listdir(AUDIO_FOLDER):
        userid = str(uuid.uuid4())
        os.makedirs(f"{AUDIO_FOLDER}/{userid}", exist_ok=True)
    try:
        if os.path.exists(os.path.join(AUDIO_FOLDER, userid, 'recent_audio.txt')):
            with open(os.path.join(AUDIO_FOLDER, userid, 'recent_audio.txt'), "r") as file_:
                audio_paths = [path for path in file_.read().split('\n') if os.path.exists(path)]
                audio_filenames = [os.path.basename(audio_file_path) for audio_file_path in audio_paths]
        else:
            audio_filenames = []
        download_links = [f'/download/{userid}/{audio_filename}.wav' for audio_filename in audio_filenames]
        return jsonify({'success': True, 'download_links': download_links, 'userid': userid})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/download/<userid>/<filename>')
def download(userid, filename):
    return send_from_directory(os.path.join(AUDIO_FOLDER, userid), filename, as_attachment=True)
    
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(os.path.join(app.root_path, 'static'), filename)

if __name__ == '__main__':
    try:
        app.run(debug=True, port=5500)
    finally:
        scheduler.shutdown()