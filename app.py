from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
import joblib

app = Flask(__name__)

# Load model dan scaler
model = load_model('mlp_gym_model.h5')
scaler = joblib.load('scaler.pkl')

# Mapping label output
label_map = {0: "Push Pull Legs", 1: "Full Body Workout", 2: "Upper Lower"}
video_id = {
    "Push Pull Legs": "TJHZI3McmE4",
    "Upper Lower": "41VF8vdnJTY",
    "Full Body Workout": "_xQf1dtg24E"
}

# Data gerakan per program
program_gerakan = {
    "Push Pull Legs": {
        "Push": ["Incline Bench Press", "Chest Fly", "Cable Fly", "Shoulder Press", "Lateral Raises", "Rear Delt", "Tricep Pushdown"],
        "Pull": ["Lat Pulldown", "T-bar row", "Single Arm Pulldown", "Bicep Curl", "Hammer Curl"],
        "Leg": ["Squat", "Leg Press", "Hamstring Curl", "Leg Curl"],
        "notes": "Silahkan cari gerakannya di internet berdasarkan nama gerakan tersebut! Program latihan ini dilakukan dengan memisahkan latihan Push di hari pertama, Pull di hari berikutnya, serta melakukan latihan Leg di hari setelah Push dan Pull."
    },
    "Full Body Workout": {
        "Full Body": ["Incline Bench Press", "Chest Fly", "Cable Fly", "Shoulder Press", "Lateral Raises", "Rear Delt", "Lat Pulldown", "T-Bar Row", "Single Arm Pulldown", "Bicep Curl", "Hammer Curl", "Tricep Overhead", "Tricep Pushdown", "Squat", "Leg Press", "Hamstring Curl", "Leg Curl", "Plank", "Leg Raises", "Russian Twist"],
        "notes": "Silahkan cari gerakannya di internet berdasarkan nama gerakan tersebut! Program latihan ini melatih semua otot dalam satu sesi sekaligus."
    },
    "Upper Lower": {
        "Upper": ["Incline Bench Press", "Chest Fly", "Cable Fly", "Shoulder Press", "Lateral Raises", "Rear Delt", "Lat Pulldown", "T-Bar Row", "Single Arm Pulldown", "Bicep Curl", "Hammer Curl", "Tricep Overhead", "Tricep Pushdown"],
        "Lower": ["Squat", "Lunges", "Bulgarian Squat", "Leg Press", "Leg Curl", "Hamstring Curl", "Plank", "Leg Raises", "Russian Twist"],
        "notes": "Silahkan cari gerakannya di internet berdasarkan nama gerakan tersebut! Program latihan ini dilakukan dengan memisahkan latihan Upper (menyatukan gerakan Push dan Pull dalam satu sesi) di hari pertama, Lower di hari berikutnya."
    }
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Ambil data dari form
        profesi = request.form['profesi']
        frekuensi = float(request.form['frekuensi'])
        durasi = float(request.form['durasi'])
        kesibukan = request.form['kesibukan']

        # Profesi manual (pakai list dari training)
        profesi_list = ['sekolah', 'mahasiswa', 'karyawan', 'ibu rumah tangga', 'pns']
        profesi_encoded = profesi_list.index(profesi) if profesi in profesi_list else 0

        # Encoding kesibukan
        kesibukan_map = {'rendah': 1, 'sedang': 2, 'tinggi': 3}
        kesibukan_encoded = kesibukan_map.get(kesibukan, 2)
        
        # Pisahkan fitur numerik yang diskalakan saat training
        numerik = np.array([[frekuensi, durasi]])  # hanya 2 fitur

        # Skala 2 fitur numerik
        numerik_scaled = scaler.transform(numerik)

        # Gabungkan dengan fitur lainnya (profesi, kesibukan)
        X_scaled = np.hstack(([profesi_encoded, kesibukan_encoded], numerik_scaled[0]))
        X_scaled = np.array([X_scaled])  # bentuk (1, 4)

        # Prediksi
        pred_probs = model.predict(X_scaled)[0]
        pred = np.argmax(pred_probs)
        label = label_map[pred]
        gerakan = program_gerakan[label]
        
        # Gabungkan label dengan probabilitasnya
        probabilitas_kelas = {
            label_map[i]: round(float(prob), 4)
            for i, prob in enumerate(pred_probs)
        }

        return render_template('result.html', label=label, gerakan=gerakan, probabilitas=probabilitas_kelas, video_id=video_id)

    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)