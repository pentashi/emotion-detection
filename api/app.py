from flask import Flask, request, jsonify
from emotion_model import predict_emotion

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file provided'}), 400
    file_path = f"./uploads/{file.filename}"
    file.save(file_path)
    emotion = predict_emotion(file_path)
    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True)
