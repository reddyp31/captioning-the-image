from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from image_captioning_model import ImageCaptioningModel

app = Flask(_name_)

# Load the tokenizer and create the captioning model
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=15000, oov_token='<unk>')
tokenizer_vocab = pickle.load(open('vocab_coco.file', 'rb'))
tokenizer.set_vocabulary(tokenizer_vocab)

encoder = TransformerEncoderLayer(512, 1)
decoder = TransformerDecoderLayer(512, 512, 8)

cnn_model = CNN_Encoder()
caption_model = ImageCaptioningModel(
    cnn_model=cnn_model, encoder=encoder, decoder=decoder, image_aug=image_augmentation,
)
caption_model.load_weights('model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img_url = request.form['img_url']
    im = Image.open(requests.get(img_url, stream=True).raw)
    im = im.convert('RGB')
    im.save('tmp.jpg')

    pred_caption = generate_caption('tmp.jpg', add_noise=False)
    return jsonify({'predicted_caption': pred_caption})

if _name_ == '_main_':
    app.run(debug=True)