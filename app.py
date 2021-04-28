from flask import Flask, render_template, request, send_from_directory
import tensorflow as tf
import os

from static.functions import *

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
model = "Semantic-Model/semantic_model.meta"

tf.reset_default_graph()
sess = tf.InteractiveSession()
cont = 0

# Restore weights
saver = tf.compat.v1.train.import_meta_graph(model)
saver.restore(sess, model[:-5])

graph = tf.compat.v1.get_default_graph()

input = graph.get_tensor_by_name("model_input:0")
seq_len = graph.get_tensor_by_name("seq_lengths:0")
rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
height_tensor = graph.get_tensor_by_name("input_height:0")
width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
logits = tf.compat.v1.get_collection("logits")[0]

# Constants that are saved inside the model itself
WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])
decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)


@app.route("/", methods = ['GET', 'POST'])
def index():
    all_tracks = []
    if request.method == 'GET':
        all_tracks = read_directory(None)
        print(all_tracks)
    return render_template("index.html", all_tracks=all_tracks)


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    image_path2 = ''
    image_path3 = ''

    if request.method == 'POST':

        bpm = request.form.get('bpm', type=float)
        file = request.files['input_file']
        file2 = request.files['input_file2']
        file3 = request.files['input_file3']
        image = pre_processing(file, HEIGHT)

        seq_lengths = [image.shape[2] / WIDTH_REDUCTION]
        prediction = sess.run(decoded,
                              feed_dict={
                                  input: image,
                                  seq_len: seq_lengths,
                                  rnn_keep_prob: 1.0,
                              })
        str_predictions = sparse_tensor_to_strs(prediction)

        array_of_notes = from_prediction_to_note(str_predictions)

        full_audio = recreate_sound(bpm, array_of_notes)
        image_path1 = "static/semple-test/" + file.filename
        if file2.filename != '' and file3.filename != '':
            image_path2 = "static/semple-test/" + file2.filename
            image_path3 = "static/semple-test/" + file3.filename

            image2 = pre_processing(file2, HEIGHT)
            image3 = pre_processing(file3, HEIGHT)
            seq_lengths2 = [image2.shape[2] / WIDTH_REDUCTION]
            seq_lengths3 = [image3.shape[2] / WIDTH_REDUCTION]

            prediction2 = sess.run(decoded,
                                   feed_dict={
                                       input: image2,
                                       seq_len: seq_lengths2,
                                       rnn_keep_prob: 1.0,
                                   })

            prediction3 = sess.run(decoded,
                                   feed_dict={
                                       input: image3,
                                       seq_len: seq_lengths3,
                                       rnn_keep_prob: 1.0,
                                   })
            str_predictions2 = sparse_tensor_to_strs(prediction2)
            str_predictions3 = sparse_tensor_to_strs(prediction3)
            array_of_notes2 = from_prediction_to_note(str_predictions2)
            array_of_notes3 = from_prediction_to_note(str_predictions3)

            full_audio2 = recreate_sound(bpm, array_of_notes2)
            full_audio3 = recreate_sound(bpm, array_of_notes3)
            full_audio = full_audio.overlay(full_audio2).overlay(full_audio3)
        local_counter = str(update_counter())

        full_audio.export("static/full_audio" + local_counter + ".wav", format="wav")
        audio_path = "static/full_audio" + local_counter + ".wav"
        except_track = 'full_audio' + local_counter + ".wav"
        all_tracks = read_directory(except_track)
    return render_template('index.html', path=audio_path, image_path1=image_path1, image_path2=image_path2,
                           image_path3=image_path3, bpm=bpm, scroll='your_music', all_tracks=all_tracks)


if __name__ == "__main__":
    app.run()
