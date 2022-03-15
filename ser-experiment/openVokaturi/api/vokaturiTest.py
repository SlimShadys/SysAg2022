# measure_wav.py
# Paul Boersma 2019-06-02
#
# A sample script that uses the OpenVokaturi library to extract the emotions from
# a wav file on disk. The file can contain a mono or stereo recording.
#
# Call syntax:
#   python3 measure_wav.py path_to_sound_file.wav

import sys
import os
import scipy.io.wavfile
import pandas as pd
from tqdm import tqdm

import Vokaturi

print("Loading library...")

path = sys.path[0]

Vokaturi.load(path.replace("api", "lib") + "/open/win/OpenVokaturi-3-4-win64.dll")
print("Analyzed by: %s" % Vokaturi.versionAndLicense())

print("Loading dataset...")
if(os.getcwd().endswith("api")):
    datasetDirectory = "../../../Datasets"
else:
    datasetDirectory = "../Datasets"

count = 0
skipped = 0
badAudios = 0

df = pd.read_csv("{}/val_opera7_all.csv".format(datasetDirectory), sep=";", index_col=False)

dfVokaturi = pd.DataFrame(columns=['NOME_FILE','NEUTRAL','HAPPY','SAD','ANGRY','FEAR', 'CORRECT', 'VALIDATION', 'BAD AUDIOS', 'SKIPPED AUDIOS'])
dfStats = pd.DataFrame(columns=['Neutrality','Happiness','Sadness','Anger','Fearful'])

for index, row in tqdm(df.iterrows()):

    if(row['NOME_FILE'].split("/")[1].startswith("surprised")):
        skipped += 1
        continue

    if(row['NOME_FILE'].split("/")[1].startswith("disgust")):
        skipped += 1
        continue

    file_name = datasetDirectory + '/OPERA7_wav' + row['NOME_FILE']

    #print("Reading sound file: {}".format(row['NOME_FILE'].split("/")[0]))

    (sample_rate, samples) = scipy.io.wavfile.read(file_name)
    #print("Audio file sample rate %.3f Hz" % sample_rate)

    #print("Allocating Vokaturi sample array...")
    buffer_length = len(samples)
    #print("%d samples, %d channels" % (buffer_length, samples.ndim))
    c_buffer = Vokaturi.SampleArrayC(buffer_length)
    if samples.ndim == 1:
        c_buffer[:] = samples[:] / 32768.0  # mono
    else:
        c_buffer[:] = 0.5*(samples[:,0]+0.0+samples[:,1]) / 32768.0  # stereo

    #print("Creating VokaturiVoice...")
    voice = Vokaturi.Voice(sample_rate, buffer_length)

    #print("Filling VokaturiVoice with samples...")
    voice.fill(buffer_length, c_buffer)

    #print("Extracting emotions from VokaturiVoice...")
    quality = Vokaturi.Quality()
    emotionProbabilities = Vokaturi.EmotionProbabilities()
    voice.extract(quality, emotionProbabilities)

    if quality.valid:
        #print("Neutral: %.3f" % emotionProbabilities.neutrality)
        #print("Happy: %.3f" % emotionProbabilities.happiness)
        #print("Sad: %.3f" % emotionProbabilities.sadness)
        #print("Angry: %.3f" % emotionProbabilities.anger)
        #print("Fear: %.3f" % emotionProbabilities.fear)
        neutrality = '%.3f' % emotionProbabilities.neutrality
        happiness = '%.3f' % emotionProbabilities.happiness
        sadness = '%.3f' % emotionProbabilities.sadness
        anger = '%.3f' % emotionProbabilities.anger
        fear = '%.3f' % emotionProbabilities.fear

        list = [emotionProbabilities.neutrality,emotionProbabilities.happiness,emotionProbabilities.sadness,emotionProbabilities.anger,emotionProbabilities.fear]
        maxIndex = list.index(max(list))

        num = list[maxIndex]

        if(num == emotionProbabilities.neutrality):
            correct = 'neutral'
        elif(num == emotionProbabilities.happiness):
            correct = 'happy'
        elif(num == emotionProbabilities.sadness):
            correct = 'sad'
        elif(num == emotionProbabilities.anger):
            correct = 'anger'
        elif(num == emotionProbabilities.fear):
            correct = 'fearful'

        if(row['NOME_FILE'].split("/")[1].split("_")[0] == correct):
            isCorrect = 'True'
            count += 1
        else:
            isCorrect = 'False'
            
    else:
        badAudios += 1
        continue

    dfVokaturi = dfVokaturi.append({
        'NOME_FILE': row['NOME_FILE'],
        'NEUTRAL': neutrality,
        'HAPPY': happiness,
        'SAD': sadness,
        'ANGRY': anger,
        'FEAR': fear,
        'CORRECT': isCorrect}, ignore_index=True)

    #print("\n===================================================================================\n")

    voice.destroy()

val = (count / len(dfVokaturi)) * 100

dfVokaturi.iat[0,7] = "Rapporto correttezza: {validation_accuracy:.3f}%".format(validation_accuracy=val)
dfVokaturi.iat[0,8] = F"Audio nulli: {badAudios}"
dfVokaturi.iat[0,9] = F"Emozioni non presenti in Vokaturi: {skipped}"
dfVokaturi.to_csv(path + "/val_opera7_all_vokaturi_new.csv", sep=";", index=False)
