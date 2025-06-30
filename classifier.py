# train_classifier.py
import os, librosa, numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

count = {}

def extract_features(folder):
   # 2 second window, 1 second hop, assuming SAMPLE_RATE = 44100
   #win_length = int(2.0 * SAMPLE_RATE)
   #hop_length = int(1.0 * SAMPLE_RATE)

    X, y = [], []
    for label in ["music","speech"]:
        path = f"data/{label}"
        for fn in os.listdir(path):
            if label not in count.keys():
               count[label] = 0
 
            c = count[label]
            if not fn.endswith(".wav"): continue
            print("Processing:" + fn)
            y_arr, sr = librosa.load(os.path.join(path,fn), sr=44100)
            #print("SR:" + str(sr))
            #mfcc = librosa.feature.mfcc(y=y_arr, sr=sr, n_mfcc=13)
            win_length = int(2.0 * sr)
            hop_length = int(1.0 * sr)

            mfcc = librosa.feature.mfcc(y=y_arr, sr=sr, n_mfcc=13, hop_length=hop_length, n_fft=2048)

            X.append(np.mean(mfcc, axis=1))
            y.append(label)

            c += 1
            count[label] = c
    return np.array(X), np.array(y)

X, y = extract_features("data")
X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=0.2, random_state=42)
clf = SVC(kernel="linear", probability=True)
clf.fit(X_tr, y_tr)

print(classification_report(y_te, clf.predict(X_te)))

for l,c in count.items():
   print(l + ":" + str(c))
print
joblib.dump(clf, "model.pkl")

