# ML
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
import numpy as np

# Database
import models
from database import engine, get_db
from sqlalchemy.orm import Session
models.Base.metadata.create_all(bind=engine)

#Sign Vision Model
class SVModel:
    def __init__(self):
        self.model_path = os.path.join('')
        #self.model_path = os.path.join('Server/actions.h5')
        self.sequences = 30
        self.frames = 30
        self.keypoints = 1662
        
        if os.path.exists(self.model_path):
            print(f"Loading existing model: {self.model_path}")
            self.model = load_model(self.model_path)
        else:
            print("Fetching data from database...")
            actions, data = self._db_get_actions()
        #    print("Creating new model...")
        #    self.model = self._create_model(actions)
        #    print("Training new model...")
        #    self._train_model(data)
        #    print("Saving new model...")
        #    self.model.save(self.model_path)

    #data = sequence[0:29]frame[0:29]keypoint[0:1661]
    def predict(self, data):
        if data.size == self.sequences and data[0].size == self.frames and data[0][0].size == self.keypoints:
            return self.model.predict(np.expand_dims(data, axis=0))[0]
        else:
            return False

    def _db_get_actions(self):
        with Session(engine) as db:
            actions = db.query(models.Action.action).distinct().all()
            data = db.query(models.Action).all()
        return actions, data

#def read_all_entries():
#    # Create a session
#    db = SessionLocal()
#
#    try:
#        # Query all entries in the Frames table
#        frameSet = db.query(models.Frames).all()
#
#        # Print the entries
#        for frames in frameSet:
#            print(f"    ID: {frames.id}")
#            print(f"Action: {frames.action}")
#            #print(f"Landmarks: {frames.landmarkSet}")
#            print("-" * 30)
#    except Exception as e:
#        print(f"Error occurred: {str(e)}")
#    finally:
#        db.close()
#
## Call the function to read all entries
#read_all_entries()


    def _create_model(self, actions):
        self.model = Sequential()
        self.model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(self.frames,self.keypoints)))
        self.model.add(LSTM(128, return_sequences=True, activation='relu'))
        self.model.add(LSTM(64, return_sequences=False, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(actions.shape[0], activation='softmax'))
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    #def _train_model(self, data):
    #    label_map = {label:num for num, label in enumerate(self.actions)}
    #    log_dir = os.path.join('Logs')
    #    tb_callback = TensorBoard(log_dir=log_dir)
    #    sequences, labels = [], []
    #    for action in self.actions:
    #        for seq_num in range(self.sequences):
    #            window = []
    #            for frame_num in range(self.frames):
    #                res = np.load(os.path.join(self.data_path, action, str(seq_num), "{}.npy".format(frame_num)))
    #                window.append(res)
    #            sequences.append(window)
    #            labels.append(label_map[action])
    #    X = np.array(sequences)
    #    y = to_categorical(labels).astype(int)
    #    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    #    self.model.fit(X_train, y_train, epochs=150, callbacks=[tb_callback])



if __name__ == '__main__':
    model = SVModel()