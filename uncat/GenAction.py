import numpy as np
import os
ACTION_PATH = os.path.join('actions.npy')
DATA_PATH = os.path.join('MP_Data') 
#actions = np.array(['hello', 'thanks', 'iloveyou'])
#np.save(os.path.join(DATA_PATH, ACTION_PATH), actions)
actions = np.load(os.path.join(DA TA_PATH, ACTION_PATH))
print(actions.shape[0])
print(actions)


for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass