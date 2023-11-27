import pickle
import os
import math
import copy

def load_Q():
    try:
        Q = None
        path = os.path.join('Q.pkl')
        print("Loading Q from {}".format(path))
        with open(path, 'rb') as f:
            Q = pickle.load(f)

        keys_to_remove = []

        # Create a copy of the dictionary
        Q_copy = copy.deepcopy(Q)

        for key, value in Q_copy.items():
            print('State: ', key)
            for key2, value2 in value.items():
                if isinstance(value2['accuracy'], float) and math.isnan(value2['accuracy']):
                    keys_to_remove.append((key, key2))

                print(key2, 'participation_success: ', value2['participation_success'], 'accuracy: ', value2['accuracy']/value2['count'], 'count: ', value2['count'])

        # # Remove the keys from the original dictionary
        # for key, key2 in keys_to_remove:
        #     del Q[key][key2]
        
        # #remove keys that have 'quantization' in them
        # keys_to_remove = []
        # for key, value in Q.items():
        #     for key2, value2 in value.items():
        #         # print(key2)
        #         if 'quantization' in key2:
        #             keys_to_remove.append((key, key2))
        # for key, key2 in keys_to_remove:
        #     del Q[key][key2]
        
        # #save the updated Q
        # with open(path, 'wb') as f:
        #     pickle.dump(Q, f)
            
        
        # #save the updated Q
        # with open(path, 'wb') as f:
        #     pickle.dump(Q, f)
            

    except Exception as e:
        print("Error in load_Q")
        print(e)

load_Q()
