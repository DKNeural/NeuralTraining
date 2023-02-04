
from scipy.io import wavfile
import argparse
import numpy as np
import random
import json

def save_wav(name, data):
    wavfile.write(name, 44100, data.flatten().astype(np.float32))

def save_wav_dont_flatten(name, data):
    wavfile.write(name, 44100, data.astype(np.float32))
    
def normalize(data):
    data_max = max(data)
    data_min = min(data)
    data_norm = max(data_max,abs(data_min))
    if data_norm == 0:
        print("[WARNING]: Audio file appears to contain all 0's, indicating a completely silent wav file. Check your out.wav file.")
    return data / data_norm

def sliceOnMod(input_data, mod):
    # Split the data on a modulus.

    # type cast to an integer the modulus
    mod = int(mod)

    # Split the data into 100 pieces
    input_split = np.array_split(input_data, 100)

    val_input_data = []
    # Traverse the range of the indexes of the input signal reversed and pop every 5th for val
    for i in reversed(range(len(input_split))):
        if i%mod == 0:
            # Store the validation data
            val_input_data.append(input_split[i])
            # Remove the validation data from training
            input_split.pop(i)

    # Flatten val_data down to one dimension and concatenate
    val_input_data = np.concatenate(val_input_data)

    # Concatinate b back together
    training_input_data = np.concatenate(input_split)
    return (training_input_data, val_input_data)


def conditionedWavParse(args):
    '''
    Note: Assumes all .wav files are mono, float32, no metadata
    '''
    # Open the configuration
    with open(args.parameterize, "r") as read_file:
        data = json.load(read_file)

    # Test to see if there are separate test data sets
    seprateTestSet = True
    try:
        a = data['Data Sets'][0]["TestClean"]
        a = data['Data Sets'][0]["TestTarget"]
    except KeyError:
        seprateTestSet = False
        print("The test set and validation set are the same.")


    params = data["Number of Parameters"]

    # Load and Preprocess Data ###########################################
    all_clean_test = np.array([[]]*(params+1)) # 1 channel for audio, n channels per parameters

    for ds in data["Data Sets"]:

        # Load and Preprocess Data
        in_rate, in_data = wavfile.read(ds["TrainingClean"])       

        #If stereo data, use channel 0
        if len(in_data.shape) > 1:
            print("[WARNING] Stereo data detected for in_data, only using first channel (left channel)")
            in_data = in_data[:,0]

        # Convert PCM16 to FP32
        if in_data.dtype == "int16":
            in_data = in_data/32767
            print("In data converted from PCM16 to FP32")  
        
        clean_data = in_data.astype(np.float32).flatten()

        # If Desired Normalize the data
        if (args.normalize):
            clean_data = normalize(clean_data).reshape(len(clean_data),1)

        # Make the Training and validation split
        # Split the data on a twenty percent mod
        in_train, in_val = sliceOnMod(clean_data, args.mod_split)


        # Process the Test Data. If it's a separate set, process that set. 
        # If there is not a separeate set, use the validation data for testing.
        if (seprateTestSet):
            test_in_rate, test_in_data = wavfile.read(ds["TestClean"])
            in_test = test_in_data.astype(np.float32).flatten()
            if (args.normalize):
                in_test = normalize(in_test).reshape(len(test_in_data),1)
        else:
            in_test = in_val
            if (args.normalize):
                in_test = normalize(in_test).reshape(len(in_test),1)

        # Initialize lists to handle the number of parameters
        params_test = []

        # Create a list of np arrays of the parameter values
        for val in ds["Parameters"]:
            # Create the parameter arrays
            params_test.append(np.array([val]*len(in_test)))

        # Convert the lists to numpy arrays
        params_test = np.array(params_test)
        
        # Reformat for normalized data
        if (args.normalize):
            in_test = np.array([in_test.flatten()])
        
        # Append the audio and paramters to the full data sets 
        all_clean_test = np.append(all_clean_test , np.append(in_test,params_test, axis=0), axis = 1)

        del in_data, clean_data, in_train, in_val

    # Save the wav files 
    save_wav_dont_flatten(args.path + "/test/" + args.name + "-input.wav", all_clean_test.T)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='''This script prepairs the data data to be trained'''
    )
    parser.add_argument("name")
    parser.add_argument("--snapshot", "-s", nargs="+", help="Snapshot configuration. TRAINING_IN TRAINING_OUT OPTIONAL_TEST_IN OPTIONAL_TEST_OUT")
    parser.add_argument("--normalize", "-n", type=bool, default=False)
    parser.add_argument("--parameterize", "-p", type=str, default=None)
    parser.add_argument("--mod_split", '-ms', default=5, help="The default splitting mechanism. Splits the training and validation data on a 5 mod (or 20 percent).")
    parser.add_argument("--random_split", '-rs', type=float, default=None, help="By default, the training is split on a modulus. However, desingnating a percentage between 0 and 100 will create a random data split between the training and validatation sets.")
    parser.add_argument("--path", type=str, default="Data", help="Path to store the processed data.")

    args = parser.parse_args()

    if args.parameterize:
        print("Parameterized Data")
        conditionedWavParse(args)

    else:
        print("Parameterization Data Missing.. Exit..")