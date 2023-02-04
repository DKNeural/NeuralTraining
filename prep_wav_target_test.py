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

def sliceOnMod(target_data, mod):
    # Split the data on a modulus.

    # type cast to an integer the modulus
    mod = int(mod)

    # Split the data into 100 pieces
    target_split = np.array_split(target_data, 100)

    val_target_data = []
    # Traverse the range of the indexes of the input signal reversed and pop every 5th for val
    for i in reversed(range(len(target_split))):
        if i%mod == 0:
            # Store the validation data
            val_target_data.append(target_split[i])
            # Remove the validation data from training
            target_split.pop(i)

    # Flatten val_data down to one dimension and concatenate
    val_target_data = np.concatenate(val_target_data)

    # Concatinate b back together
    training_target_data = np.concatenate(target_split)
    return (training_target_data, val_target_data)

def conditionedWavParse(args):
    '''
    Note: Assumes all .wav files are mono, float32, no metadata
    '''
    # Open the configureation
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
    all_target_test = np.array([[]]) # 1 channels of all (out audio)


    for ds in data["Data Sets"]:

        # Load and Preprocess Data
        out_rate, out_data = wavfile.read(ds["TrainingTarget"])
        
        if out_rate != 44100:
            print("\n\n\n[ERROR] The out*.wav file has an invalid samplerate " +"("+ str(out_rate) +")")
            print("[ERROR] Please re-export your wav file as 44100 samplerate (44.1kHz).\n\n\n")
            return
    
        if out_data.dtype != "int16" and out_data.dtype != "float32":
            print("\n\n\n[ERROR] The out*.wav file has an invalid data type " +"("+ str(out_data.dtype) +")")
            print("[ERROR] Please re-export your wav file as either PCM16 or FP32 data type (bit depth).\n\n\n")
            return        

        #If stereo data, use channel 0
        if len(out_data.shape) > 1:
            print("[WARNING] Stereo data detected for out_data, only using first channel (left channel)")
            out_data = out_data[:,0]

        # Convert PCM16 to FP32
        if out_data.dtype == "int16":
            out_data = out_data/32767
            print("Out data converted from PCM16 to FP32")    
        
        target_data = out_data.astype(np.float32).flatten()

        # If Desired Normalize the data
        if (args.normalize):
            target_data = normalize(target_data).reshape(len(target_data),1)

        # Make the Training and validation split
        # Split the data on a twenty percent mod
        out_train, out_val = sliceOnMod(target_data, args.mod_split)


        # Process the Test Data. If it's a separate set, process that set. 
        # If there is not a separeate set, use the validation data for testing.
        if (seprateTestSet):
            test_out_rate, test_out_data = wavfile.read(ds["TestTarget"])
            out_test = test_out_data.astype(np.float32).flatten()

            if (args.normalize):
                out_test = normalize(out_test).reshape(len(test_out_data),1)
        else:
            out_test = out_val
            if (args.normalize):
                out_test = normalize(out_test).reshape(len(out_test),1)
        
        # Append the audio and paramters to the full data sets 
        all_target_test = np.append(all_target_test, out_test)

        del out_data, target_data, out_train, out_val

    # Save the wav files 
    save_wav(args.path + "/test/" + args.name + "-target.wav", all_target_test)


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