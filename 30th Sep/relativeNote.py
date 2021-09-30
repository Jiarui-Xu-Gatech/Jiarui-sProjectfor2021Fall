import scipy
import numpy as np
import matplotlib.pyplot as plt
import librosa
from collections import Counter

#need a filter

#need a relative pitch:
def relative_pitch(inputNotes):
    result=np.zeros(len(inputNotes))
    for i in range(len(inputNotes)-1):
        result[i]=inputNotes[i]-inputNotes[i+1]
    result[len(inputNotes)-1]=0
    return result

def convert_freq2midi(freqInHz):
    if (not type(freqInHz)==np.ndarray) and freqInHz>0:
        return 69.0+12.0*(np.log(freqInHz/440)/np.log(2))
    elif not type(freqInHz)==np.ndarray:
        return 0
    else:
        result=np.zeros(len(freqInHz))
        for i in range(len(freqInHz)):
            if not freqInHz[i]==0:
                result[i]= 69.0+12.0*(np.log(freqInHz[i]/440)/np.log(2))#we assume that A4 is 69
            else:
                result[i]=0
        return result


def find_f0(filename=' '):
    filename = filename
    y, sr = librosa.load(filename, sr=None)
    f0 = librosa.yin(y, fmin=80, fmax=400,hop_length=512)
    f0[np.isnan(f0)] = 0
    f0=np.trunc(convert_freq2midi(f0)+np.ones(len(f0))*0.5)
    #print(f0[0:400])
    #times = librosa.times_like(f0)
    #plt.plot(times, f0, "_", linewidth=1)
    #plt.xlabel("Time(s)")
    #plt.ylabel("F0")
    #plt.show()
    return f0

def find_relativeNote(filename=' '):
    f0=find_f0(filename)
    midiNote=convert_freq2midi(f0)
    result=relative_pitch(midiNote)
    return result


def find_onset(file_name='01-D_AMairena.wav'):
    y, sr = librosa.load(file_name)
    help(librosa.onset.onset_detect)
    return librosa.onset.onset_detect(y,hop_length=256)
    


def get_note(file_name='01-D_AMairena.wav'):
    on_set=find_onset(file_name)
    f0=find_f0(file_name)
    for i in range(len(on_set)-1):
        #pitch_number=np.argmax(np.bincount(f0[on_set[i]:on_set[i+1]]))
        pitch_number=Counter(f0[on_set[i]:on_set[i+1]]).most_common(1)[0][0]
        divide_number=2
        f0[on_set[i]:on_set[i]+int((on_set[i+1]-on_set[i])/divide_number)]=np.ones(int((on_set[i+1]-on_set[i])/divide_number))*pitch_number
        f0[on_set[i]+int((on_set[i+1]-on_set[i])/divide_number):on_set[i+1]]=np.zeros(on_set[i+1]-on_set[i]-int((on_set[i+1]-on_set[i])/divide_number))
    times = librosa.times_like(f0)
    plt.plot(times, f0, "_", linewidth=1)
    plt.xlabel("Time(s)")
    plt.ylabel("F0")
    plt.show()
#constant=np.zeros(len(find_relativeNote("01-D_AMairena.wav")))
#relative=find_relativeNote("01-D_AMairena.wav")+constant
#for i in range(len(constant)):
#    relative[i]=int(relative[i]+0.5)
#find_f0("01-D_AMairena.wav")
get_note("01-D_AMairena.wav")


