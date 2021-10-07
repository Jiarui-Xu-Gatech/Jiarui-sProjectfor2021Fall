import scipy
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import librosa
from collections import Counter
import pretty_midi

#need a filter

Time_length=30.0

#need a relative pitch:
def filtRest(data):
    data[data<5e-3]=0
    return data


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


def Bfilter(data,wn=20,hn=20000):
    b, a = signal.butter(8, [0.02,0.17], 'bandpass')   #配置滤波器 8 表示滤波器的阶数
    filtedData = signal.filtfilt(b, a, data)  #data为要过滤的信号
    print(filtedData)
    return filtedData

def find_f0(filename=' '):
    filename = filename
    y, sr = librosa.load(filename,duration=Time_length)
    #print(len(y))
    y=Bfilter(y,wn=20)
    y=filtRest(y)
    #print(len(y))
    #f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),sr=44100)
    f0 = librosa.yin(y,fmin=20,fmax=8000,sr=44100)
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
    y, sr = librosa.load(file_name,duration=Time_length)
    #y=Bfilter(y,wn=20)
    y=filtRest(y)
    #help(librosa.onset.onset_detect)
    return librosa.onset.onset_detect(y,hop_length=512)
    


def get_note(file_name='01-D_AMairena.wav'):
    on_set=find_onset(file_name)
    f0=find_f0(file_name)
    #print(len(f0))
    for i in range(len(on_set)-1):
        #pitch_number=np.argmax(np.bincount(f0[on_set[i]:on_set[i+1]]))
        pitch_number=Counter(f0[on_set[i]:on_set[i+1]]).most_common(1)[0][0]
        divide_number=1
        f0[on_set[i]:on_set[i]+int((on_set[i+1]-on_set[i])/divide_number)]=np.ones(int((on_set[i+1]-on_set[i])/divide_number))*pitch_number
        f0[on_set[i]+int((on_set[i+1]-on_set[i])/divide_number):on_set[i+1]]=np.zeros(on_set[i+1]-on_set[i]-int((on_set[i+1]-on_set[i])/divide_number))
    times = librosa.times_like(f0,hop_length=2)
    plt.plot(times, f0, "_", linewidth=1)
    plt.xlabel("Time(s)")
    plt.ylabel("F0")
    plt.show()
    return f0


def cal_error(seq_a,seq_b):
    if len(seq_a)>len(seq_b):
        while not len(seq_a)==len(seq_b):
            seq_b=np.append(seq_b,0)
    elif len(seq_b)>len(seq_a):
        return cal_error(seq_b,seq_a)
    return seq_a-seq_b

def get_midi_inf(file_name='一千年以后_TEST_PIANO.mid',hop_length=512):
    midi_data=pretty_midi.PrettyMIDI(file_name)
    notes=midi_data.instruments[0].notes
    #print(notes[len(notes)-1].end)
    sequence=np.zeros(min((int(notes[len(notes)-1].end*44100/(hop_length))+1),int(Time_length*44100/hop_length)))
    #for i in range(len(sequence)):
    #    for item in notes:
    #        if item.start<=int(i*hop_length/44100) and item.end>=int(i*hop_length/44100):
    #            sequence[i]=item.pitch
    i=0
    #print(len(notes))
    total=[]
    for item in notes:
        if item.end<=Time_length:
            while i*hop_length/44100<item.start:
                i=i+1
            while i*hop_length/44100<=item.end:
                sequence[i]=item.pitch
                if not item in total:
                    total.append(item)
                i=i+1
        else:
            break
    print('total')
    print(len(total))
    return sequence


                

#constant=np.zeros(len(find_relativeNote("01-D_AMairena.wav")))
#relative=find_relativeNote("01-D_AMairena.wav")+constant
#for i in range(len(constant)):
#    relative[i]=int(relative[i]+0.5)
#find_f0("01-D_AMairena.wav")
#wav_data=find_f0("一千年以后_TEST_VOCAL.wav")
a=get_note("CSD\english\wav\en001a.wav")
b=get_midi_inf(file_name='CSD\english\mid\en001a.mid',hop_length=1024)
#a=get_note("一千年以后_TEST_VOCAL.wav")
#b= get_midi_inf(file_name='一千年以后_TEST_PIANO.mid',hop_length=1024)
error=cal_error(a,b)
inputto=error
times = librosa.times_like(inputto,hop_length=2)
plt.plot(times, inputto, "_", linewidth=1)
plt.xlabel("Time(s)")
plt.ylabel("Error")
plt.show()

