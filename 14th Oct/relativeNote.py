import scipy
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import librosa
from collections import Counter
import pretty_midi

#need a filter

Time_length=40.0

#need a relative pitch:
def filtRest(data):
    data[data<5e-3]=0
    return data

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
    #暂时去掉四舍五入
    #f0=convert_freq2midi(f0)
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
    f0[f0>=90]=0
    #times = librosa.times_like(f0,hop_length=2)
    #plt.plot(times, f0, "_", linewidth=1)
    #plt.xlabel("Time(s)")
    #plt.ylabel("F0")
    #plt.show()
    return f0


def cal_error(seq_a,seq_b):
    if len(seq_a)>len(seq_b):
        while not len(seq_a)==len(seq_b):
            seq_b=np.append(seq_b,0)
    elif len(seq_b)>len(seq_a):
        return cal_error(seq_b,seq_a)
    return seq_a-seq_b

def get_midi_inf(notes,hop_length=512):
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
        #这里原本加了time_length的限制条件
        if item.end<=notes[len(notes)-1].end:#Time_length:
            while i*hop_length/44100<item.start:
                i=i+1
            while i*hop_length/44100<=item.end and i<len(sequence):
                sequence[i]=item.pitch
                if not item in total:
                    total.append(item)
                i=i+1
        else:
            break
    print('total')
    print(len(total))
    return sequence

def dtw_distance(ts_a, ts_b, d=lambda x,y: abs(x-y), mww=10000):
    """Computes dtw distance between two time series
    
    Args:
        ts_a: time series a
        ts_b: time series b
        d: distance function
        mww: max warping window, int, optional (default = infinity)
        
    Returns:
        dtw distance
    """
    
    # Create cost matrix via broadcasting with large int
    ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    M, N = len(ts_a), len(ts_b)
    cost = np.ones((M, N))

    # Initialize the first row and column
    cost[0, 0] = d(ts_a[0], ts_b[0])
    for i in range(1, M):
        cost[i, 0] = cost[i-1, 0] + d(ts_a[i], ts_b[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j-1] + d(ts_a[0], ts_b[j])

    # Populate rest of cost matrix within window
    for i in range(1, M):
        for j in range(max(1, i - mww), min(N, i + mww)):
            choices = cost[i-1, j-1], cost[i, j-1], cost[i-1, j]
            cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

    # Return DTW distance given window 
    return cost[-1, -1]

def relative_pitch(inputSeq,fs=44100):
    onset_index=0
    for i in range(len(inputSeq)):
        if (not inputSeq[i]==0) and np.all(inputSeq[i:i+23]==inputSeq[i]):
            onset_index=i
            break
    inputSeq=inputSeq[onset_index:len(inputSeq)]
    inputSeq=inputSeq-np.ones(len(inputSeq))*inputSeq[0]
    return inputSeq

def Levenshtein_Distance(str1, str2):
    """
    计算字符串 str1 和 str2 的编辑距离
    :param str1
    :param str2
    :return:
    """
    matrix = [[ i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
 
    for i in range(1, len(str1)+1):
        for j in range(1, len(str2)+1):
            if(str1[i-1] == str2[j-1]):
                d = 0
            else:
                d = 1
            
            matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1, matrix[i-1][j-1]+d)
 
    return matrix[len(str1)][len(str2)]

                

#constant=np.zeros(len(find_relativeNote("01-D_AMairena.wav")))
#relative=find_relativeNote("01-D_AMairena.wav")+constant
#for i in range(len(constant)):
#    relative[i]=int(relative[i]+0.5)
#find_f0("01-D_AMairena.wav")
#wav_data=find_f0("一千年以后_TEST_VOCAL.wav")
#a=get_note("一千年以后_TEST_VOCAL.wav")
#b= get_midi_inf(file_name='一千年以后_TEST_PIANO.mid',hop_length=1024)
'''
a=get_note("CSD\english\wav\en001a.wav")
a=relative_pitch(a)
distance=np.zeros(10)
for w in range(1,10):
    midi_data=pretty_midi.PrettyMIDI('CSD\english\mid\en00'+str(w)+'a.mid')
    notes=midi_data.instruments[0].notes
    dismatri=np.array([])
    for i in range(len(notes)):
        end_index=0
        for j in range(i,len(notes)):
            if notes[j].end-notes[i].start>=Time_length:
                end_index=j
                print(i)
                print(j)
                break
        if not end_index==0:
            b=get_midi_inf(notes[i:end_index],hop_length=1024)
            b=relative_pitch(b)
            dismatri=np.append(dismatri,Levenshtein_Distance(a,b))
            print(dismatri)
    if not dismatri.size==0:
        distance[w]=min(dismatri)
    else:
        distance[w]=np.inf
distance=np.delete(distance,0)
index=np.argmin(distance)
print('en00'+str(index+1)+'a')
'''
Error_all=0
for w in range(1,51):
    if w<10:
        a=get_note('CSD\english\wav\en00'+str(w)+'a.wav')
        midi_data=pretty_midi.PrettyMIDI('CSD\english\mid\en00'+str(w)+'a.mid')
    elif w>=10:
        a=get_note('CSD\english\wav\en0'+str(w)+'a.wav')
        midi_data=pretty_midi.PrettyMIDI('CSD\english\mid\en0'+str(w)+'a.mid')

    notes=midi_data.instruments[0].notes
    b= get_midi_inf(notes,hop_length=1024)
    error=cal_error(a,b+np.ones(len(b))*12)
    error[error<5]=0
    Error_all+=len(error[np.nonzero(error)])
    print(len(error))
    print(Error_all)
#inputto=error
#times = librosa.times_like(inputto,hop_length=2)
#plt.plot(times, inputto, "_", linewidth=1)
#plt.xlabel("Time(s)")
#plt.ylabel("Error")
#plt.show()
print(Error_all/50)

