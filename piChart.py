import matplotlib.pyplot as plt

dict_emotions = {'Angry':0,'Disgust':0,'Fear':0,'Happy':0,'Neutral':0, 'Sad':0, 'Surprise':0}

# for emotion in dict_emotions:

for line in open('suhali_file.txt', 'r'):
    dict_emotions[line[:len(line) - 1]] = dict_emotions[line[:len(line)-1]] + 1

print(dict_emotions)

labels = []
sizes = []

for x, y in dict_emotions.items():
    if(y !=0):
        labels.append(x)
        sizes.append(y)

# Plot
plt.pie(sizes, labels=labels)
plt.axis('equal')
plt.savefig('emotionAnalysis.png')
plt.show()