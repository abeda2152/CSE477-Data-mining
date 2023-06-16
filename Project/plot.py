import numpy as np
import matplotlib.pyplot as plt

accuracy = [78.48588537211292, 51.083547191331625, 67.20844026233249, 66.63815226689478]
name = ['DecisionTreeClassifier','KNeighborsClassifier','RandomForestClassifier','SupportVectorMachine']

y = np.arange(len(name))

plt.barh(y, accuracy, align='center', alpha=0.5)
plt.yticks(y, name)
plt.xlabel('Accuracy')
plt.title('Accuracy of different models using count vectorizer')
 
plt.show()