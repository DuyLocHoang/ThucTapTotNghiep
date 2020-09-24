import joblib
from sklearn.svm import LinearSVC
from utilities.hog import HOG
from utilities import dataset
from sklearn import neighbors
import  time
from sklearn.metrics import accuracy_score
# Define paths
dataset_path_train = 'data/mnist_train.csv'
dataset_path_test = "data/mnist_test.csv"
model_path = 'models/knnK1.npy'
start_time = time.time()
# Load the dataset and initialize the data matrix
(digits, target) = dataset.load_digits(dataset_path_train)
(digits_test, target_test) = dataset.load_digits(dataset_path_train)
data = []

# Initialize the HOG descriptor
hog = HOG(orientations=18, pixels_per_cell=(10, 10), cells_per_block=(1, 1), transform=True)

# Loop over the images
for image in digits:
	# De-skew the image and center it
	image = dataset.de_skew(image, 20)
	image = dataset.center_extent(image, (20, 20))

	# Describe the image and update the data matrix
	hist = hog.describe(image)
	data.append(hist)
X_test = []
for image in digits_test:
	# De-skew the image and center it
	image = dataset.de_skew(image, 20)
	image = dataset.center_extent(image, (20, 20))

	# Describe the image and update the data matrix
	hist = hog.describe(image)
	X_test.append(hist)
# Train the model
# model = LinearSVC(random_state=42)
# model.fit(data, target)
clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
clf.fit(data,target)
y_pred= clf.predict(X_test)

# Save the model to file
end_time = time.time()
print ("Accuracy of KNN for MNIST: %.2f %%" %(100*accuracy_score(target_test, y_pred)))
print("Running time: %.2f (s)" % (end_time - start_time))
joblib.dump(clf, model_path)
