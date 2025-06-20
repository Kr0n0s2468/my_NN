import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from matplotlib.widgets import Button, Slider
from scipy.ndimage import zoom


#MNIST dataset, with training images of 28*28=784, cada pixel com valor entre 0 e 255 (preto a branco), 10 classes, since 10 digits

#so 2 layers: 1 input layer with 784 units, 1 hiden with 10 and a last one with 10  

#forward propagation: A = entrance value x; Z hiden layer = wight * A + b (bias) 
#Activation function for it to not be linear, use relu(A)
#last layer is the same of the iden but with the relu result
#to finish use a softmax funciton (it is also a activation function) - converts numbers to probability from 0 to 1

#now we need backpropagation to learn: go from prediction backwards, how much it deviated from the label and how much each layer intervined in the error: erro = prediction-correct
#updates parameters with learning rate too , hyperparameter (set by not model)
#loop

##### DATA #####

# Load and shuffle
data = pd.read_csv('./digit-recognizer/train.csv')
data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle with reproducibility

# Split label and features
Y_all = data['label'].to_numpy()
X_all = data.drop('label', axis=1).to_numpy()

# Normalize
X_all = X_all.T / 255.0
Y_all = Y_all

# Split train/dev
X_dev = X_all[:, :1000]
Y_dev = Y_all[:1000]
X_train = X_all[:, 1000:]
Y_train = Y_all[1000:]

# For completeness
_, m_train = X_train.shape
##### NN #####

def init_params(): #initialize parameters
    W1 = np.random.randn(10, 784) * np.sqrt(2. / 784)
    b1 = np.random.rand(10,1) - 0.5
    W2 = np.random.randn(10, 10) * np.sqrt(2. / 10)
    b2 = np.random.rand(10,1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z,0)

def softmax(Z):
    #return np.exp(Z) / np.sum(np.exp(Z))
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z_shifted)
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2 

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max()+1))
    one_hot_Y[np.arange(Y.size),Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m=Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2,0)

def get_accuracy(predictions,Y):
    #print(predictions,Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in tqdm(range(iterations), desc="Training"):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
    
        if i % 10 == 0:
            train_acc = get_accuracy(get_predictions(A2), Y)
            _, _, _, A2_dev = forward_prop(W1, b1, W2, b2, X_dev)
            dev_acc = get_accuracy(get_predictions(A2_dev), Y_dev)
            tqdm.write(f"Iteration {i}: Train Acc = {train_acc:.4f}, Val Acc = {dev_acc:.4f}")

    accuracy = get_accuracy(get_predictions(A2), Y)
    return W1, b1, W2, b2, accuracy
        
#testing_predicitons

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    

def save_model(W1, b1, W2, b2, accuracy):
    np.savez("model_weights.npz", W1=W1, b1=b1, W2=W2, b2=b2, accuracy=accuracy)
    
def load_model():
        data = np.load("model_weights.npz")
        W1_flat, b1_flat, W2_flat, b2_flat, accuracy = data['W1'], data['b1'], data['W2'], data['b2'], data['accuracy'].item()  
        return W1_flat, b1_flat, W2_flat, b2_flat, accuracy
    
def use_saved(X):
    W1_flat, b1_flat, W2_flat, b2_flat, accuracy = load_model()
    
    W1 = W1_flat.reshape((10, 784))
    b1 = b1_flat.reshape((10, 1))
    W2 = W2_flat.reshape((10, 10))
    b2 = b2_flat.reshape((10, 1))
    
    return make_predictions(X, W1, b1, W2, b2), accuracy


def run_model(X_train,Y_train,epochs,lr):
    W1, b1, W2, b2, accuracy = gradient_descent(X_train, Y_train, epochs, lr)
    print("Accuracy:", accuracy)
    save_model(W1, b1, W2, b2, accuracy)
    
def run_saved(X):
    predictions, accuracy = use_saved(X)
    print("Accuracy:", accuracy)
    return print(predictions)
    
#run_model(X_train,Y_train,1000,0.1)
#run_saved(X_dev)

def test_model(W1, b1, W2, b2, test_path='./digit-recognizer/test.csv'):
    
    test_data = pd.read_csv(test_path).to_numpy().T  
    X_test = test_data / 255.0  
 
    _, _, _, A2_test = forward_prop(W1, b1, W2, b2, X_test)
    predictions = get_predictions(A2_test)

    return print(predictions)

#test_model(*load_model()[:-1])


def display_predictions_grid(X_test, predictions, img_size=28, cols=10, max_images=50):
    num_samples = min(X_test.shape[1], max_images)
    rows = math.ceil(num_samples / cols)
    
    plt.figure(figsize=(cols * 2, rows * 2))
    
    for i in range(num_samples):
        img = X_test[:, i].reshape(img_size, img_size)
        pred = predictions[i]
        
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f'Pred: {pred}', fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
def test_model_and_show_grid(W1, b1, W2, b2, test_path='./digit-recognizer/test.csv'):
    test_data = pd.read_csv(test_path).to_numpy()
    X_test = test_data.T / 255.0
    
    _, _, _, A2_test = forward_prop(W1, b1, W2, b2, X_test)
    predictions = get_predictions(A2_test)
    
    display_predictions_grid(X_test, predictions)
    return predictions
#test_model_and_show_grid(*load_model()[:-1])


