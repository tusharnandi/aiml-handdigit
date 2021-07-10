import sys
import cv2
import numpy as np
from datetime import datetime
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json

class DelectBWNumber():
	
    def __init__(self):

        #Load Model and load Model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        
        self.loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        self.loaded_model.load_weights("model.h5")

        self.loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  
    def record(self, imageFile):
        
        self.imagefile = imageFile
        
        #print ("Image file: " + self.imagefile)
        #shape1: (28, 28)
        #shape2: (28, 28)
        #shape3: (28, 28, 1)
        #shape4: (1, 28, 28, 1)
        
        #print ("Starting: record function")
        img_array = cv2.imread(self.imagefile, 0)
        
        #print("shape-1:",img_array.shape)
        
        img_array = img_array.astype("float32") / 255
        #print("shape-2:",img_array.shape)   0-255
        
        img_array = np.expand_dims(img_array, -1)
        #print("shape-3:",img_array.shape)
        
        img_array = img_array.reshape(1, 28, 28, 1)
        
        #print("shape-4:",img_array.shape)
        #print(img_array)
        
        pred = self.loaded_model.predict(img_array)
        pnum = pred.argmax();
        return pnum;
############################################
#DelectBWNumber END


class DelectDogCat():
	
    def __init__(self):

        #Load Model and load Model
        json_file = open('./dogcatsModel/model_vgg16.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        
        self.loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        self.loaded_model.load_weights("./dogcatsModel/model_vgg16_weight01.h5")

        opt = Adam(learning_rate=0.001)
        #//model.compile(optimizer=opt, loss=keras.losses.binary_crossentropy, metrics=["accuracy"])

        self.loaded_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  
    def record(self, imageFile):
        
        self.imagefile = imageFile
        
        #load the image you want to classify
        image = cv2.imread(self.imagefile)
        image = cv2.resize(image, (224,224))
        
        #////cv2_imshow(image)
        
        #predict the image
        preds = self.loaded_model.predict(np.expand_dims(image, axis=0))
        #if preds==0:
        #    pnum="Cat"
        #    print("Predicted Label:Cat")
        #else:
        #    pnum="Dog"
        #    print("Predicted Label: Dog")

        
        #pred = self.loaded_model.predict(img_array)
        pnum = preds.argmax();
        print("Predicted Label:", pnum)
        return pnum;
############################################
#DelectBWNumber END
#================================
from flask import Flask
import os
from flask import render_template, flash, request, redirect, url_for

app = Flask(__name__)
app.config['UPLOAD_FOLDER']= "./queueImage/"


@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact',
        year=datetime.now().year,
        message='Your contact page.'
    )

@app.route('/upload')
def upload():
    """Renders the contact page."""
    return render_template(
        'upload.html',
        title='upload',
        year=datetime.now().year,
        message='Your contact page.'
    )

@app.route('/dogcat')
def dogcat():
    """Renders the contact page."""
    return render_template(
        'dogcat.html',
        title='dogcat',
        year=datetime.now().year,
        message='Your contact page.'
    )

@app.route('/dogcatdetect', methods = ['POST'])  
def dogcatdetect():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        imagePath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        
        detectObj = DelectDogCat()
        predictionObj= detectObj.record(imagePath)

        return render_template("successdogcat.html", name = f.filename,prediction=predictionObj)  



@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']  


        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))

        imagePath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        
        detectObj = DelectBWNumber()
        pnumber= detectObj.record(imagePath)

        return render_template("success.html", name = f.filename,predictedNumber=pnumber)  


@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )


#===============================
if __name__== "__main__":
    app.run(debug=True)