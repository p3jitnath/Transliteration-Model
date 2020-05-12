# Transliteration Model
We have developed a transliteration model @ `Xelpmoc` which transliterates বাংলা to English, with an accuracy of approximately `96.5 %`.

## Important Files
- `transliteration_model.ipynb` - This the Jupyter notebook file which was used to train the transliteration model. The neural network was trained on a Google Cloud Platform instance having 2vCPUs and a single NVIDIA Tesla T4 GPU, taking around 5 minutes for a single epoch.   

- `transliteration_model_usage.ipynb` - This the Jupyter notebook file which was used to develop the API `use_model.py`. It includes all the necessary preprocessing required to feed the বাংলা word to the API, so as to get the necessary output.

- `transliteration_model.h5` - This is the trained Keras Sequential Model which does all the prediction for us.  

- `use_model.py` - This is the API which we have developed, that takes the input through a command line argument, and gives the output in the file `output.txt`.   

- `data/` - This folder contains the dataset on which the neural network was trained.

## Additional Feature(s)
- `machine_translation.ipynb` - This Jupyter Notebook contains 5 models which developed a translation model which could convert English to French. The hybrid model is the one on which we based our final transliteration model. [Credit : Tommy Tracey]

## Note
- Version of TensorFlow used : `1.11.0`
- Version of Keras used : `2.2.4`

**Please do not remove any file while tampering with this repo.**

### Usage :
Correct Usage of API : `python3 use_model.py -word <bengali_word>`  
For example : `python3 use_model.py -word সরকার`
