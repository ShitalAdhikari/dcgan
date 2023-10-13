1. Clone the repo using following command:
    ```bash
    git clone https://github.com/ShitalAdhikari/dcgan.git
    cd dcgan
    ```

2. Create a virtual environment and install the required dependencies
    ```python
    python3 -m venv dcgan_venv
    source dcgan_venv/bin/activate
    pip install -r requirements.txt
    ```
3. Run the file main.py using command line. It performs the following operation: 
   1. Downloads the MNIST dataset and stores in the data folder.
   2. Train the generator and descriminator and stores the image in `data` folder.
    ```python
    python3 main.py
    ```

Reference:
1. [UNSUPERVISED REPRESENTATION LEARNING
WITH DEEP CONVOLUTIONAL
GENERATIVE ADVERSARIAL NETWORKS](https://arxiv.org/pdf/1511.06434.pdf)
