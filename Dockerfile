# Select base image
FROM python:3.11-slim

# Create user name and home directory variables. 
# The variables are later used as $USER and $HOME. 
ENV USER=username
ENV HOME=/home/$USER

# Add user to system
RUN useradd -m -u 1000 $USER

# Set working directory (this is where the code should go)
WORKDIR $HOME/app

# Update system and install dependencies.
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    software-properties-common

# Copy requirements.txt and install packages listed there with pip (this will place the files in home/username/)
COPY requirements.txt $HOME/app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install torch, torchvision, torchaudio packages with pip separately from the packages installed through the requirementx.txt
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy all files that are needed for your app to run with the directory structure as your files expect
COPY main.py $HOME/app/main.py
COPY start-script.sh $HOME/app/start-script.sh
COPY assets/example_images/ $HOME/app/assets/example_images/
COPY assets/flower_dataset_labels.txt $HOME/app/assets/flower_dataset_labels.txt
# Because the model file is too large we download it separately and put in the correct location
ADD https://nextcloud.dc.scilifelab.se/s/GSf2g5CAFxBPtMN/download $HOME/app/assets/flower_model_vgg19.pth

# Give access to appripriate files and folders to the created user
RUN chmod +x start-script.sh \
    && chown -R $USER:$USER $HOME \
    && rm -rf /var/lib/apt/lists/*

USER $USER
EXPOSE 8080

ENTRYPOINT ["./start-script.sh"]