Welcome!

This repository contains a project to automatically extract multiple attributes from face images.

The attributes are:
- ethnicity (4 classes)
- ethnicity (7 classes)
- gender
- age range (+the scores for each assigned annotation)
- some face boundingbox ('bbox')
- the scores for the emotion 
- orientation ('yaw', 'pitch', 'roll')

Having additional information about the face images in a dataset can be useful to carry out or improve the development of machine learning models by targeting specific groups of people for which a lower performance is observed. This is particularly relevant for those datasets in which little / no metadata about face images is available.

This project combines three different facial attribute extractors:
- [FairFace](https://github.com/dchen236/FairFace): Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation. In Proceedings of 2021 the IEEE/CVF Winter Conference on Applications of Computer Vision.
- [DMUE](https://github.com/JDAI-CV/FaceX-Zoo/tree/main/addition_module/DMUE):  Dive into Ambiguity: Latent Distribution Mining and Pairwise Uncertainty Estimation for Facial Expression Recognition in CVPR2021. 
- [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2): Pose estimation (and much more) from ECCV2020.

This code is not too pretty nor optimized for big batches of images at the moment but it should work ok and it can be easily improved. 

### Getting started

First, it is necessary to set your environment. After having installed docker, the environment can be set up with the following instruction. 

```
cd FaceAnnotate
docker build -t face_annotate_no_cuda_ctr -f docker/Dockerfile .
docker run -i -t face_annotate_no_cuda_ctr
```
Otherwise, if you are familiar with conda, you can create your own conda environment with requirements.txt + `dlib` (specified in docker/Dockerfile).

The original code was written to run on GPU only, but it was modified to make predictions on CPU only, it should be okay as we are not aiming to train the models but just to make the inference.

### Run 

To configure the FairFace module, follow the instructions:

- Download the pretrained models from [here](https://drive.google.com/drive/folders/1F_pXfbzWvG-bhCpNsRj6F_xsdjpesiFu?usp=sharing) and save it in the same folder as where predict.py is located. Two models are included, race_4 model predicts race as White, Black, Asian and Indian and race_7 model predicts races as White, Black, Latino_Hispanic, East, Southeast Asian, Indian, Middle Eastern.

To configure the DMUE module, follow the instructions:

- Download the trained model in this [link](https://drive.google.com/drive/folders/1p_vRIClF5ZXdDVzQC0oYnffspA5TjqnU), and put it to `DMUE/checkpoints/`.
- Run convert_weights.py to convert the multi-branches weights to the target-branch weights.

The 3DDFA_V2 module should be configured automatically when creating the docker image. 

To create the annotations, please include your face images in the `test/` directory. Then, run:

`python label_image.py`

The produced output should be in the form of a JSON file as follows:

```
{
    "giuseppe.jpg": {
        "face_name_align": "detected_faces/giuseppe_face0.jpg",
        "race": "White",
        "race4": "White",
        "gender": "Male",
        "age": "30-39",
        "race_scores_fair": "[9.85710621e-01 1.30783656e-05 1.01303426e-03 1.51345876e-05\n 3.57824974e-05 3.26548980e-05 1.31796915e-02]",
        "race_scores_fair_4": "[9.9974412e-01 3.9166869e-05 9.5054493e-06 2.0719091e-04]",
        "gender_scores_fair": "[9.9991304e-01 8.7006061e-05]",
        "age_scores_fair": "[5.1857921e-07 7.0258275e-06 1.2442061e-03 4.3200600e-01 5.3913969e-01\n 2.7463835e-02 1.3595227e-04 2.5744030e-06 8.4393015e-08]",
        "bbox": "[(84, 50) (178, 144)]",
        "Neutral": 0.4619,
        "Happy": 0.0118,
        "Sad": 0.32,
        "Surprise": 0.0251,
        "Fear": 0.0003,
        "Disgust": 0.0911,
        "Anger": 0.0854,
        "Contempt": 0.0043,
        "yaw": -25.45,
        "pitch": 1.01,
        "roll": -3.89
    }
}
```

The JSON file will be saved as `results/results.json`. 

Please refer to the original projects for any additional information.