This repository contains code to automatically extract multiple attributes from face images.

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
    "synth_image.jpg": {
        "face_name_align": "detected_faces/synth_image_face0.jpg",
        "race": "White",
        "race4": "White",
        "gender": "Female",
        "age": "40-49",
        "race_scores_fair": "[9.9889070e-01 3.1716743e-07 8.9687738e-04 2.8304245e-07 4.3613665e-07\n 1.0309393e-05 2.0103849e-04]",
        "race_scores_fair_4": "[9.9990189e-01 1.3173075e-06 7.3273827e-06 8.9435052e-05]",
        "gender_scores_fair": "[3.0999051e-06 9.9999696e-01]",
        "age_scores_fair": "[8.8756700e-08 1.7967748e-06 8.5989202e-05 4.4218916e-03 9.3341194e-02\n 5.6833559e-01 3.2458252e-01 9.1349808e-03 9.5980373e-05]",
        "bbox": "[(202, 238) (743, 778)]",
        "Neutral": 0.0,
        "Happy": 0.9903,
        "Sad": 0.0,
        "Surprise": 0.0,
        "Fear": 0.0,
        "Disgust": 0.0,
        "Anger": 0.0,
        "Contempt": 0.0096,
        "yaw": 2.05,
        "pitch": 5.94,
        "roll": -1.2
    }
}
```

The JSON file will be saved as `results/results.json`. 

Please refer to the original projects for any additional information and licenses.
