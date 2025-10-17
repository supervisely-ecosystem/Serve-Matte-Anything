<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/Serve-Matte-Anything/releases/download/v0.0.1/poster.png"/>  

# Serve Matte Anything

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#example-apply-matte-anything-to-image-in-labeling-tool">Example: apply Matte-Anything to image in labeling tool</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/Serve-Matte-Anything)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/Serve-Matte-Anything)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/serve-matte-anything/serving_app.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/serve-matte-anything/serving_app.png)](https://supervisely.com)

</div>

# Overview

Matte Anything, a combination of several machine learning models proposed by researchers from Huazhong University of Science and Technology, unites segmentation and matting models into a single pipeline to produce accurate alpha channels for images with high quality and simple user interaction.

Matte Anything firstly employs Segment Anything Model to generate a segmentation mask for target object on input image. Subsequently, open-vocabulary object detection model (GroudingDINO) is used to detect commonly occurring transparent objects. Trimaps are then generated based on the segmentation and transparent object detection results, which are subsequently passed into image matting model - ViTMatte (we also added opportunity to use [DiffMatte](https://arxiv.org/pdf/2312.05915)).

![matte anything](https://github.com/supervisely-ecosystem/Serve-Matte-Anything/releases/download/v0.0.1/matte-anything.png)

# How To Run

**Step 1.** Select pretrained model architecture and press the **Serve** button

![pretrained_models](https://github.com/supervisely-ecosystem/Serve-Matte-Anything/releases/download/v0.0.1/pretrained_models.png)

**Step 2.** Wait for the model to deploy

![deployed](https://github.com/supervisely-ecosystem/Serve-Matte-Anything/releases/download/v0.0.1/deployed.png)

# Example: apply Matte-Anything to image in labeling tool

Open your images project, open project settings, go to Visuals, select image matting as project labeling interface:

<video width="100%" preload="auto" autoplay muted loop>
    <source src="https://github.com/supervisely-ecosystem/Serve-Matte-Anything/releases/download/v0.0.1/matting_project_settings.mp4" type="video/mp4">
</video>

Select smart tool, create annotation class of shape 'Alpha mask' and draw a bounding box around target object on image (you can also adjust mask by adding positive and negative points if necessary):

<video width="100%" preload="auto" autoplay muted loop>
    <source src="https://github.com/supervisely-ecosystem/Serve-Matte-Anything/releases/download/v0.0.1/matting_example.mp4" type="video/mp4">
</video>

# Acknowledgment

This app is based on the great works [Matte Anything](https://github.com/hustvl/Matte-Anything) and [DiffMatte](https://github.com/YihanHu-2022/DiffMatte).

