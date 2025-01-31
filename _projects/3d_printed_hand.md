---
title: 3-D Printed Gesture Tracking Robot Hand
subtitle: Robot hand that replicates a user's hand gestures using computer vision.
shorttitle: Gesture Tracking Robot Hand
image: 
    - assets/images/robo_hand.jpeg
    - assets/images/robo_hand_track.png
layout: default
date: 2021-01-01
custom_date: Summer 2021
keywords: blogging, writing
published: true
---

When I first entered college, I was introduced to a senior who had snuck a 3D printer into his dormitory room. We hit it off and he taught me lots about 3D Printing, and before long, I was dreaming up ideas about what I could make. Of course, i chose to make a cool robot hand!

<div class="md-image-container">
    <img class="post-image" src="/assets/images/robo_hand_diagram1.jpg" height=auto width="100%">
</div>

The robot fingers move with the help of four (really strong!) servo motors, all controlled with an Arduino Uno. The joints are made of flexible TPU while the shell is made of PLA.


After assembling the hand and testing the movement with basic arduino code, I decided to upgrade the robot hand with computer vision capabilities, and make it replicate my hand gestures! This was done with OpenCV and CVZone.

<div class="md-image-container">
    <img class="post-image" src="/assets/images/robo_hand_diagram2.jpg" height=auto width="100%">
</div>