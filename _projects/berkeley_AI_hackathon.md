---
title: UC Berkeley AI Hackathon
subtitle: Smart glasses that served as an IOT-enabled travel companion & guide for the visually impaired.
shorttitle: UC Berkeley AI Hackathon
image: 
    - assets/images/berk_hack_exploded_view.jpg
    - assets/images/berk_hack_product.jpg
    - assets/images/berk_hack_team.jpg
layout: default
date: 2023-06-01
custom_date: Summer 2023
keywords: blogging, writing
published: true
---

When I first heard about the latest developments in LLMs and ML, I was excited about the potential for LLMs to serve as an intelligent, yet simple-to-use tool that greatly improved human - computer interactibility.

<div class="md-image-container">
    <img class="post-image" src="/assets/images/berk_hack_team.jpg" height=auto width="80%">
</div>

<br>

For the hackathon, my teammate and I innovated and designed a Smart Glasses concept that served as an IOT-enabled travel companion & guide for the visually impaired.

We used many low-cost sensors: a camera, mic, bone-conducting speaker, GPS module, and many more! To tie all the data together, we created a unified format and process that would be provided to the LLM. Users would be able to ask various questions into their mic, such as "Is there a water bottle in front of me" or "what street am I on?"

<div class="md-image-container">
    <img class="post-image" src="/assets/images/berk_hack_diagram1.jpg" height=auto width="100%">
</div>

<br>

We also implemented a pre-trained depth perception model, as well as image processing (BEV Grid map creation) and used the A* Search algorithm to create a path-finding feature for our users.

<div class="md-image-container">
    <img class="post-image" src="/assets/images/berk_hack_diagram2.jpg" height=auto width="100%">
</div>