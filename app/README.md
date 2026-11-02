---
title: Drishti - Disaster Detection from Drone Images
emoji: 🛰️
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Explainable drone-image disaster recognition (AIDER)
---

# 🛰️ Drishti — explainable disaster detection from drone images

Upload an aerial / drone photo and the model labels it as **collapsed building, fire / smoke, flood, traffic accident or normal**, shows **where** it looked (Grad-CAM), suggests a **response**, and routes **uncertain** frames to a human analyst. The *Area scan* tab tiles a large frame and maps where the emergency is.

* Model: ImageNet-pretrained CNN (MobileNetV3-Large / EfficientNet-B0, best chosen on validation) fine-tuned in PyTorch, calibrated with temperature scaling.
* Data: AIDER — C. Kyrkou & T. Theocharides, *EmergencyNet*, IEEE JSTARS 13 (2020) 1687–1699; Zenodo DOI 10.5281/zenodo.3888300 (CC BY 4.0).
* Built for EC-422 Artificial Intelligence, IIT (BHU) Varanasi. Code and training notebook: see the GitHub repository linked in the project README.

This is a decision-support demo trained on web-collected images; it is not a certified emergency system.
