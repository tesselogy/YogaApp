# YogaTrainerApp (macOS SwiftUI + Vision + CoreML)

## Steps to build:

1. Export models from Python:
   yolo export model=yolov8n.pt format=coreml
   yolo export model=best.pt format=coreml

2. Place:
   - yolov8n.mlmodel
   - best.mlmodel

   into the Models/ folder in Xcode.

3. Open Xcode → Create new macOS SwiftUI App
4. Replace generated files with this project structure
5. Drag models into project
6. Build & Run

Enjoy your native Metal-accelerated Yoga Trainer 🚀