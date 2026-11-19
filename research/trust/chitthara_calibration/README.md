Lucid – Temperature Calibration



1. What I implemented



- Integrated temperature scaling into the backend inference service.

- Added checkpoint SHA256 validation to ensure the calibration file matches the model.

- Updated prediction probabilities using temperature scaling.

- Added tests to verify calibrated probabilities.

- Fixed the frontend to display calibration metadata correctly.

- Verified that the application displays the calibration method, temperature and checkpoint hash.



2. Setup and Run Commands



Open Command Prompt and navigate to the project folder:



cd C:\\Users\\Chitthara\\Lucid



Run the complete test suite:



python -m pytest



Observed result:42 tests passed.



Run the calibration demo:



python -m research.trust.chitthara\_calibration.run --demo

Observed result:Demo completed successfully.



3. Model and Dataset Details



- Model architecture: ResNet18

- Classes: NORMAL and PNEUMONIA

- Image size: 224 × 224

- Preprocessing version: 1.0

- Dataset: Chest X-ray dataset

- Split file: `data/processed/splits.json`



4. Checkpoint and Split Hashes



Classifier checkpoint SHA256:



`040dca03ae91682de538f824339264162afbc741423265b6452a66081997a510`



Split file SHA256:



`196dbf0407a6a3f6b61921ff5ffe25cfa4b4b2c3577cf0111ac0cf92d1fb3f49`



5. Calibration Details



- Calibration method: Temperature Scaling

- Temperature (T): 1.074348705760295

- Calibration file: `trust\_artifacts/temperature.json`

- Calibration was fitted using the Kermany validation dataset.

- The calibration checkpoint hash matches the classifier checkpoint hash.



6. Test Results



- Backend and frontend integration completed.

- All 42 tests passed.

- Calibrated probabilities were independently verified.

- Checkpoint SHA256 validation was successful.

- Frontend calibration metadata display was verified.

- Calibration demo ran successfully.



7. Demo Results



The demo showed improvements in:



- Expected Calibration Error (ECE)

- Brier Score

- Negative Log-Likelihood (NLL)



Accuracy remained unchanged, and error-detection AUC values remained unchanged.



8. Limitations and Pending Work



- External-data evaluation is pending until the required NIH inputs and exports are available.

- Calibration artifacts have been shared with Nikhitha. Final confirmation and any additional handoff requirements are pending.

- Further validation is required before using the model outside research evaluation.





