# SentinelX

A fall-detection and restricted-area alarm. A laptop webcam runs **YOLO11n-pose** to find people and their body keypoints. A **trained classifier** decides whether each person has fallen, and a **virtual restricted zone** that you draw on the video checks whether anyone has stepped inside it. An ESP32 drives a buzzer and a red LED, and a DHT11 sensor on the ESP32 adds a high-temperature alarm.

A person simply standing in view no longer sets off the buzzer.

| Event | Buzzer | Red LED | Priority |
|---|---|---|---|
| Person fall detected | Fast urgent beeping | Fast blink | 1 (highest) |
| Temperature ≥ 35 °C | Two-tone siren | Slow blink | 2 |
| Person enters restricted area | Steady buzz | Solid on | 3 |
| Nothing | Off | Off | – |

When several alarms are active at once, the one with the highest priority sounds.

## How it works

```
Laptop webcam ──> YOLO11n-pose ──> 17 keypoints per person
                                     ├─> fall classifier (Random Forest) ──> FALL 1/0 ─┐
                                     └─> feet inside restricted polygon? ──> ZONE 1/0 ─┤
                                                                                       │ USB serial
                                                     DHT11 ──> ESP32 <─────────────────┘
                                                                 └──> Buzzer + red LED
```

### Fall detection (the machine-learning part)

1. **Pose estimation:** YOLO11n-pose (pretrained, downloaded automatically) gives a box and 17 body keypoints (shoulders, hips, knees, ankles, …) for every person.
2. **Features ([fall_features.py](fall_features.py)):** each person becomes a 54-number vector: the keypoint positions relative to the box (scale-invariant), which keypoints are visible, the box's width/height ratio, and the angle of the torso from vertical.
3. **Classifier:** a Random Forest trained on **your own recorded data** labels each person `normal` or `fall`. If you haven't trained one yet, a simple posture rule (torso tilted more than about 60°, or a very wide box) is used instead, so the system still works.
4. **Confirmation over time:** the FALL alarm only starts after the person has looked fallen for 8 frames in a row (about 1 second). Bending down to pick something up doesn't trigger it.

### Restricted area

You draw a polygon on the live video by clicking its corners. Each person's **feet point** (bottom-centre of their box) is tested against the polygon. If the feet are inside for 3 frames, the ZONE alarm starts. It switches off after 15 frames with nobody inside. Using the feet rather than the whole box means someone walking *in front of* the zone doesn't count.

### Other details

- **Event photos:** each time an alarm starts, a photo of the annotated frame is saved in `events/`. These are useful as evidence in your report.
- **Recording:** `--save demo.mp4` records the annotated video, so you have a demo clip to submit.
- **Safety watchdog:** the laptop resends the alarm state every second. If the ESP32 hears nothing for 5 seconds (for example the laptop crashed or the cable was pulled), it switches the camera alarms off so the buzzer isn't stuck on. The temperature alarm keeps working on its own.

## Hardware

- ESP32 DevKit (ESP32 Dev Module)
- 3-pin buzzer module (GND / VCC / I/O). The firmware drives it with a tone, so passive buzzer modules work too.
- DHT11 temperature and humidity sensor (3-pin module)
- Red LED and a 220–330 Ω resistor
- USB cable from the ESP32 to the laptop

### Wiring

| Component | Component pin | ESP32 pin |
|---|---|---|
| Buzzer | VCC | 3V3 |
| Buzzer | GND | GND |
| Buzzer | I/O | GPIO25 |
| DHT11 | VCC (+) | 3V3 (use VIN/5V if readings are wrong) |
| DHT11 | GND (−) | GND |
| DHT11 | DATA / OUT | GPIO27 |
| Red LED | Long leg (+) | GPIO26, through a 220–330 Ω resistor |
| Red LED | Short leg (−) | GND |

```
ESP32 GPIO25 ──────────────────── Buzzer I/O
ESP32 GPIO27 ──────────────────── DHT11 DATA
ESP32 GPIO26 ──[ 220–330 Ω ]───── LED (+)   LED (−) ── GND
ESP32 3V3 ─────────────────────── Buzzer VCC, DHT11 VCC
ESP32 GND ─────────────────────── Buzzer GND, DHT11 GND, LED (−)
```

Avoid GPIO 0, 2, 5, 12 and 15, which affect booting; GPIO 6–11, which the flash uses; GPIO 34–39, which are input only; and GPIO 1 and 3, the USB serial lines.

## Setup

### 1. Flash the ESP32

1. Install the [Arduino IDE](https://www.arduino.cc/en/software), then install the **esp32 by Espressif Systems** board package (version 3.x) from Boards Manager.
2. In Library Manager, install **DHT sensor library for ESPx** by beegee_tokyo.
3. Open `firmware/sentinelx_esp32/sentinelx_esp32.ino`, select the **ESP32 Dev Module** board and your COM port, and click Upload.
4. Open Serial Monitor at **115200** baud, with the line ending set to Newline. You should see `SentinelX ESP32 Ready`, followed by a `TEMP ... HUM ...` line every 2 seconds. To test the sounds, type `FALL 1` or `ZONE 1`, and then `CLEAR`. (Without the laptop program running, the watchdog switches the sound off by itself after 5 seconds. That is expected.)
5. **Close Serial Monitor before running the Python scripts.** Only one program can use the COM port at a time.

### 2. Set up Python

Python 3.10 or newer is required.

```bash
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Linux / macOS

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To keep the install small, or if pip runs out of memory, install the CPU build of PyTorch on its own first:

```bash
python -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-cache-dir -r requirements.txt
```

### 3. Fall detector: already trained

The project comes with a trained model (`fall_model.joblib`) and its training data (`fall_data.csv`), so **you don't need to train anything**. Just run step 4.

If your laptop has a different scikit-learn version from the one the model was saved with, `sentinelx.py` automatically rebuilds the model from `fall_data.csv` the first time it runs. This takes a few seconds.

**How the included model was trained**

| | |
|---|---|
| Data | 2,123 people from 2 public datasets (1,210 fallen, 913 normal: standing, walking, sitting, lying on a bed, …), with mirrored copies added as augmentation |
| Model | Random Forest, 200 trees, on 54 pose features from YOLO11n-pose |
| Evaluation | 5-fold cross-validation; near-copies of the same photo are always kept in the same fold |
| **Result** | **Accuracy 91.2 %, fall F1 0.92** (fall precision 0.94, recall 0.90) |
| Simple posture rule, for comparison | Accuracy 74.0 %, fall F1 0.72 |

The full numbers are in `training_report.txt`. To rebuild the dataset from scratch, see `tools/build_public_dataset.py`.

Datasets used (credit):
- *Human-Fall-Detection* image set by bakshtb / Y-B-Class-Projects: https://github.com/Y-B-Class-Projects/Human-Fall-Detection
- *CCTV Fall Detection Dataset (free sample)* © Simuletic, CC BY-NC 4.0: https://github.com/Simuletic/cctv-fall-detection-dataset. Non-commercial use only, which is fine for a college project.

### Optional: improve the model with your own recordings

The public images are a good start. Adding a few clips recorded **with your own camera and room** makes it more reliable in your demo. It takes about 15 minutes.

**a) Record data.** Run this (`--out my_data.csv` keeps your recordings separate):

```bash
python collect_fall_data.py --out my_data.csv
```

A camera window opens. Press **N** and act normally (stand, walk, sit on a chair, bend down, tie a shoelace, wave). Then press **F** and lie on the floor in different ways: on your back, on your side, face down, curled up, with your head towards and away from the camera. Press **SPACE** to pause while you get into position, and **Q** to quit.

Tips for a good model:
- Use a **yoga mat or a few jackets** on the floor, and **lie down, don't actually fall hard**. The model learns the *fallen posture*, not the impact.
- Record **at least 6–8 short clips of each class** (press N or F again to start a new clip), aiming for around 500 or more rows each. The counter on screen shows how many rows you've saved.
- Change the camera angle, distance and lighting between clips, and if possible get 2–3 friends to take part. This matters more than the number of rows.
- Include the confusing cases under **N**: sitting on the floor, bending down, crouching. These are what make the model better than the simple rule.
- You can also label existing videos, for example a public fall dataset such as UR Fall Detection or Le2i: `python collect_fall_data.py --source clip.mp4` (label with the keys), or `--label fall` / `--label normal` for clips that contain only one posture.

**b) Train on the public data and your data together.** Run:

```bash
python train_fall_model.py --data fall_data.csv --data my_data.csv
```

This prints accuracy, F1, a confusion matrix and the most important features, and compares them with the rule-based baseline. It saves `fall_model.joblib` and writes everything to `training_report.txt`, which you can copy into your report. The evaluation keeps each recording clip entirely in either the training or the test set, so the scores reflect how well the model handles new situations.

`sentinelx.py` loads `fall_model.joblib` automatically. If there is no model and no `fall_data.csv`, the rule-based check is used.

### 4. Run

```bash
python sentinelx.py                     # webcam + ESP32 on COM7
python sentinelx.py --port COM5         # a different COM port
python sentinelx.py --port none         # webcam only, no ESP32 (test mode)
python sentinelx.py --save demo.mp4     # also record the annotated video
python sentinelx.py --source video.mp4  # use a video file instead of the webcam
```

Keys in the video window:

| Key | Action |
|---|---|
| **Z** | Start drawing the restricted zone. **Left-click** each corner, **right-click** to undo a corner, then press **Z** again to save it. The zone is saved in `zone.json` and loaded automatically next time. |
| **C** | Clear the zone |
| **Q** | Quit. The buzzer is always switched off on exit. |

To find your COM port, open Windows Device Manager → Ports (COM & LPT).

## Live demo plan (works in a normal classroom or lab)

Your college has no real restricted area, so you **make one**. The zone is virtual: any part of the floor the camera can see can be set as a restricted area.

**Setup (5 minutes):**
1. Put the laptop on a desk at one end of the room, with the webcam seeing the floor area in front of the board or near a door, about 3–5 m away. A higher camera works better (stack a few books).
2. Choose a demo restricted area, such as the space in front of the **teacher's desk** or a **lab equipment cabinet**, or a **"server rack / chemical store" corner**. Mark its corners on the floor with **coloured tape** and stick up a printed **"⚠ RESTRICTED AREA – AUTHORISED PERSONNEL ONLY"** sign, so the audience can see a physical boundary.
3. Run `python sentinelx.py --save demo.mp4`, press **Z**, click the four taped corners on the video, and press **Z** again. A red shaded "RESTRICTED AREA" now appears exactly over the taped area.
4. Lay a yoga mat (or jackets) in view but **outside** the zone for the fall demo.

**Demo script:**

| Step | What you do | What the audience sees and hears |
|---|---|---|
| 1 | Walk around normally in front of the camera | Green boxes, "MONITORING", **no buzzer**. Point out that a person alone doesn't trigger anything. |
| 2 | Walk past just in front of the zone without stepping in | Still no alarm, because only the feet count. |
| 3 | Step inside the taped area | Box turns orange, the zone flashes red, the banner shows "RESTRICTED AREA ENTERED", and the **steady buzz** with the **LED on** starts. A photo is saved in `events/`. |
| 4 | Step out | The alarm stops after about a second. |
| 5 | Sit on a chair, then bend down to pick up a pen | No fall alarm. This shows the model isn't fooled by ordinary movements. |
| 6 | Slowly "collapse" onto the mat and stay down | Box turns red and the fall score rises. After about 1 s: "PERSON FALL DETECTED", **fast beeping** and a **fast-blinking LED**. |
| 7 | Get up | The alarm clears. |
| 8 | Breathe warm air on the DHT11, or hold it near a warm (not hot) phone charger | Temperature rises on screen, and at 35 °C the **two-tone siren** starts. |
| 9 | Unplug the USB cable during an alarm (optional) | The buzzer stops within 5 s, showing the safety watchdog. |

**If you can't do it live** (no space, or a viva in a small cabin), use the backups:
- **Recorded demo:** record `demo.mp4` beforehand with `--save` and play it. You can also run the system on that recording with `--source demo.mp4` and the ESP32 connected: the buzzer reacts to the video exactly as it would live.
- **Desk-sized version:** point the camera at a wall or open door and set the zone as the **doorway**. Anyone walking through the door is an "intrusion", which makes a good "unauthorised entry into the lab" story.
- **Phone video:** play a fall clip on a phone or second laptop screen in front of the webcam. YOLO will detect the person in the video.

Talking points: hospital wards and elderly-care homes (fall alerts), labs, server rooms, high-voltage areas and factory machine zones (intrusion alerts), and the temperature alarm for fire or overheating.

## Options

| Option | Default | Description |
|---|---|---|
| `--source` | `0` | Camera index (`0` = built-in webcam) or video file path |
| `--port` | `COM7` | ESP32 serial port, or `none` for test mode |
| `--model` | `yolo11n-pose.pt` | YOLO pose weights |
| `--fall-model` | `fall_model.joblib` | Trained fall classifier |
| `--zone` | `zone.json` | File the restricted zone is saved in |
| `--conf` | `0.5` | Minimum confidence for a person detection |
| `--imgsz` | `640` | Inference size; `480` or `320` is faster on slow CPUs |
| `--fall-threshold` | `0.6` | Fall probability needed for a frame to count as "fallen" |
| `--fall-frames` | `8` | Fallen frames in a row before the FALL alarm |
| `--zone-frames` | `3` | Frames inside the zone before the ZONE alarm |
| `--off-frames` | `15` | Clear frames in a row before an alarm switches off |
| `--save` | – | Record the annotated video to this `.mp4` |
| `--snapshots` | `events` | Folder for alarm photos (`""` turns them off) |
| `--no-window` | off | Run without a window (for testing on video files) |

To change the temperature limit, edit `TEMP_LIMIT` (alarm on) and `TEMP_RESET` (alarm off) at the top of the firmware. Keep `TEMP_RESET` about 1 °C lower so the alarm doesn't flicker at the limit.

### Serial protocol

| Direction | Message | Meaning |
|---|---|---|
| Laptop → ESP32 | `FALL 1` / `FALL 0` | Fall alarm on / off (resent every second) |
| Laptop → ESP32 | `ZONE 1` / `ZONE 0` | Restricted-area alarm on / off (resent every second) |
| Laptop → ESP32 | `CLEAR` | Both camera alarms off |
| ESP32 → Laptop | `TEMP 28.0 HUM 55 ALARM 0` | Sensor reading every 2 s |
| ESP32 → Laptop | `DHT ERROR <reason> (...)` | Bad sensor reading (ignored) |

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not open COM7 ... Access is denied` | Another program is using the port, usually the Arduino Serial Monitor. Close it, or unplug and replug the ESP32. |
| Camera doesn't open | Try `--source 1`, and check Windows Settings → Privacy → Camera. |
| Fall alarm triggers when sitting or bending | Record more **N** clips of those movements and retrain, or raise `--fall-threshold` to `0.7` or `--fall-frames` to `12`. |
| Fall not detected when lying down | Record more **F** clips from your demo camera angle and retrain. Make sure your whole body is in view. |
| Zone alarm doesn't trigger | The feet must be visible. Make sure the camera sees the floor and the zone isn't drawn too close to the bottom edge of the frame. |
| Buzzer is very quiet | Move the buzzer's VCC from 3V3 to VIN (5V). |
| LED doesn't light | The LED is probably in backwards. The long leg goes to the resistor and GPIO26. |
| Video shows `DHT11 ERROR - check wiring` | The ESP32 got an unrealistic reading and ignored it. Check that DATA is on GPIO27, use short wires, and try powering the DHT11 from VIN (5V). If readings stay wrong, replace the sensor. |
| Low FPS | Use `--imgsz 480` or `--imgsz 320`. |
| `MemoryError` during `pip install` | Close other apps, install with `--no-cache-dir`, and make sure the Windows page file is enabled. |

## Project structure

```
sentinelx.py                                  Main program: webcam + pose + fall + zone + serial link
fall_features.py                              Pose features, rule-based fallback, classifier loader
collect_fall_data.py                          Record and label your own pose data (N = normal, F = fall)
train_fall_model.py                           Train and evaluate the Random Forest -> fall_model.joblib
fall_model.joblib                             Trained fall classifier (included)
fall_data.csv                                 Training data from the public datasets (included)
training_report.txt                           Accuracy, confusion matrix, feature importance (included)
tools/build_public_dataset.py                 How fall_data.csv was built from the public datasets
requirements.txt                              Python dependencies
firmware/sentinelx_esp32/sentinelx_esp32.ino  ESP32: buzzer patterns, LED, DHT11 alarm, watchdog

Created when you run it:
zone.json            your restricted zone
events/              photos of each alarm
yolo11n-pose.pt      pose model (downloaded automatically on the first run)
```
