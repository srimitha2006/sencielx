"""
SentinelX - fall detection + restricted-zone intrusion + DHT11 temperature alarm.

Laptop webcam -> YOLO11n-pose (people + 17 body keypoints)
  -> fall classifier (trained Random Forest, or a posture rule if not trained yet)
  -> restricted zone check (feet of each person inside a polygon you draw)
  -> sends "FALL 1/0" and "ZONE 1/0" to the ESP32 over USB serial.
The ESP32 runs the DHT11 temperature alarm itself and reports readings back.

A person just standing in view does NOT trigger the buzzer any more.

Keys in the video window:
  Z  start / finish drawing the restricted zone (left-click = add corner,
     right-click = undo last corner). The zone is saved to zone.json.
  C  clear the zone
  Q  quit

Run:
    python sentinelx.py                     # webcam + ESP32 on COM7
    python sentinelx.py --port none         # test mode, no ESP32
    python sentinelx.py --save demo.mp4     # also record the annotated video
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from fall_features import FallClassifier, extract_people, person_features

KEEPALIVE_S = 1.0  # resend alarm state this often; the ESP32 clears camera alarms if the laptop goes quiet


def parse_args():
    p = argparse.ArgumentParser(description="SentinelX fall + restricted-zone alarm")
    p.add_argument("--source", default="0", help="camera index (0 = laptop webcam) or video file path")
    p.add_argument("--port", default="COM7", help="ESP32 serial port (use --port none for test mode)")
    p.add_argument("--model", default="yolo11n-pose.pt", help="YOLO pose weights (downloaded automatically)")
    p.add_argument("--fall-model", default="fall_model.joblib", help="trained fall classifier from train_fall_model.py")
    p.add_argument("--zone", default="zone.json", help="file the restricted zone is saved in")
    p.add_argument("--conf", type=float, default=0.5, help="minimum person confidence")
    p.add_argument("--imgsz", type=int, default=640, help="inference size; 480 or 320 is faster on slow CPUs")
    p.add_argument("--fall-threshold", type=float, default=0.6, help="fall probability needed for a 'fallen' frame")
    p.add_argument("--fall-frames", type=int, default=8, help="fallen frames in a row before the FALL alarm")
    p.add_argument("--zone-frames", type=int, default=3, help="frames in the zone before the ZONE alarm")
    p.add_argument("--off-frames", type=int, default=15, help="clear frames in a row before an alarm switches off")
    p.add_argument("--save", help="record the annotated video to this .mp4 file")
    p.add_argument("--snapshots", default="events", help="folder for a photo of every alarm (empty = off)")
    p.add_argument("--no-window", action="store_true", help="don't open a window (for testing on video files)")
    return p.parse_args()


# ---------------------------------------------------------------- camera / serial

def open_camera(source):
    if source.isdigit():
        # CAP_DSHOW opens Windows laptop webcams much faster than the default backend
        cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(int(source))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source {source!r}. Try --source 1 or check camera privacy settings.")
    return cap


def open_serial(port):
    if port is None or port.lower() == "none":
        print("[SentinelX] Test mode (no ESP32).")
        return None
    import serial

    try:
        # Open with DTR/RTS released. On many ESP32 boards an open port with DTR/RTS
        # asserted holds the chip in reset, so it never receives the commands.
        esp = serial.Serial()
        esp.port, esp.baudrate, esp.timeout = port, 115200, 0
        esp.dtr = False
        esp.rts = False
        esp.open()
    except serial.SerialException as e:
        print("\n" + "!" * 70)
        print(f"[SentinelX] COULD NOT OPEN {port}: {e}")
        print("[SentinelX] The BUZZER WILL NOT WORK. Close the Arduino IDE / Serial Monitor")
        print("[SentinelX] and check the COM port in Device Manager, then run again.")
        print("!" * 70 + "\n")
        return None
    time.sleep(2)  # give the ESP32 time to boot if it reset
    esp.reset_input_buffer()
    print(f"[SentinelX] Serial port {port} opened - waiting for the ESP32 to answer...")
    return esp


def send(esp, command):
    if esp is None:
        return
    try:
        esp.write((command + "\n").encode("utf-8"))
    except Exception as e:
        print(f"[SentinelX] Serial write failed: {e}")


def read_sensor(esp, buf, sensor):
    """Non-blocking read of lines from the ESP32:
    'TEMP 28.0 HUM 55 ALARM 0 FALL 0 ZONE 0', 'DHT ERROR ...', 'FALL ON', 'ZONE OFF', ..."""
    if esp is None:
        return buf
    try:
        buf += esp.read(esp.in_waiting or 1).decode("utf-8", errors="ignore")
    except Exception:
        return buf
    *lines, buf = buf.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not sensor["seen"]:
            sensor["seen"] = True
            print("[SentinelX] ESP32 is answering - buzzer link OK.")
        parts = line.split()
        if len(parts) >= 6 and parts[0] == "TEMP":
            try:
                temp, hum, alarm = float(parts[1]), float(parts[3]), parts[5] == "1"
            except ValueError:
                continue
            if alarm != sensor["alarm"]:
                print(f"[SentinelX] TEMPERATURE {'HIGH -> SIREN ON' if alarm else 'NORMAL -> SIREN OFF'} ({temp:.1f} C)")
            sensor.update(temp=temp, hum=hum, alarm=alarm, error=False)
        elif line.startswith("DHT ERROR"):  # firmware adds details after "DHT ERROR"
            if not sensor["error"]:
                print(f"[{time.strftime('%H:%M:%S')}] DHT11 read error - check wiring ({line})")
            sensor["error"] = True
        elif parts[0] == "STATE" and len(parts) >= 5:  # "STATE FALL 0 ZONE 1" every 2 s
            sensor["new_fw"] = True
            sensor["esp_fall"], sensor["esp_zone"] = parts[2] == "1", parts[4] == "1"
        elif parts[0] in ("FALL", "ZONE") and len(parts) == 2 and parts[1] in ("ON", "OFF"):
            sensor["new_fw"] = True
            sensor["esp_" + parts[0].lower()] = parts[1] == "ON"
            print(f"[ESP32] confirmed: {parts[0]} alarm {parts[1]}")
        elif line.startswith("HOST TIMEOUT") or line.startswith("SentinelX"):
            print(f"[ESP32] {line}")
    return buf


# ---------------------------------------------------------------- restricted zone

class Zone:
    """Polygon stored in 0..1 coordinates so it works at any camera resolution."""

    def __init__(self, path):
        self.path = path
        self.points = []   # normalised [x, y]
        self.editing = False
        self.draft = []
        try:
            with open(path) as f:
                self.points = json.load(f)["points"]
            print(f"[SentinelX] Restricted zone loaded from {path} ({len(self.points)} corners).")
        except FileNotFoundError:
            print("[SentinelX] No restricted zone yet - press Z in the video window to draw one.")
        except Exception as e:
            print(f"[SentinelX] Could not read {path}: {e}")

    def save(self):
        with open(self.path, "w") as f:
            json.dump({"points": self.points}, f, indent=1)

    def pixel_poly(self, w, h, pts=None):
        pts = self.points if pts is None else pts
        return np.array([[int(x * w), int(y * h)] for x, y in pts], dtype=np.int32)

    def contains(self, x, y, w, h):
        if len(self.points) < 3:
            return False
        return cv2.pointPolygonTest(self.pixel_poly(w, h), (float(x), float(y)), False) >= 0

    def toggle_edit(self):
        if not self.editing:
            self.editing, self.draft = True, []
            print("[SentinelX] Drawing zone: left-click corners, right-click to undo, press Z to finish.")
        else:
            self.editing = False
            if len(self.draft) >= 3:
                self.points = self.draft
                self.save()
                print(f"[SentinelX] Restricted zone saved to {self.path}.")
            else:
                print("[SentinelX] Zone needs at least 3 corners - kept the old zone.")

    def clear(self):
        self.points, self.draft, self.editing = [], [], False
        if os.path.exists(self.path):
            os.remove(self.path)
        print("[SentinelX] Restricted zone cleared.")

    def on_mouse(self, event, x, y, flags, size):
        if not self.editing:
            return
        w, h = size
        if event == cv2.EVENT_LBUTTONDOWN:
            self.draft.append([x / w, y / h])
        elif event == cv2.EVENT_RBUTTONDOWN and self.draft:
            self.draft.pop()

    def draw(self, img, intruded):
        h, w = img.shape[:2]
        if self.editing:
            poly = self.pixel_poly(w, h, self.draft)
            for p in poly:
                cv2.circle(img, tuple(int(v) for v in p), 6, (0, 255, 255), -1)
            if len(poly) >= 2:
                cv2.polylines(img, [poly], len(poly) >= 3, (0, 255, 255), 2)
            cv2.putText(img, "DRAW ZONE: click corners, right-click undo, Z = save",
                        (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            return
        if len(self.points) < 3:
            return
        poly = self.pixel_poly(w, h)
        overlay = img.copy()
        cv2.fillPoly(overlay, [poly], (0, 0, 255))
        cv2.addWeighted(overlay, 0.40 if intruded else 0.18, img, 0.60 if intruded else 0.82, 0, img)
        cv2.polylines(img, [poly], True, (0, 0, 255), 3 if intruded else 2)
        top = poly[np.argmin(poly[:, 1])]
        label_x = min(max(int(top[0]) - 70, 5), w - 180)
        cv2.putText(img, "RESTRICTED AREA", (label_x, max(int(top[1]) - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


# ---------------------------------------------------------------- alarm debounce

class Debounce:
    """Turns ON after `on` true frames in a row, OFF after `off` false frames in a row."""

    def __init__(self, on, off):
        self.on, self.off = on, off
        self.state, self.hits, self.misses = False, 0, 0

    def update(self, value):
        if value:
            self.hits, self.misses = self.hits + 1, 0
        else:
            self.hits, self.misses = 0, self.misses + 1
        if not self.state and self.hits >= self.on:
            self.state = True
            return "on"
        if self.state and self.misses >= self.off:
            self.state = False
            return "off"
        return None


# ---------------------------------------------------------------- main

def main():
    args = parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[SentinelX] Inference device: {'GPU (CUDA)' if device == 0 else 'CPU'}")

    model = YOLO(args.model)
    fall_clf = FallClassifier(args.fall_model)
    print(f"[SentinelX] Fall detection: {fall_clf.source}")
    zone = Zone(args.zone)
    cap = open_camera(args.source)
    esp = open_serial(args.port)

    win = "SentinelX"
    frame_size = [640, 480]
    if not args.no_window:
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, zone.on_mouse, frame_size)

    if args.snapshots:
        os.makedirs(args.snapshots, exist_ok=True)

    fall_alarm = Debounce(args.fall_frames, args.off_frames)
    zone_alarm = Debounce(args.zone_frames, args.off_frames)
    sensor = {"temp": None, "hum": None, "alarm": False, "error": False,
              "seen": False, "new_fw": False, "esp_fall": False, "esp_zone": False}
    start_time = time.time()
    warned_fw = False
    serial_buf = ""
    writer = None
    last_sent = 0.0
    send(esp, "CLEAR")

    prev = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[SentinelX] No more frames.")
                break
            h, w = frame.shape[:2]
            frame_size[0], frame_size[1] = w, h

            result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, device=device, verbose=False)[0]
            people = extract_people(result)
            scores = fall_clf.scores([person_features(b, k, c) for b, k, c in people])

            # ---- per-person checks
            any_fall, any_zone = False, False
            view = frame.copy()
            for (box, kxy, kconf), score in zip(people, scores):
                x1, y1, x2, y2 = map(int, box)
                feet = ((x1 + x2) // 2, y2)  # bottom-centre of the box = where the person stands
                fallen = score >= args.fall_threshold
                inside = zone.contains(feet[0], feet[1], w, h)
                any_fall |= fallen
                any_zone |= inside

                color = (0, 0, 255) if fallen else (0, 140, 255) if inside else (0, 200, 0)
                tag = "FALLEN" if fallen else "IN ZONE" if inside else "person"
                cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
                for (kx, ky), kc in zip(kxy, kconf):
                    if kc >= 0.3:
                        cv2.circle(view, (int(kx), int(ky)), 3, color, -1)
                cv2.circle(view, feet, 6, color, -1)
                cv2.putText(view, f"{tag}  fall {score:.2f}", (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # ---- alarms
            events = []
            change = fall_alarm.update(any_fall)
            if change:
                events.append(("FALL", change))
            change = zone_alarm.update(any_zone)
            if change:
                events.append(("ZONE", change))

            now = time.time()
            for name, change in events:
                stamp = time.strftime("%H:%M:%S")
                msg = {"FALL": "PERSON FALL DETECTED", "ZONE": "RESTRICTED AREA ENTERED"}[name]
                print(f"[{stamp}] {msg + ' -> BUZZER ON' if change == 'on' else name + ' cleared'}")
            if events or now - last_sent >= KEEPALIVE_S:
                send(esp, f"FALL {int(fall_alarm.state)}")
                send(esp, f"ZONE {int(zone_alarm.state)}")
                last_sent = now

            # ---- overlays
            zone.draw(view, zone_alarm.state)
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now

            if fall_alarm.state:
                banner, bcolor = "ALERT: PERSON FALL DETECTED", (0, 0, 255)
            elif zone_alarm.state:
                banner, bcolor = "ALERT: RESTRICTED AREA ENTERED", (0, 0, 255)
            else:
                banner, bcolor = f"MONITORING - {len(people)} person(s)", (0, 200, 0)
            (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(view, (10, 10), (20 + bw, 20 + bh + 8), (0, 0, 0), -1)
            cv2.putText(view, banner, (15, 15 + bh), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bcolor, 2)
            cv2.putText(view, f"FPS {fps:.1f}  |  {'ESP32 ' + args.port if esp else 'TEST MODE'}  |  "
                              f"Z = draw zone  C = clear  Q = quit",
                        (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            serial_buf = read_sensor(esp, serial_buf, sensor)
            if sensor["error"]:
                temp_text, temp_color = "DHT11 ERROR - check wiring", (0, 165, 255)
            elif sensor["temp"] is None:
                temp_text, temp_color = "TEMP --", (255, 255, 255)
            else:
                temp_text = f"TEMP {sensor['temp']:.1f} C  HUM {sensor['hum']:.0f}%"
                if sensor["alarm"]:
                    temp_text += "  - HIGH TEMP SIREN"
                temp_color = (0, 0, 255) if sensor["alarm"] else (0, 200, 0)
            (tw, th), _ = cv2.getTextSize(temp_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(view, (10, 72), (20 + tw, 82 + th), (0, 0, 0), -1)
            cv2.putText(view, temp_text, (15, 77 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.7, temp_color, 2)

            # ---- buzzer link status: is the ESP32 really getting the alarms?
            waited = now - start_time
            if esp is None:
                link, lcolor = "BUZZER: NOT CONNECTED (test mode)", (0, 0, 255)
            elif not sensor["seen"]:
                link, lcolor = ("BUZZER: waiting for ESP32..." if waited < 8 else
                                "BUZZER: NO REPLY - wrong COM port?"), (0, 165, 255)
            elif not sensor["new_fw"]:
                link, lcolor = ("BUZZER: checking firmware..." if waited < 8 else
                                "BUZZER: OLD ARDUINO CODE - upload sentinelx_esp32.ino"), (0, 165, 255)
            else:
                on = [n for n, s in (("FALL", sensor["esp_fall"]), ("ZONE", sensor["esp_zone"])) if s]
                link = "BUZZER (ESP32): " + (" + ".join(on) + " ALARM ON" if on else "ready")
                lcolor = (0, 0, 255) if on else (0, 200, 0)
            (lw, lh), _ = cv2.getTextSize(link, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(view, (10, 92 + th), (20 + lw, 102 + th + lh), (0, 0, 0), -1)
            cv2.putText(view, link, (15, 97 + th + lh), cv2.FONT_HERSHEY_SIMPLEX, 0.6, lcolor, 2)

            if esp is not None and not warned_fw and waited >= 8:
                warned_fw = True
                if not sensor["seen"]:
                    print("[SentinelX] WARNING: no data from the ESP32 on " + args.port +
                          ". Wrong COM port, or the board is not running. Press the EN/RST button on the ESP32.")
                elif not sensor["new_fw"]:
                    print("[SentinelX] WARNING: the ESP32 is running the OLD Arduino code. It ignores FALL/ZONE,"
                          " so the buzzer stays off. Upload firmware/sentinelx_esp32/sentinelx_esp32.ino.")

            # photo of every new alarm, taken after all overlays are drawn
            if args.snapshots:
                for name, change in events:
                    if change == "on":
                        cv2.imwrite(os.path.join(args.snapshots,
                                                 f"{time.strftime('%Y%m%d-%H%M%S')}_{name.lower()}.jpg"), view)

            if args.save:
                if writer is None:
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
                writer.write(view)

            if not args.no_window:
                cv2.imshow(win, view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("z"):
                    zone.toggle_edit()
                elif key == ord("c"):
                    zone.clear()
    finally:
        send(esp, "CLEAR")  # always leave the buzzer off
        if esp is not None:
            esp.close()
        if writer is not None:
            writer.release()
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
