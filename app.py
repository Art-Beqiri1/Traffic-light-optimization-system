# ============================================================
# VEHICLE DETECTION & COUNTING SYSTEM
# Copyright © 2026 KujtimHackeri
#
# Sistem për:
# - Detektimin e automjeteve
# - Numërimin e automjeteve
# - Numërimin hyrje / dalje
# - Kamera RTSP
# - YOLO + ByteTrack
# - Web GUI
# ============================================================

import os
import cv2
import json
import threading
import time

from flask import (
    Flask,
    render_template,
    Response,
    request,
    jsonify
)

from ultralytics import YOLO


# ============================================================
# KONFIGURIMI I FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# SKEDARI I KONFIGURIMIT
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

CONFIG_FILE = os.path.join(
    BASE_DIR,
    "config.json"
)


# ============================================================
# KONFIGURIMI FILLESTAR
# ============================================================

DEFAULT_CONFIG = {

    # RTSP e kamerës
    "rtsp_url": "",

    # Aktivizo detektimin
    "detect": True,

    # Aktivizo numërimin
    "count": True,

    # Pozita e vijës 0.55 = 55%
    "line_position": 0.55,

    # Siguria minimale e YOLO
    "confidence": 0.35
}


# ============================================================
# NGARKO KONFIGURIMIN
# ============================================================

def load_config():

    try:

        if os.path.exists(CONFIG_FILE):

            with open(
                CONFIG_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

            # Plotësojmë vlerat që mungojnë
            for key, value in DEFAULT_CONFIG.items():

                if key not in data:

                    data[key] = value

            return data

    except Exception as error:

        print(
            "Gabim gjatë leximit të config.json:"
        )

        print(error)

    return DEFAULT_CONFIG.copy()


# ============================================================
# KONFIGURIMI AKTUAL
# ============================================================

config = load_config()


# ============================================================
# RUAJTJA E KONFIGURIMIT
# ============================================================

def save_config():

    try:

        with open(
            CONFIG_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                config,
                file,
                indent=4,
                ensure_ascii=False
            )

        return True

    except Exception as error:

        print(
            "Gabim gjatë ruajtjes së konfigurimit:"
        )

        print(error)

        return False


# ============================================================
# VARIABLAT E KAMERËS
# ============================================================

camera = None

running = False

camera_thread = None

latest_frame = None

frame_lock = threading.Lock()

camera_lock = threading.Lock()


# ============================================================
# MODEL YOLO
# ============================================================

model = None

model_lock = threading.Lock()


# ============================================================
# KLASAT E AUTOMJETEVE COCO
#
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
# ============================================================

VEHICLE_CLASSES = {

    2: "car",

    3: "motorcycle",

    5: "bus",

    7: "truck"
}


# ============================================================
# EMRAT SHQIP
# ============================================================

VEHICLE_NAMES = {

    "car": "Makina",

    "motorcycle": "Moto",

    "bus": "Autobus",

    "truck": "Kamion"
}


# ============================================================
# STATISTIKAT
# ============================================================

vehicle_counts = {

    "car": 0,

    "motorcycle": 0,

    "bus": 0,

    "truck": 0
}


total_count = 0

in_count = 0

out_count = 0


# ============================================================
# TRACKING
# ============================================================

# Pozita e fundit e ID-së
previous_positions = {}


# ID që janë numëruar
counted_ids = set()


# ============================================================
# STATUSI I KAMERËS
# ============================================================

camera_status = "Kamera nuk është startuar"

last_error = ""


# ============================================================
# NGARKIMI I MODELIT YOLO
# ============================================================

def load_model():

    global model

    print("")
    print(
        "=========================================="
    )

    print(
        "Po ngarkohet YOLO..."
    )

    print(
        "=========================================="
    )

    try:

        # yolov8n.pt është model i vogël
        # dhe më i shpejtë për CPU

        with model_lock:

            model = YOLO(
                "yolov8n.pt"
            )

        print(
            "YOLO u ngarkua me sukses."
        )

        return True

    except Exception as error:

        print(
            "Gabim gjatë ngarkimit të YOLO:"
        )

        print(error)

        model = None

        return False


# ============================================================
# RESET STATISTIKAT
# ============================================================

def reset_counters():

    global vehicle_counts

    global total_count

    global in_count

    global out_count

    global previous_positions

    global counted_ids


    vehicle_counts = {

        "car": 0,

        "motorcycle": 0,

        "bus": 0,

        "truck": 0
    }


    total_count = 0

    in_count = 0

    out_count = 0


    previous_positions.clear()

    counted_ids.clear()


# ============================================================
# MBYLLJA E KAMERËS
# ============================================================

def release_camera():

    global camera

    try:

        if camera is not None:

            camera.release()

    except Exception:

        pass

    camera = None


# ============================================================
# HAP KAMERËN RTSP
# ============================================================

def open_camera(rtsp_url):

    global camera

    if not rtsp_url:

        return False


    print(
        "Po hapet RTSP:"
    )

    print(
        rtsp_url
    )


    try:

        release_camera()


        # Hapim kamerën me FFmpeg
        new_camera = cv2.VideoCapture(

            rtsp_url,

            cv2.CAP_FFMPEG
        )


        # Buffer sa më i vogël
        new_camera.set(

            cv2.CAP_PROP_BUFFERSIZE,

            1
        )


        # Kontrollojmë kamerën
        if not new_camera.isOpened():

            new_camera.release()

            print(
                "RTSP nuk u hap."
            )

            return False


        camera = new_camera


        print(
            "RTSP u lidh me sukses."
        )


        return True


    except Exception as error:

        print(
            "Gabim RTSP:"
        )

        print(error)

        release_camera()

        return False


# ============================================================
# TESTIMI I KAMERËS RTSP
# ============================================================

def test_rtsp(url):

    if not url:

        return False


    test = None


    try:

        test = cv2.VideoCapture(

            url,

            cv2.CAP_FFMPEG
        )


        test.set(

            cv2.CAP_PROP_BUFFERSIZE,

            1
        )


        if not test.isOpened():

            return False


        # Provojmë të lexojmë një frame
        success, frame = test.read()


        if not success:

            return False


        if frame is None:

            return False


        return True


    except Exception as error:

        print(
            "Gabim gjatë testimit:"
        )

        print(error)

        return False


    finally:

        if test is not None:

            try:

                test.release()

            except Exception:

                pass


# ============================================================
# NUMËRIMI I AUTOMJETIT
# ============================================================

def count_vehicle(
    track_id,
    vehicle_type,
    previous_y,
    current_y,
    line_y
):

    global total_count

    global in_count

    global out_count


    # Nëse nuk kemi ID
    if track_id is None:

        return


    # Nëse ky ID është numëruar më parë
    if track_id in counted_ids:

        return


    # --------------------------------------------------------
    # KALIMI NGA LART POSHTË
    # --------------------------------------------------------

    if (

        previous_y < line_y

        and

        current_y >= line_y

    ):

        counted_ids.add(
            track_id
        )

        vehicle_counts[
            vehicle_type
        ] += 1

        total_count += 1

        in_count += 1


        print(
            f"HYRJE: {vehicle_type} "
            f"ID={track_id}"
        )


    # --------------------------------------------------------
    # KALIMI NGA POSHTË LART
    # --------------------------------------------------------

    elif (

        previous_y > line_y

        and

        current_y <= line_y

    ):

        counted_ids.add(
            track_id
        )

        vehicle_counts[
            vehicle_type
        ] += 1

        total_count += 1

        out_count += 1


        print(
            f"DALJE: {vehicle_type} "
            f"ID={track_id}"
        )


# ============================================================
# LOOP KRYESOR I KAMERËS
# ============================================================

def camera_loop():

    global camera

    global running

    global latest_frame

    global camera_status

    global last_error


    print(
        "Thread-i i kamerës filloi."
    )


    while running:

        # ----------------------------------------------------
        # Marrim RTSP
        # ----------------------------------------------------

        rtsp_url = config.get(
            "rtsp_url",
            ""
        )


        if not rtsp_url:

            camera_status = (
                "RTSP nuk është konfiguruar"
            )

            time.sleep(1)

            continue


        # ----------------------------------------------------
        # LIDHJA ME KAMERËN
        # ----------------------------------------------------

        if camera is None:

            camera_status = (
                "Duke u lidhur me kamerën..."
            )


            if not open_camera(
                rtsp_url
            ):

                camera_status = (
                    "Gabim lidhjeje me RTSP"
                )

                time.sleep(3)

                continue


            camera_status = (
                "Kamera është online"
            )


        # ----------------------------------------------------
        # LEXIMI I FRAME
        # ----------------------------------------------------

        try:

            success, frame = camera.read()


        except Exception as error:

            print(
                "Gabim gjatë leximit:"
            )

            print(error)

            success = False

            frame = None


        # ----------------------------------------------------
        # KAMERA U SHKËPUT
        # ----------------------------------------------------

        if not success or frame is None:

            camera_status = (
                "Kamera u shkëput - reconnect..."
            )


            print(
                "Frame nuk u lexua."
            )


            release_camera()


            time.sleep(2)

            continue


        # ----------------------------------------------------
        # DIMENSIONET
        # ----------------------------------------------------

        try:

            height, width = frame.shape[:2]

        except Exception:

            continue


        # ----------------------------------------------------
        # POZITA E VIJËS
        # ----------------------------------------------------

        try:

            line_position = float(
                config.get(
                    "line_position",
                    0.55
                )
            )

        except Exception:

            line_position = 0.55


        line_position = max(
            0.05,
            min(
                0.95,
                line_position
            )
        )


        line_y = int(
            height * line_position
        )


        # ====================================================
        # VIJA E NUMËRIMIT
        # ====================================================

        if config.get(
            "count",
            True
        ):

            cv2.line(

                frame,

                (0, line_y),

                (width, line_y),

                (0, 255, 255),

                3
            )


            cv2.putText(

                frame,

                "VIJA E NUMERIMIT",

                (
                    20,
                    max(
                        30,
                        line_y - 10
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 255, 255),

                2
            )


        # ====================================================
        # DETEKTIMI YOLO
        # ====================================================

        if (

            config.get(
                "detect",
                True
            )

            and

            model is not None

        ):

            try:

                # Confidence
                confidence = float(

                    config.get(
                        "confidence",
                        0.35
                    )
                )


                confidence = max(
                    0.10,
                    min(
                        0.95,
                        confidence
                    )
                )


                # ------------------------------------------------
                # YOLO TRACKING
                # ------------------------------------------------

                with model_lock:

                    results = model.track(

                        frame,

                        persist=True,

                        tracker="bytetrack.yaml",

                        classes=list(
                            VEHICLE_CLASSES.keys()
                        ),

                        conf=confidence,

                        verbose=False
                    )


                if not results:

                    continue


                result = results[0]


                if result.boxes is None:

                    continue


                boxes = result.boxes


                # ------------------------------------------------
                # KALIMI NËPËR AUTOMJETE
                # ------------------------------------------------

                for i in range(
                    len(boxes)
                ):

                    try:

                        # Koordinatat
                        xyxy = (

                            boxes
                            .xyxy[i]
                            .cpu()
                            .numpy()
                            .astype(int)
                        )


                        x1, y1, x2, y2 = xyxy


                        # Kufizojmë koordinatat
                        x1 = max(
                            0,
                            min(
                                width - 1,
                                x1
                            )
                        )

                        x2 = max(
                            0,
                            min(
                                width - 1,
                                x2
                            )
                        )

                        y1 = max(
                            0,
                            min(
                                height - 1,
                                y1
                            )
                        )

                        y2 = max(
                            0,
                            min(
                                height - 1,
                                y2
                            )
                        )


                        # ------------------------------------------------
                        # KLASA
                        # ------------------------------------------------

                        class_id = int(

                            boxes
                            .cls[i]
                            .cpu()
                            .numpy()
                        )


                        vehicle_type = (

                            VEHICLE_CLASSES.get(
                                class_id
                            )
                        )


                        if vehicle_type is None:

                            continue


                        # ------------------------------------------------
                        # CONFIDENCE
                        # ------------------------------------------------

                        conf = float(

                            boxes
                            .conf[i]
                            .cpu()
                            .numpy()
                        )


                        # ------------------------------------------------
                        # TRACK ID
                        # ------------------------------------------------

                        track_id = None


                        if boxes.id is not None:

                            try:

                                track_id = int(

                                    boxes
                                    .id[i]
                                    .cpu()
                                    .numpy()
                                )

                            except Exception:

                                track_id = None


                        # ------------------------------------------------
                        # QENDRA
                        # ------------------------------------------------

                        center_x = int(
                            (x1 + x2) / 2
                        )

                        center_y = int(
                            (y1 + y2) / 2
                        )


                        # ------------------------------------------------
                        # KUTIA
                        # ------------------------------------------------

                        cv2.rectangle(

                            frame,

                            (x1, y1),

                            (x2, y2),

                            (0, 255, 0),

                            2
                        )


                        # ------------------------------------------------
                        # TEKSTI
                        # ------------------------------------------------

                        label = (

                            f"{VEHICLE_NAMES[vehicle_type]} "
                            f"{conf:.2f}"
                        )


                        if track_id is not None:

                            label += (
                                f" ID:{track_id}"
                            )


                        cv2.putText(

                            frame,

                            label,

                            (
                                x1,
                                max(
                                    25,
                                    y1 - 10
                                )
                            ),

                            cv2.FONT_HERSHEY_SIMPLEX,

                            0.6,

                            (0, 255, 0),

                            2
                        )


                        # ------------------------------------------------
                        # PIKA E QENDRËS
                        # ------------------------------------------------

                        cv2.circle(

                            frame,

                            (
                                center_x,
                                center_y
                            ),

                            5,

                            (0, 0, 255),

                            -1
                        )


                        # =================================================
                        # NUMËRIMI
                        # =================================================

                        if (

                            config.get(
                                "count",
                                True
                            )

                            and

                            track_id is not None

                        ):

                            previous_y = (

                                previous_positions.get(
                                    track_id
                                )
                            )


                            if previous_y is not None:

                                count_vehicle(

                                    track_id,

                                    vehicle_type,

                                    previous_y,

                                    center_y,

                                    line_y
                                )


                            # Ruajmë pozicionin
                            previous_positions[
                                track_id
                            ] = center_y


                    except Exception as object_error:

                        print(
                            "Gabim në objekt:"
                        )

                        print(
                            object_error
                        )


            except Exception as detection_error:

                last_error = str(
                    detection_error
                )

                print(
                    "Gabim YOLO:"
                )

                print(
                    detection_error
                )


        # ====================================================
        # PANELI I INFORMACIONIT
        # ====================================================

        overlay_height = 235


        cv2.rectangle(

            frame,

            (10, 10),

            (370, overlay_height),

            (0, 0, 0),

            -1
        )


        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        cv2.putText(

            frame,

            "STATUS: " + camera_status,

            (20, 35),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.48,

            (0, 255, 0),

            1
        )


        # ----------------------------------------------------
        # STATISTIKAT
        # ----------------------------------------------------

        stats_text = [

            (
                f"MAKINA: "
                f"{vehicle_counts['car']}",
                70
            ),

            (
                f"MOTO: "
                f"{vehicle_counts['motorcycle']}",
                102
            ),

            (
                f"AUTOBUS: "
                f"{vehicle_counts['bus']}",
                134
            ),

            (
                f"KAMION: "
                f"{vehicle_counts['truck']}",
                166
            ),

            (
                f"TOTAL: "
                f"{total_count}",
                200
            )
        ]


        for text, y in stats_text:

            color = (

                (0, 255, 255)

                if "TOTAL" in text

                else

                (255, 255, 255)
            )


            cv2.putText(

                frame,

                text,

                (25, y),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                color,

                2
            )


        # ====================================================
        # FRAME I FUNDIT
        # ====================================================

        with frame_lock:

            latest_frame = frame.copy()


    # ========================================================
    # KUR SISTEMI NDALON
    # ========================================================

    release_camera()


    print(
        "Thread-i i kamerës u ndal."
    )


# ============================================================
# VIDEO STREAM PËR BROWSER
# ============================================================

def generate_frames():

    while True:

        with frame_lock:

            if latest_frame is None:

                frame = None

            else:

                frame = latest_frame.copy()


        # Nëse nuk ka frame
        if frame is None:

            # Krijojmë ekran bosh
            frame = cv2.imread(
                "no_signal.jpg"
            )


        if frame is None:

            time.sleep(0.1)

            continue


        try:

            success, buffer = cv2.imencode(

                ".jpg",

                frame,

                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    80
                ]
            )


            if not success:

                continue


            frame_bytes = (
                buffer.tobytes()
            )


            yield (

                b"--frame\r\n"

                b"Content-Type: image/jpeg\r\n\r\n"

                + frame_bytes

                + b"\r\n"
            )


        except Exception:

            time.sleep(0.1)


# ============================================================
# FAQJA KRYESORE
# ============================================================

@app.route("/")
def index():

    try:

        return render_template(

            "index.html",

            config=config
        )


    except Exception as error:

        print(
            "=========================================="
        )

        print(
            "GABIM NË index.html"
        )

        print(error)

        print(
            "=========================================="
        )


        return (

            "<h1>Gabim në Web GUI</h1>"

            "<p>Kontrollo folderin templates.</p>"

            "<pre>"
            + str(error)
            + "</pre>"

        ), 500


# ============================================================
# VIDEO
# ============================================================

@app.route("/video")
def video():

    return Response(

        generate_frames(),

        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# START
# ============================================================

@app.route(
    "/start",
    methods=["POST"]
)
def start():

    global running

    global camera_thread


    if running:

        return jsonify({

            "success": True,

            "message":
                "Sistemi është aktiv."
        })


    # Kontrollojmë RTSP
    if not config.get(
        "rtsp_url",
        ""
    ):

        return jsonify({

            "success": False,

            "message":
                "Vendos RTSP URL."
        }), 400


    running = True


    camera_thread = threading.Thread(

        target=camera_loop,

        daemon=True,

        name="VehicleCamera"
    )


    camera_thread.start()


    return jsonify({

        "success": True,

        "message":
            "Sistemi u startua."
    })


# ============================================================
# STOP
# ============================================================

@app.route(
    "/stop",
    methods=["POST"]
)
def stop():

    global running


    running = False


    release_camera()


    return jsonify({

        "success": True,

        "message":
            "Sistemi u ndal."
    })


# ============================================================
# SETTINGS
# ============================================================

@app.route(
    "/settings",
    methods=["POST"]
)
def settings():

    global config


    try:

        data = request.get_json(
            silent=True
        ) or {}


        rtsp_url = str(
            data.get(
                "rtsp_url",
                ""
            )
        ).strip()


        detect = bool(
            data.get(
                "detect",
                True
            )
        )


        count = bool(
            data.get(
                "count",
                True
            )
        )


        line_position = float(

            data.get(
                "line_position",
                0.55
            )
        )


        confidence = float(

            data.get(
                "confidence",
                0.35
            )
        )


        # Kufizojmë vlerat
        line_position = max(
            0.05,
            min(
                0.95,
                line_position
            )
        )


        confidence = max(
            0.10,
            min(
                0.95,
                confidence
            )
        )


        config = {

            "rtsp_url":
                rtsp_url,

            "detect":
                detect,

            "count":
                count,

            "line_position":
                line_position,

            "confidence":
                confidence
        }


        save_config()


        return jsonify({

            "success": True,

            "message":
                "Konfigurimi u ruajt."
        })


    except Exception as error:

        print(
            "Gabim settings:"
        )

        print(error)


        return jsonify({

            "success": False,

            "message":
                str(error)

        }), 500


# ============================================================
# TEST CAMERA
# ============================================================

@app.route(
    "/test_camera",
    methods=["POST"]
)
def test_camera():

    try:

        data = request.get_json(
            silent=True
        ) or {}


        url = str(
            data.get(
                "rtsp_url",
                ""
            )
        ).strip()


        if not url:

            return jsonify({

                "success": False,

                "message":
                    "RTSP URL është bosh."
            }), 400


        print(
            "Po testohet kamera:"
        )

        print(
            url
        )


        result = test_rtsp(
            url
        )


        if result:

            return jsonify({

                "success": True,

                "message":
                    "Kamera RTSP funksionon."
            })


        return jsonify({

            "success": False,

            "message":
                "Nuk mund të lidhet me kamerën RTSP."
        })


    except Exception as error:

        return jsonify({

            "success": False,

            "message":
                str(error)

        }), 500


# ============================================================
# STATISTIKAT
# ============================================================

@app.route("/stats")
def stats():

    return jsonify({

        "car":
            vehicle_counts["car"],

        "motorcycle":
            vehicle_counts["motorcycle"],

        "bus":
            vehicle_counts["bus"],

        "truck":
            vehicle_counts["truck"],

        "total":
            total_count,

        "in_count":
            in_count,

        "out_count":
            out_count,

        "running":
            running,

        "camera_status":
            camera_status,

        "model_loaded":
            model is not None,

        "last_error":
            last_error
    })


# ============================================================
# RESET
# ============================================================

@app.route(
    "/reset",
    methods=["POST"]
)
def reset():

    reset_counters()


    return jsonify({

        "success": True,

        "message":
            "Numërimi u resetua."
    })


# ============================================================
# STATUS
# ============================================================

@app.route("/status")
def status():

    return jsonify({

        "running":
            running,

        "camera":
            camera is not None,

        "model":
            model is not None,

        "camera_status":
            camera_status,

        "last_error":
            last_error
    })


# ============================================================
# API PËR KONFIGURIMIN
# ============================================================

@app.route("/config")
def get_config():

    return jsonify(config)


# ============================================================
# ERROR HANDLER 500
# ============================================================

@app.errorhandler(500)
def internal_error(error):

    print(
        "=========================================="
    )

    print(
        "FLASK INTERNAL SERVER ERROR"
    )

    print(error)

    print(
        "=========================================="
    )


    return jsonify({

        "success":
            False,

        "error":
            "Internal Server Error",

        "details":
            str(error)

    }), 500


# ============================================================
# ERROR HANDLER 404
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({

        "success":
            False,

        "error":
            "Adresa nuk ekziston."

    }), 404


# ============================================================
# PROGRAMI KRYESOR
# ============================================================

if __name__ == "__main__":

    print("")
    print(
        "=================================================="
    )

    print(
        "      KUJTIMHACKERI  DETEKTIMI I AUTOMJETEVE"
    )

    print(
        "      DETEKTIMI I AUTOMJETEVE DHE NUMERIMI"
    )

    print(
        "      Copyright © 2026 KujtimHackeri"
    )

    print(
        "=================================================="
    )

    print("")


    # --------------------------------------------------------
    # NGARKO YOLO
    # --------------------------------------------------------

    load_model()


    print("")


    # --------------------------------------------------------
    # ADRESA
    # --------------------------------------------------------

    print(
        "Web GUI:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print(
        "http://IP-E-PC:5000"
    )


    print("")


    # --------------------------------------------------------
    # START FLASK
    #
    # MOS përdor ssl_context këtu.
    #
    # Serveri është HTTP:
    # http://IP-E-PC:5000
    # --------------------------------------------------------

    app.run(

        host="0.0.0.0",

        port=5000,

        threaded=True,

        debug=False,

        use_reloader=False
    )