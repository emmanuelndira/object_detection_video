import argparse
import os
import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Object detection on video/webcam using YOLOv8 + OpenCV (spacebar to pause/resume, q to quit)."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/input.mp4",
        help="Video path (e.g. data/input.mp4) or webcam index (e.g. 0).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model weights (e.g. yolov8n.pt, yolov8s.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold (0 to 1). Higher = fewer detections.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (0 to 1).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated output video to outputs/annotated.mp4",
    )
    return parser.parse_args()


def release_resources(cap: cv2.VideoCapture, writer: cv2.VideoWriter | None):
    """Safely release OpenCV resources."""
    try:
        if cap is not None:
            cap.release()
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()

    # Ensure outputs folder exists
    os.makedirs("outputs", exist_ok=True)

    # Load pretrained YOLO model
    model = YOLO(args.model)

    # Source can be webcam index "0", "1", etc. or a filepath
    source = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0  # fallback if FPS is not reported

    writer = None
    out_path = "outputs/annotated.mp4"
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Rough detection counts (for learning + README)
    total_counts: dict[str, int] = {}

    window_name = "YOLOv8 Object Detection (SPACE=pause/resume, q=quit)"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference on this frame
            results = model.predict(frame, conf=args.conf, iou=args.iou, verbose=False)
            r0 = results[0]

            # Count detections by class (rough, counts per-frame detections)
            if r0.boxes is not None and len(r0.boxes) > 0:
                cls_ids = r0.boxes.cls.tolist()
                for cid in cls_ids:
                    name = model.names.get(int(cid), str(int(cid)))
                    total_counts[name] = total_counts.get(name, 0) + 1

            # Draw predictions (boxes + labels)
            annotated = r0.plot()

            # Show frame
            cv2.imshow(window_name, annotated)

            # Save output if requested
            if writer is not None:
                writer.write(annotated)

            # Key handling:
            # - q quits
            # - space pauses/resumes (frame stays on screen)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" "):
                # Paused: wait indefinitely for next key press
                while True:
                    key2 = cv2.waitKey(0) & 0xFF
                    if key2 == ord(" "):  # resume
                        break
                    if key2 == ord("q"):  # quit while paused
                        return

    finally:
        release_resources(cap, writer)

    # Print detection summary
    if total_counts:
        print("\nDetection counts (rough):")
        for k in sorted(total_counts.keys()):
            print(f"  {k}: {total_counts[k]}")
    else:
        print("\nNo detections recorded (try lowering --conf).")

    if args.save:
        print(f"\nSaved annotated video to: {out_path}")


if __name__ == "__main__":
    main()
