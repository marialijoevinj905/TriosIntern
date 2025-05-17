import cv2
import numpy as np
import pandas as pd
import argparse
from sklearn.cluster import DBSCAN
import os
import time
from datetime import datetime
from ultralytics import YOLO


class CrowdDetector:
    def __init__(self, conf_threshold=0.5, nms_threshold=0.4, min_crowd_size=3, min_consecutive_frames=10, proximity_threshold=100, model_path=None):
        
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.min_crowd_size = min_crowd_size
        self.min_consecutive_frames = min_consecutive_frames
        self.proximity_threshold = proximity_threshold
        self.model_path = model_path
        
        self.model = self._load_yolo_model()
    
        self.crowd_frames_count = {}  #{cluster_id: count}
        self.crowd_logs = []  #Store frame, count, timestamp for each crowd event
        self.current_frame = 0
        self.detected_crowds = set()
        
        #Frame-by-frame results for CSV
        self.frame_results = []  
        
    def _load_yolo_model(self):

        print("Loading YOLOv8 model...")
        
        if self.model_path and os.path.exists(self.model_path):
            model = YOLO(self.model_path)
            print(f"Loaded YOLOv8 model from: {self.model_path}")
        else:
            model = YOLO("yolov8n.pt")
            print("Downloaded YOLOv8n model from Ultralytics")
            
        return model
        
    def detect_persons(self, frame):

        results = self.model(frame, conf=self.conf_threshold, classes=0)  # class 0 is person
        
        # Process detections
        person_boxes = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates (x1, y1, x2, y2 format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Convert to (x, y, w, h) format
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                
                person_boxes.append([x, y, w, h])
                
        return person_boxes
        
    def cluster_persons(self, boxes):
        if len(boxes) < self.min_crowd_size:
            return {}
            
        # Calculate center points of each person
        centers = []
        for box in boxes:
            x, y, w, h = box
            center_x = x + w // 2
            center_y = y + h // 2
            centers.append([center_x, center_y])
            
        # Apply DBSCAN clustering
        centers = np.array(centers)
        clustering = DBSCAN(eps=self.proximity_threshold, min_samples=self.min_crowd_size).fit(centers)
        
        # Group person indices by cluster
        clusters = {}
        for idx, label in enumerate(clustering.labels_):
            if label >= 0:  # -1 means noise (not part of any cluster)
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(idx)
                
        return clusters
        
    def process_frame(self, frame):

        self.current_frame += 1
        frame_copy = frame.copy()
        
        # Detect persons
        person_boxes = self.detect_persons(frame)
        total_person_count = len(person_boxes)
        
        # Annotate persons
        for box in person_boxes:
            x, y, w, h = box
            # Draw red boxes around each person
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
        # Cluster persons to find potential crowds
        clusters = self.cluster_persons(person_boxes)
        
        current_clusters = set()
        crowd_person_count = 0
        
        for cluster_id, person_indices in clusters.items():
            crowd_size = len(person_indices)
            crowd_person_count += crowd_size
            
            avg_x = sum(person_boxes[i][0] + person_boxes[i][2]//2 for i in person_indices) / crowd_size
            avg_y = sum(person_boxes[i][1] + person_boxes[i][3]//2 for i in person_indices) / crowd_size
            cluster_key = f"{cluster_id}_{int(avg_x)}_{int(avg_y)}"
            
            current_clusters.add(cluster_key)
            if cluster_key not in self.crowd_frames_count:
                self.crowd_frames_count[cluster_key] = 0
            self.crowd_frames_count[cluster_key] += 1
            
            # Check if this is a persistent crowd
            if self.crowd_frames_count[cluster_key] >= self.min_consecutive_frames:
                # Mark this frame as having a crowd
                if cluster_key not in self.detected_crowds:
                    self.detected_crowds.add(cluster_key)
                    # Log the crowd event
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.crowd_logs.append({
                        "Frame": self.current_frame,
                        "PersonCount": crowd_size,
                        "Timestamp": timestamp
                    })
                    print(f"Crowd detected at frame {self.current_frame} with {crowd_size} people")
                
                # Draw a red rectangle around detection
                min_x = min(person_boxes[i][0] for i in person_indices)
                min_y = min(person_boxes[i][1] for i in person_indices)
                max_x = max(person_boxes[i][0] + person_boxes[i][2] for i in person_indices)
                max_y = max(person_boxes[i][1] + person_boxes[i][3] for i in person_indices)
                cv2.rectangle(frame_copy, (min_x, min_y), (max_x, max_y), (0, 0, 255), 3)
                cv2.putText(frame_copy, f"Crowd: {crowd_size} people", (min_x, min_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        keys_to_remove = [k for k in self.crowd_frames_count.keys() if k not in current_clusters]
        for k in keys_to_remove:
            del self.crowd_frames_count[k]
        
        # Add total person count to the frame
        cv2.putText(frame_copy, f"Total persons: {total_person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Store frame results for CSV
        self.frame_results.append({
            "Frame": self.current_frame,
            "PersonCount": total_person_count
        })
            
        return frame_copy
        
    def save_crowd_events(self, output_path="crowd_events.csv"):

        if self.crowd_logs:
            df = pd.DataFrame(self.crowd_logs)
            df.to_csv(output_path, index=False)
            print(f"Crowd events saved to {output_path}")
        else:
            print("No crowds detected, no crowd events to save.")
    
    def save_frame_results(self, output_path="frame_results.csv"):

        if self.frame_results:
            df = pd.DataFrame(self.frame_results)
            df.to_csv(output_path, index=False)
            print(f"Frame results saved to {output_path}")
        else:
            print("No frames processed, no results to save.")
            
    def process_video(self, video_path, output_path=None, results_csv_path="frame_results.csv"):
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Resolution: {width}x{height}")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process frame for crowd detection
            processed_frame = self.process_frame(frame)
            
            # Write frame to output video
            if writer:
                writer.write(processed_frame)
                
            # Display processed frame
            cv2.imshow("Crowd Detection", processed_frame)
            
            if frame_count % 10 == 0:
                elapsed_time = time.time() - start_time
                processing_fps = frame_count / elapsed_time
                cv2.putText(processed_frame, f"FPS: {processing_fps:.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")
                
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Save results to CSV files
        self.save_crowd_events()
        self.save_frame_results(results_csv_path)
        
        print(f"Processed {frame_count} frames in {time.time() - start_time:.2f} seconds")
        print(f"Detected {len(self.crowd_logs)} crowd events")
        

def main():

    #CLI
    parser = argparse.ArgumentParser(description="Detect crowds in video")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default=None, help="Path to save output video")
    parser.add_argument("--results_csv", type=str, default="frame_results.csv", 
                        help="Path to save frame results CSV")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for detection")
    parser.add_argument("--nms", type=float, default=0.4, help="NMS threshold for detection")
    parser.add_argument("--min_size", type=int, default=3, help="Minimum persons to be considered a crowd")
    parser.add_argument("--min_frames", type=int, default=10, help="Minimum consecutive frames for crowd")
    parser.add_argument("--proximity", type=float, default=100, help="Maximum distance between people in a crowd")
    parser.add_argument("--model", type=str, default=None, help="Path to YOLOv8 model file (optional)")
    
    args = parser.parse_args()
    
    # Create crowd detector
    detector = CrowdDetector(
        conf_threshold=args.conf,
        nms_threshold=args.nms,
        min_crowd_size=args.min_size,
        min_consecutive_frames=args.min_frames,
        proximity_threshold=args.proximity,
        model_path=args.model
    )
    
    # Process video
    detector.process_video(args.video, args.output, args.results_csv)


if __name__ == "__main__":
    main()