import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import CompressedImage
import threading
import numpy as np
import cv2
import os
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from train.utils import preprocess_image_lane
from train.utils import preprocess_image_object
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_eval
import math
import time
import argparse
from PIL import Image



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.Session(config=config)
K.set_session(sess)


class Drive(Node):
    def __init__(self):
        super().__init__('drive')

        self.image_lock = threading.RLock()

        # ROS communications
        self.image_sub = self.create_subscription(CompressedImage, '/simulator/sensor/camera/center/compressed', self.image_callback)
        # self.control_pub = self.create_publisher(TwistStamped, '/lanefollowing/steering_cmd')

        # ROS timer
        # self.timer_period = .02  # seconds
        # self.timer = self.create_timer(self.timer_period, self.publish_steering)

        # ROS parameters
        self.enable_visualization = self.get_parameter('visualization').value
        # self.model_path = self.get_parameter('model_path').value

        # model for object detection
        self.object_model_path = self.get_parameter('object_model_path').value

        # Model parameters
        # self.model = self.get_model(self.model_path)
        self.object_model = self.get_model(self.object_model_path)

        self.img = None
        self.steering = 0.

        # For visualizations
        self.steer_ratio = 16.
        self.steering_wheel_single_direction_max = 470.  # in degree (8.2 radian)
        self.wheel_base = 2.836747  # in meters
        self.smoothed_angle = 0.
        self.inference_time = 0.

        # FPS
        self.last_time = time.time()
        self.frames = 0
        self.fps = 0.

    def image_callback(self, img):
        self.get_fps()
        if self.image_lock.acquire(True):
            self.img = img
            # if self.model is None:
                # self.model = self.get_model(self.model_path)
            #t0 = time.time()
            # self.steering = self.predict(self.model, self.img)
            self.predict_objects(self.img)
            #t1 = time.time()
            #self.inference_time = t1 - t0
            if self.enable_visualization:
                self.visualize(self.img, self.steering)
            self.image_lock.release()

    def publish_steering(self):
        if self.img is None:
            return
        message = TwistStamped()
        message.twist.angular.x = float(self.steering)
        self.control_pub.publish(message)
        self.get_logger().info('[{:.3f}] Predicted steering command: "{}"'.format(time.time(), message.twist.angular.x))

    def get_model(self, model_path):
        self.get_logger().info('Model loading: {}'.format(model_path))
        model = load_model(model_path)
        self.get_logger().info('Model loaded: {}'.format(model_path))

        return model

    def predict(self, model, img):
        c = np.fromstring(bytes(img.data), np.uint8)
        img = cv2.imdecode(c, cv2.IMREAD_COLOR)
        img = preprocess_image_lane(img)
        img = np.expand_dims(img, axis=0)  # img = img[np.newaxis, :, :]
        steering = self.model.predict(img)

        return steering

    def predict_objects(self, img):
        #self.object_model.summary()
        c = np.fromstring(bytes(img.data), np.uint8)
        img = cv2.imdecode(c, cv2.IMREAD_COLOR)
        #im_pil
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        # weight height of the image_sub
        width, height = im_pil.size
        print(width)
        print(height)
        width = np.array(width, dtype=float)
        height = np.array(height, dtype=float)
        image_shape = (height, width)

        #class names and anchors
        class_names = read_classes("/lanefollowing/ros2_ws/src/lane_following/model/coco_classes.txt")
        anchors = read_anchors("/lanefollowing/ros2_ws/src/lane_following/model/yolo_anchors.txt")
        #yolo head
        yolo_outputs = yolo_head(self.object_model.output, anchors, len(class_names))

        # yolo eval
        boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)
        print('boxes')
        print(boxes)
        #yolo preprocess
        pil_img, img_data = preprocess_image_object(im_pil, model_image_size = (608, 608))
        #yolo predict
        # print('test1')
        out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={self.object_model.input:img_data,K.learning_phase(): 0})
        # print('test2')
        print('Found {} boxes for '.format(len(out_boxes)))
        colors = generate_colors(class_names)
        draw_boxes(pil_img, out_scores, out_boxes, out_classes, class_names, colors)
        #might need to add resizing here...
        image = np.asarray(pil_img)
        image = cv2.resize(image, (round(image.shape[1] / 2), round(image.shape[0] / 2)), interpolation=cv2.INTER_AREA)
        cv2.imshow('YOLO Detector', image)
        cv2.waitKey(1)
        # img = np.expand_dims(pil_img, axis=0)  # img = img[np.newaxis, :, :]
        # obj = self.object_model.predict(img)
        # print("Test")
        #print(obj)
        # self.get_logger().info(obj)

    def visualize(self, img, steering):
        c = np.fromstring(bytes(img.data), np.uint8)
        image = cv2.imdecode(c, cv2.IMREAD_COLOR)

        steering_wheel_angle_deg = steering * self.steering_wheel_single_direction_max
        wheel_angle_deg = steering_wheel_angle_deg / self.steer_ratio  # wheel angle in degree [-29.375, 29.375]
        curvature_radius = self.wheel_base / (2 - 2 * math.cos(2 * steering_wheel_angle_deg / self.steer_ratio)) ** 2

        kappa = 1 / curvature_radius
        curvature = int(kappa * 50)
        if steering < 0:  # Turn left
            x = -curvature
            ra = 0
            rb = -70
        else:  # Turn right
            x = curvature
            ra = -110
            rb = -180

        cv2.ellipse(image, (960 + x, image.shape[0]), (curvature, 500), 0, ra, rb, (0, 255, 0), 2)

        cv2.putText(image, "Prediction: %f.7" % (steering), (30, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.putText(image, "Steering wheel angle: %.3f degrees" % steering_wheel_angle_deg, (30, 120), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.putText(image, "Wheel angle: %.3f degrees" % wheel_angle_deg, (30, 170), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.putText(image, "Prediction time: %d ms" % (self.inference_time * 1000), (30, 220), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.putText(image, "Frame speed: %d fps" % (self.fps), (30, 270), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

        image = cv2.resize(image, (round(image.shape[1] / 2), round(image.shape[0] / 2)), interpolation=cv2.INTER_AREA)
        cv2.imshow('LGSVL End-to-End Lane Following', image)
        cv2.waitKey(1)

    def get_fps(self):
        self.frames += 1
        now = time.time()
        if now >= self.last_time + 1.0:
            delta = now - self.last_time
            self.last_time = now
            self.fps = self.frames / delta
            self.frames = 0


def main(args=None):
    rclpy.init(args=args)
    drive = Drive()
    rclpy.spin(drive)


if __name__ == '__main__':
    main()
